"""Model wrappers: HuggingFace causal LMs + a dependency-free dummy model.

The key method is ``logprobs(text)``.  It returns the per-token conditional
log-probabilities (TPS) plus two compact per-token summaries used by the
baselines: the rank of the realised token and the predictive entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import List, Optional

import numpy as np

try:  # torch/transformers are optional for the math smoke test
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class ModelOutput:
    """Per-token statistics for one text."""

    tps: np.ndarray                 # (n,) log p(t_i | t_<i)
    ranks: Optional[np.ndarray] = None   # (n,) rank of realised token (1 = top)
    entropies: Optional[np.ndarray] = None  # (n,) predictive entropy

    @property
    def n_tokens(self) -> int:
        return int(self.tps.size)


class DummyModel:
    """Deterministic synthetic model for plumbing tests (no torch needed)."""

    def __init__(self, name: str = "dummy", seed: int = 0, vocab: int = 128):
        self.name = name
        self.seed = seed
        self.vocab = vocab

    def logprobs(self, text: str) -> ModelOutput:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little") + self.seed)
        n = max(8, min(64, len(text.split()) + 3))
        logits = rng.normal(0.0, 1.0, (n, self.vocab))
        target_ids = rng.integers(0, self.vocab, n)
        # tiny heuristic so the dummy run produces AUROC > 0.5:
        # shorter texts (the LLM side in tiny_pairs) get slightly higher target logits
        if len(text) < 240:
            logits[np.arange(n), target_ids] += 0.9
        # log-softmax row-wise
        m = logits.max(axis=1, keepdims=True)
        z = logits - m
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        tps = logp[np.arange(n), target_ids]
        target_logits = logits[np.arange(n), target_ids][:, None]
        ranks = (logits > target_logits).sum(axis=1) + 1
        entropies = -(np.exp(logp) * logp).sum(axis=1)
        return ModelOutput(tps=tps, ranks=ranks, entropies=entropies)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> List[str]:
        return [f"{prompt} [dummy sample {i}]" for i in range(n)]


class HFModel:
    """Causal LM wrapper that extracts TPS with a single forward pass."""

    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch/transformers are not installed. "
                "Run `pip install -r requirements.txt` first."
            )
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.float16
            if dtype == "float16" and self.device.startswith("cuda")
            else torch.float32
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=trust_remote_code
            )
        except Exception:
            # mGPT-style repos ship vocab.json + merges.txt but no tokenizer.json;
            # GPT2TokenizerFast can build the fast backend directly.
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _encode(self, text: str):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        start_id = self.tokenizer.bos_token_id
        if start_id is None:
            start_id = self.tokenizer.eos_token_id
        if start_id is not None:
            bos = torch.full(
                (1, 1), int(start_id), dtype=ids.dtype, device=self.device
            )
            ids = torch.cat([bos, ids], dim=1)
        return ids

    def logprobs(self, text: str) -> ModelOutput:
        ids = self._encode(text)
        # GPT-2 has max_position_embeddings=1024; XSum documents can be longer.
        max_len = getattr(self.model.config, "max_position_embeddings", None)
        if max_len and ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        target = ids[0, 1:]
        target_logp = logp[:-1].gather(1, target.unsqueeze(1)).squeeze(1)
        target_logits = logits[:-1].gather(1, target.unsqueeze(1))
        ranks = (logits[:-1] > target_logits).sum(dim=1) + 1
        probs = logp[:-1].exp()
        entropies = -(probs * logp[:-1]).sum(dim=1)
        return ModelOutput(
            tps=target_logp.cpu().numpy().astype(np.float64),
            ranks=ranks.cpu().numpy().astype(np.int64),
            entropies=entropies.cpu().numpy().astype(np.float64),
        )

    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> List[str]:
        max_len = getattr(self.model.config, "max_position_embeddings", 1024)
        max_prompt = max(8, int(max_len) - int(max_new_tokens))
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt,
        ).to(self.device)
        kwargs = {}
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
                **kwargs,
            )
        new_tokens = out[:, enc.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def load_model(
    name: str,
    device: Optional[str] = None,
    dtype: str = "float16",
    seed: int = 0,
    trust_remote_code: bool = False,
):
    if name in ("dummy", "dummy-model"):
        return DummyModel(name=name, seed=seed)
    return HFModel(
        name, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )

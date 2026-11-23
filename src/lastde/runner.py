"""Experiment runner: config -> scores -> AUROC -> CSV/JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .baselines import detectlrr, entropy, likelihood, logrank, sampling_discrepancy
from .cache import NullCache, OutputCache
from .core import eps_for, lastde_plus_plus, lastde_score
from .data import load_pairs
from .eval import evaluate_rows, summary_dataframe
from .models import ModelOutput, load_model

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


SAMPLE_METHODS = {"lastde_pp", "fast_detectgpt"}


def _lastde_cfg(config: Dict, key: str = "lastde") -> Dict:
    cfg = dict(config.get(key, {}))
    cfg.setdefault("s", 3)
    cfg.setdefault("eps_factor", 10.0)
    cfg.setdefault("tau_prime", 5)
    cfg.setdefault("agg", "std")
    return cfg


def score_methods(
    out: ModelOutput,
    samples: Optional[List[ModelOutput]],
    config: Dict,
) -> Dict[str, float]:
    methods = config["methods"]
    scores: Dict[str, float] = {}

    if "lastde" in methods:
        cfg = _lastde_cfg(config, "lastde")
        eps = cfg.get("eps") or eps_for(
            out.n_tokens, cfg["eps_factor"], minimum=10
        )
        scores["lastde"] = lastde_score(
            out.tps,
            s=int(cfg["s"]),
            eps=int(eps),
            tau_prime=int(cfg["tau_prime"]),
            agg=cfg["agg"],
        ).score

    if "lastde_pp" in methods:
        cfg = _lastde_cfg(config, "lastde_pp")
        cfg.setdefault("n_samples", 100)
        eps = cfg.get("eps") or eps_for(
            out.n_tokens, cfg["eps_factor"], minimum=10
        )
        z, _, _ = lastde_plus_plus(
            out.tps,
            [s.tps for s in (samples or [])],
            s=int(cfg["s"]),
            eps=int(eps),
            tau_prime=int(cfg["tau_prime"]),
            agg=cfg["agg"],
        )
        scores["lastde_pp"] = z

    if "likelihood" in methods:
        scores["likelihood"] = likelihood(out)
    if "ll_only" in methods:
        # ablation: global LL only (same statistic as Likelihood)
        scores["ll_only"] = likelihood(out)
    if "mde_only" in methods:
        # ablation: local Agg-MDE only; negate so higher = more AI-like
        cfg = _lastde_cfg(config, "lastde")
        eps = cfg.get("eps") or eps_for(
            out.n_tokens, cfg["eps_factor"], minimum=10
        )
        res = lastde_score(
            out.tps,
            s=int(cfg["s"]),
            eps=int(eps),
            tau_prime=int(cfg["tau_prime"]),
            agg=cfg["agg"],
        )
        scores["mde_only"] = -res.agg_mde
    if "logrank" in methods:
        scores["logrank"] = logrank(out)
    if "entropy" in methods:
        scores["entropy"] = entropy(out)
    if "detectlrr" in methods:
        scores["detectlrr"] = detectlrr(out)
    if "fast_detectgpt" in methods:
        scores["fast_detectgpt"] = sampling_discrepancy(out, samples or [])
    return scores


def _sample_config(config: Dict) -> Dict:
    cfg = dict(config.get("lastde_pp", {}))
    cfg.update(config.get("fast_detectgpt", {}))
    cfg.setdefault("n_samples", 100)
    cfg.setdefault("max_new_tokens", 200)
    cfg.setdefault("temperature", 1.0)
    cfg.setdefault("top_p", 0.95)
    cfg.setdefault("top_k", None)
    return cfg


def run(
    config: Dict,
    limit: Optional[int] = None,
    use_cache: bool = True,
    device: Optional[str] = None,
) -> Dict:
    run_name = config.get("run_name", "run")
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = (
        OutputCache(config.get("cache_dir", "cache"))
        if use_cache
        else NullCache()
    )
    methods = list(config["methods"])
    rows: List[Dict] = []

    for data_cfg in config["data"]:
        pairs = load_pairs(data_cfg["path"])
        cap = limit if limit is not None else data_cfg.get("limit")
        if cap:
            pairs = pairs[: int(cap)]
        for model_cfg in config["models"]:
            white_box = model_cfg.get(
                "white_box", model_cfg["source"] == model_cfg["proxy"]
            )
            if white_box:
                # White-box: one model plays both roles.
                source = load_model(
                    model_cfg["source"],
                    device=device,
                    dtype=model_cfg.get("dtype", "float16"),
                    seed=config.get("seed", 0),
                )
                proxy = source
                scorer = source
            else:
                # Black-box: the source model is NOT needed at scoring time.
                # It was only needed by build_pairs.py to generate the text.
                # Loading it here would double VRAM for no reason.
                source = None
                scorer = load_model(
                    model_cfg["proxy"],
                    device=device,
                    dtype=model_cfg.get("dtype", "float16"),
                    seed=config.get("seed", 0),
                )
                proxy = scorer
            model_name = model_cfg.get("name", f"{model_cfg['source']}->{model_cfg['proxy']}")

            for pair in tqdm(pairs, desc=f"{data_cfg['name']}/{model_name}"):
                for label, text in (("human", pair.human), ("llm", pair.llm)):
                    if not text or not text.strip():
                        continue

                    # Lastde++ / Fast-DetectGPT: sample conditioned on THIS text,
                    # not on the shared 30-token prompt (paper Eq. 11 uses t).
                    samples: Optional[List[ModelOutput]] = None
                    if SAMPLE_METHODS.intersection(methods):
                        sc = _sample_config(config)
                        sampling_text = pair.prompt if getattr(pair, "prompt", "") else text
                        texts = proxy.generate(
                            sampling_text,
                            n=int(sc["n_samples"]),
                            max_new_tokens=int(sc["max_new_tokens"]),
                            temperature=float(sc["temperature"]),
                            top_p=float(sc["top_p"]),
                            top_k=sc.get("top_k"),
                        )
                        samples = []
                        for t in texts:
                            if not t or not t.strip():
                                continue
                            out_s = cache.get(
                                proxy.name, t, lambda t=t: proxy.logprobs(t)
                            )
                            # a sample that tokenizes to 0 tokens would give empty TPS
                            if out_s.tps.size > 0:
                                samples.append(out_s)

                    out = cache.get(
                        scorer.name, text, lambda t=text: scorer.logprobs(t)
                    )
                    if out.tps.size == 0:
                        continue
                    scores = score_methods(out, samples, config)
                    for method, value in scores.items():
                        rows.append(
                            {
                                "dataset": data_cfg["name"],
                                "model": model_name,
                                "pair_id": pair.id,
                                "label": label,
                                "method": method,
                                "score": float(value),
                                "n_tokens": out.n_tokens,
                            }
                        )

    with open(out_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = evaluate_rows(
        rows,
        n_boot=int(config.get("n_boot", 500)),
        seed=int(config.get("seed", 0)),
    )
    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
        print("\nAUROC summary")
        print(summary_dataframe(summary).to_string())
    else:
        print("No summary produced (need >=2 pairs per class).")
    return {"rows": rows, "summary": summary, "out_dir": str(out_dir)}


def load_config(path: str | Path) -> Dict:
    import yaml  # lazy: only needed when loading a YAML file

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

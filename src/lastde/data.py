"""Dataset helpers: JSONL pairs, prefix splitting, pair construction."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class Pair:
    """One human/LLM pair that shares the same prompt."""

    id: str
    prompt: str
    human: str
    llm: str
    meta: Dict = field(default_factory=dict)


def read_jsonl(path: str | Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pairs(path: str | Path) -> List[Pair]:
    pairs: List[Pair] = []
    for i, row in enumerate(read_jsonl(path)):
        pairs.append(
            Pair(
                id=str(row.get("id", i)),
                prompt=row.get("prompt", ""),
                human=row["human"],
                llm=row["llm"],
                meta={k: v for k, v in row.items() if k not in ("id", "prompt", "human", "llm")},
            )
        )
    return pairs


def write_pairs(path: str | Path, pairs: Sequence[Pair]) -> None:
    write_jsonl(
        path,
        (
            {
                "id": p.id,
                "prompt": p.prompt,
                "human": p.human,
                "llm": p.llm,
                **(p.meta or {}),
            }
            for p in pairs
        ),
    )


def split_prefix(text: str, tokenizer=None, n_tokens: int = 30) -> tuple[str, str]:
    """Split a human text into (prompt, continuation).

    If a HF tokenizer is given, split on token ids; otherwise split on whitespace.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefix_ids, rest_ids = ids[:n_tokens], ids[n_tokens:]
        return (
            tokenizer.decode(prefix_ids, skip_special_tokens=True),
            tokenizer.decode(rest_ids, skip_special_tokens=True),
        )
    words = text.split()
    return " ".join(words[:n_tokens]), " ".join(words[n_tokens:])


def build_pairs(
    human_rows: Iterable[Dict],
    llm_model,
    tokenizer=None,
    prefix_tokens: int = 30,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    text_field: str = "text",
) -> List[Pair]:
    """Build paired data: original human continuation vs model continuation.

    For each human document:
      - first `prefix_tokens` tokens -> prompt
      - the rest of the document  -> human continuation
      - the same prompt fed to llm_model -> LLM continuation
    """
    pairs: List[Pair] = []
    for i, row in enumerate(human_rows):
        text = row[text_field]
        prompt, human = split_prefix(text, tokenizer=tokenizer, n_tokens=prefix_tokens)
        if tokenizer is not None:
            human_ids = tokenizer.encode(human, add_special_tokens=False)[:max_new_tokens]
            human = tokenizer.decode(human_ids, skip_special_tokens=True)
        else:
            human = " ".join(human.split()[:max_new_tokens])
        if not human.strip():
            continue
        llm_text = llm_model.generate(
            prompt,
            n=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]
        if not llm_text.strip():
            # GPT-2 can emit EOS immediately; skip this pair.
            continue
        pairs.append(
            Pair(
                id=str(row.get("id", i)),
                prompt=prompt,
                human=human,
                llm=llm_text,
                meta={
                    "source": row.get("source", ""),
                    "prefix_tokens": prefix_tokens,
                },
            )
        )
    return pairs

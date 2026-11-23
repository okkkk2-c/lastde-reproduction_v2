"""Load the paper's datasets from HuggingFace and export human-text JSONL.

Public datasets used by the paper:
  XSum, SQuAD, WritingPrompts, Reddit ELI5, WMT16 (en/de)
Zhihu is not public; put a local `data/human/zhihu.jsonl` there yourself.

If a HF dataset name/split changes, edit SPECS below.  Every loader is
deliberately small so it is easy to adapt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import hashlib
import json


@dataclass
class DatasetSpec:
    key: str
    path: str
    split: str
    text_field: str
    config: Optional[str] = None
    lang: Optional[str] = None
    min_words: int = 150
    concat: bool = False  # WMT16 is sentence-level: concatenate into documents


SPECS: Dict[str, DatasetSpec] = {
    "xsum": DatasetSpec("xsum", "EdinburghNLP/xsum", "test", "document"),
    "squad": DatasetSpec(
        "squad", "rajpurkar/squad", "train", "context", config="plain_text"
    ),
    "writingprompts": DatasetSpec(
        "writingprompts", "euclaise/writingprompts", "train", "story"
    ),
    "reddit": DatasetSpec(
        "reddit", "eli5", "train_eli5", "answers", config="all"
    ),
    "wmt16_de": DatasetSpec(
        "wmt16_de",
        "wmt/wmt16",
        "train",
        "translation",
        config="de-en",
        lang="de",
        concat=True,
    ),
    "wmt16_en": DatasetSpec(
        "wmt16_en",
        "wmt/wmt16",
        "train",
        "translation",
        config="de-en",
        lang="en",
        concat=True,
    ),
}


def _extract_text(row: Dict, spec: DatasetSpec) -> str:
    if spec.key == "reddit":
        answers = row.get("answers") or []
        if isinstance(answers, list) and answers:
            best = max(
                answers,
                key=lambda a: a.get("score", 0) if isinstance(a, dict) else 0,
            )
            if isinstance(best, dict):
                return best.get("text", "") or ""
        return row.get("selftext", "") or ""
    if spec.key.startswith("wmt16"):
        tr = row.get(spec.text_field)
        if isinstance(tr, dict):
            return tr.get(spec.lang or "en", "") or ""
        return ""
    text = row.get(spec.text_field, "")
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    return text or ""


def load_human_texts(
    spec: DatasetSpec,
    limit: int = 150,
    min_words: Optional[int] = None,
    seed: int = 0,
) -> List[Dict]:
    """Download a HF dataset and return [{id, text, source}, ...]."""
    from datasets import load_dataset

    min_words = spec.min_words if min_words is None else min_words
    ds = (
        load_dataset(spec.path, spec.config, split=spec.split)
        if spec.config
        else load_dataset(spec.path, split=spec.split)
    )
    ds = ds.shuffle(seed=seed)
    out: List[Dict] = []
    seen = set()

    def _add(text: str, i: int) -> bool:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            return False
        seen.add(h)
        out.append({"id": f"{spec.key}-{i}", "text": text, "source": spec.key})
        return len(out) >= limit

    if spec.concat:
        # WMT16 is sentence-level; concatenate consecutive pairs into documents.
        buf: List[str] = []
        words = 0
        for i, row in enumerate(ds):
            text = _extract_text(row, spec).strip()
            if not text:
                continue
            buf.append(text)
            words += len(text.split())
            if words >= min_words:
                if _add(" ".join(buf), i):
                    break
                buf, words = [], 0
        return out

    for i, row in enumerate(ds):
        text = _extract_text(row, spec).strip()
        if len(text.split()) < min_words:
            continue
        if _add(text, i):
            break
    return out


def write_human_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

#!/usr/bin/env python
"""Download the paper's public datasets and export human-text JSONL.

Datasets:
  xsum, squad, writingprompts, reddit (ELI5), wmt16_de, wmt16_en

Zhihu is not public; put `data/human/zhihu.jsonl` there manually if you have it.

Examples:
  python scripts/prepare_datasets.py --datasets all
  python scripts/prepare_datasets.py --datasets xsum,squad --limit 150
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.datasets import SPECS, load_human_texts, write_human_jsonl  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


PUBLIC = ["xsum", "squad", "writingprompts", "reddit", "wmt16_de", "wmt16_en"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma separated keys, or 'all' for the public datasets",
    )
    parser.add_argument("--out-dir", default="data/human")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--min-words", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keys = PUBLIC if args.datasets == "all" else [k.strip() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in SPECS:
            print(f"[skip] unknown dataset: {key}")
            continue
        spec = SPECS[key]
        print(f"[{key}] {spec.path} {spec.config or ''} split={spec.split}")
        rows = load_human_texts(
            spec, limit=args.limit, min_words=args.min_words, seed=args.seed
        )
        out = Path(args.out_dir) / f"{key}.jsonl"
        write_human_jsonl(out, rows)
        print(f"  wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate human/LLM pairs for each (dataset, source model).

Input : data/human/{dataset}.jsonl   (from scripts/prepare_datasets.py)
Output: data/pairs/{dataset}/{safe_model}.jsonl

Example:
  python scripts/build_pairs.py \
    --datasets xsum,squad,writingprompts \
    --models gpt2,facebook/opt-2.7b \
    --limit 150
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.data import build_pairs, read_jsonl, write_pairs  # noqa: E402
from lastde.models import load_model  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="comma separated keys")
    parser.add_argument("--models", required=True, help="comma separated HF names")
    parser.add_argument("--human-dir", default="data/human")
    parser.add_argument("--out-dir", default="data/pairs")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--prefix-tokens", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    for model_name in models:
        model = load_model(model_name, device=args.device)
        for key in datasets:
            human_path = Path(args.human_dir) / f"{key}.jsonl"
            if not human_path.exists():
                print(f"[skip] missing {human_path}")
                continue
            out_path = Path(args.out_dir) / key / f"{safe_name(model_name)}.jsonl"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] exists {out_path}")
                continue
            rows = list(read_jsonl(human_path))[: args.limit]
            pairs = []
            for row in tqdm(rows, desc=f"{key}/{safe_name(model_name)}"):
                pairs.extend(
                    build_pairs(
                        [row],
                        model,
                        tokenizer=getattr(model, "tokenizer", None),
                        prefix_tokens=args.prefix_tokens,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                )
            write_pairs(out_path, pairs)
            print(f"  wrote {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()

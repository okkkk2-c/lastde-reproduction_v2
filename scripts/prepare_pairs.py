#!/usr/bin/env python
"""Build paired human/LLM JSONL from a human-text JSONL.

Input rows need a `text` field (and optionally `id`).
For each document: first N tokens = prompt; the rest = human continuation;
the same prompt is fed to the source model to generate the LLM continuation.

Example:
  python scripts/prepare_pairs.py \
    --input data/human_xsum.jsonl \
    --output data/xsum_pairs.jsonl \
    --model gpt2 \
    --limit 150 --prefix-tokens 30 --max-new-tokens 200
"""
from __future__ import annotations

import argparse
from pathlib import Path
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="human JSONL with `text`")
    parser.add_argument("--output", required=True, help="output pairs JSONL")
    parser.add_argument("--model", required=True, help="HF model name, e.g. gpt2")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--prefix-tokens", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))[: args.limit]
    model = load_model(args.model, device=args.device)
    pairs = []
    for row in tqdm(rows, desc="building pairs"):
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
                text_field=args.text_field,
            )
        )
    write_pairs(args.output, pairs)
    print(f"wrote {len(pairs)} pairs -> {args.output}")


if __name__ == "__main__":
    main()

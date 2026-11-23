#!/usr/bin/env python
"""Create a paraphrased variant of a pairs JSONL (T5-Paraphraser).

This mirrors the paper's paraphrasing attack.  It is optional and heavy:
the first run downloads the T5 paraphraser.

Example:
  python scripts/paraphrase_pairs.py --input data/xsum_pairs.jsonl \
      --output data/xsum_pairs_paraphrased.jsonl --limit 150
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.data import load_pairs, write_pairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Vamsi/T5_Paraphrase_Paws")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU")
    args = parser.parse_args()

    from transformers import pipeline

    pipe = pipeline(
        "text2text-generation",
        model=args.model,
        device=args.device,
    )

    pairs = load_pairs(args.input)[: args.limit]

    def rewrite(texts):
        prompts = ["paraphrase: " + t for t in texts]
        outs = pipe(
            prompts,
            max_length=256,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1,
            batch_size=args.batch_size,
        )
        return [o["generated_text"] for o in outs]

    for p in pairs:
        p.human, p.llm = rewrite([p.human, p.llm])
    write_pairs(args.output, pairs)
    print(f"wrote {len(pairs)} paraphrased pairs -> {args.output}")


if __name__ == "__main__":
    main()

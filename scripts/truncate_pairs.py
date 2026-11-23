#!/usr/bin/env python
"""Create a length-controlled variant of a pairs JSONL (for the length study).

Example:
  python scripts/truncate_pairs.py --input data/xsum_pairs.jsonl \
      --output data/xsum_pairs_120w.jsonl --words 120
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.data import load_pairs, write_pairs  # noqa: E402


def truncate(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--words", type=int, required=True)
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    for p in pairs:
        p.human = truncate(p.human, args.words)
        p.llm = truncate(p.llm, args.words)
    write_pairs(args.output, pairs)
    print(f"wrote {len(pairs)} pairs ({args.words} words) -> {args.output}")


if __name__ == "__main__":
    main()

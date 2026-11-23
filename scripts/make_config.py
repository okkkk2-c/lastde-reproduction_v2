#!/usr/bin/env python
"""Generate one runner YAML per source model.

Because the LLM continuation depends on the source model, pairs live at
data/pairs/{dataset}/{model}.jsonl.  This script creates one config per
model so the runner never mixes pairs across models.

Example (white-box):
  python scripts/make_config.py \
    --datasets xsum,squad,writingprompts \
    --models gpt2,facebook/opt-2.7b \
    --mode white --out-dir configs/generated_white

Example (black-box, fixed GPT-J proxy):
  python scripts/make_config.py \
    --datasets xsum,writingprompts,reddit \
    --models gpt2,facebook/opt-2.7b \
    --mode black --proxy EleutherAI/gpt-j-6B \
    --out-dir configs/generated_black
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--mode", choices=["white", "black"], default="white")
    parser.add_argument("--proxy", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--out-dir", default="configs/generated")
    parser.add_argument("--run-prefix", default="main")
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        sm = safe_name(model)
        data = [
            {
                "name": key,
                "path": f"{args.pairs_dir}/{key}/{sm}.jsonl",
                "limit": args.limit,
            }
            for key in datasets
        ]
        cfg = {
            "run_name": f"{args.run_prefix}_{args.mode}_{sm}",
            "seed": 0,
            "cache_dir": "cache",
            "n_boot": 1000,
            "data": data,
            "models": [
                {
                    "name": f"{sm}_{args.mode}",
                    "source": model,
                    "proxy": model if args.mode == "white" else args.proxy,
                    "white_box": args.mode == "white",
                }
            ],
            "methods": [
                "lastde",
                "lastde_pp",
                "likelihood",
                "logrank",
                "entropy",
                "detectlrr",
                "fast_detectgpt",
            ],
            "lastde": {
                "s": 3,
                "eps_factor": 10,
                "tau_prime": 5,
                "agg": "std",
            },
            "lastde_pp": {
                "s": 4,
                "eps_factor": 8,
                "tau_prime": 15,
                "agg": "std",
                "n_samples": 100,
                "max_new_tokens": 200,
                "temperature": 1.0,
                "top_p": 0.95,
            },
            "fast_detectgpt": {"n_samples": 100, "max_new_tokens": 200},
        }
        path = out_dir / f"{sm}_{args.mode}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"wrote {path}")

    print("\nRun with:")
    for model in models:
        print(
            f"  python scripts/run_experiment.py --config "
            f"{out_dir / (safe_name(model) + '_' + args.mode + '.yaml')}"
        )


if __name__ == "__main__":
    main()

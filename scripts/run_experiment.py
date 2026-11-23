#!/usr/bin/env python
"""Run a configured reproduction experiment.

Examples
--------
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/tiny_gpt2.yaml --limit 6
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastde.runner import load_config, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--limit", type=int, default=None, help="cap pairs per dataset")
    parser.add_argument("--no-cache", action="store_true", help="disable TPS cache")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"run_name = {config.get('run_name')}")
    print(f"config   = {args.config}")
    print(f"methods  = {config['methods']}")
    run(
        config,
        limit=args.limit,
        use_cache=not args.no_cache,
        device=args.device,
    )


if __name__ == "__main__":
    main()

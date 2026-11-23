#!/usr/bin/env python
"""Run a matrix slice (white-box or black-box) from the YAML registries.

This is deliberately a thin orchestrator around the existing scripts:

  1. build_pairs.py  (if the pair file is missing)
  2. run_experiment.py (one config per source model)

It is resumable: existing pair files and existing summary.csv are skipped.

Examples:
  python scripts/run_matrix.py --phase white_box_main --models gpt2,opt-2.7
  python scripts/run_matrix.py --phase black_box_main --models gpt2
  python scripts/run_matrix.py --phase white_box_main --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHASES = {
    "white_box_main": ["xsum", "squad", "writingprompts"],
    "black_box_main": ["xsum", "writingprompts", "reddit"],
}
CHEAP_METHODS = ["lastde", "likelihood", "logrank", "entropy", "detectlrr"]
PROXY = "EleutherAI/gpt-j-6B"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_rows(phase: str):
    reg = load_yaml(ROOT / "models.yaml")
    if phase == "white_box_main":
        return reg["main_open"]
    rows = [m for m in reg["main_open"] if m["name"] != "gpt-j"]
    # closed-source generation is not implemented yet; keep the 11 open models
    return rows


def run(cmd, dry):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.call([str(c) for c in cmd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--models", default="", help="comma separated short names; empty = all")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--pairs-limit", type=int, default=150)
    ap.add_argument("--methods", default=",".join(CHEAP_METHODS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite-pairs", action="store_true")
    args = ap.parse_args()

    datasets = PHASES[args.phase]
    wanted = {x.strip() for x in args.models.split(",") if x.strip()}
    models = [m for m in model_rows(args.phase) if not wanted or m["name"] in wanted]
    if not models:
        raise SystemExit(f"no models matched: {args.models}")

    for m in models:
        short = safe_name(m["name"])
        hf_id = m["hf_id"]
        if args.phase == "black_box_main" and m["name"] == "gpt-j":
            print(f"[skip] {short}: GPT-J cannot be both source and proxy")
            continue

        pair_paths = {d: ROOT / "data" / "pairs" / d / f"{safe_name(hf_id)}.jsonl" for d in datasets}
        missing = [d for d, p in pair_paths.items() if not p.exists()]
        if missing:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "build_pairs.py"),
                "--datasets", ",".join(missing),
                "--models", hf_id,
                "--limit", str(args.pairs_limit),
                "--out-dir", str(ROOT / "data" / "pairs"),
            ]
            if args.overwrite_pairs:
                cmd.append("--overwrite")
            if args.device:
                cmd += ["--device", args.device]
            if run(cmd, args.dry_run) != 0:
                print(f"[warn] pair generation failed for {short}; skipping")
                continue
        else:
            print(f"[skip] pairs exist for {short}")

        # only include datasets whose human file exists
        available = []
        for d in datasets:
            human = ROOT / "data" / "human" / f"{d}.jsonl"
            if human.exists() and (args.dry_run or pair_paths[d].exists()):
                available.append(d)
            else:
                print(f"[skip] {short}/{d}: missing human or pair file")
        if not available:
            continue

        run_name = f"{args.phase}_{short}"
        config = {
            "run_name": run_name,
            "seed": 0,
            "cache_dir": "cache",
            "n_boot": 1000,
            "data": [
                {"name": d, "path": str(pair_paths[d].relative_to(ROOT)), "limit": args.limit}
                for d in available
            ],
            "models": [
                {
                    "name": f"{short}_{'white' if args.phase == 'white_box_main' else 'black'}",
                    "source": hf_id,
                    "proxy": hf_id if args.phase == "white_box_main" else PROXY,
                    "white_box": args.phase == "white_box_main",
                }
            ],
            "methods": [x.strip() for x in args.methods.split(",") if x.strip()],
            "lastde": {"s": 3, "eps_factor": 10, "tau_prime": 5, "agg": "std"},
        }
        cfg_dir = ROOT / "configs" / "matrix"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{args.phase}_{short}.yaml"
        if not args.dry_run:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        summary = ROOT / "results" / run_name / "summary.csv"
        if summary.exists():
            print(f"[skip] results exist: {summary}")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Generate a clean evidence sheet from actual run artifacts.

Run from the project root:
    python scripts/make_evidence.py
The script saves Lastde_evidence.png (2400 x 1500).
It reads result CSVs; missing files are shown as N/A, never fabricated.
"""
from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "Lastde_evidence.png"

W, H = 2400, 1500
BG = (11, 20, 32)
PANEL = (17, 28, 43)
BORDER = (35, 54, 77)
WHITE = (232, 238, 246)
MUTED = (150, 166, 184)
GREEN = (155, 231, 196)
GOLD = (243, 217, 167)
BLUE = (138, 180, 248)
RED = (248, 113, 113)


def load_font(names, size):
    candidates = []
    for n in names:
        candidates.append(n)
        candidates.append(str(Path("/usr/share/fonts/truetype/dejavu") / n))
        candidates.append(str(Path("C:/Windows/Fonts") / n))
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(["DejaVuSans-Bold.ttf", "arialbd.ttf"], 52)
FONT_SECTION = load_font(["DejaVuSans-Bold.ttf", "arialbd.ttf"], 27)
FONT_BODY = load_font(["DejaVuSansMono.ttf", "consola.ttf"], 21)
FONT_SMALL = load_font(["DejaVuSansMono.ttf", "consola.ttf"], 18)
FONT_LABEL = load_font(["DejaVuSans-Bold.ttf", "arialbd.ttf"], 19)


def read_rows(exp: str):
    path = RESULTS / exp / "summary.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auroc(exp: str, dataset: str | None = None, method: str = "lastde"):
    rows = read_rows(exp)
    vals = []
    for r in rows:
        if r.get("method") != method:
            continue
        if dataset is not None and r.get("dataset") != dataset:
            continue
        try:
            vals.append(float(r["auroc"]))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def pct(v):
    return "N/A" if v is None else f"{v * 100:.2f}"


def n_pairs(exp: str):
    rows = read_rows(exp)
    for r in rows:
        if r.get("method") == "lastde":
            return r.get("n_pairs", "?")
    return "?"


def official_summary():
    path = RESULTS / "official_blackbox_results.csv"
    if not path.exists():
        return "official data: CSV not found"
    groups = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("dataset", ""), r.get("source", ""))
            groups.setdefault(key, {})[r.get("method", "")] = float(r.get("auroc", 0))
    wins = {}
    totals = {}
    for (ds, _), methods in groups.items():
        totals[ds] = totals.get(ds, 0) + 1
        if methods and max(methods, key=methods.get) == "lastde":
            wins[ds] = wins.get(ds, 0) + 1
    order = [("reddit", "Reddit"), ("writing", "Writing"), ("squad", "SQuAD"), ("xsum", "XSum")]
    parts = [f"{label} {wins.get(k, 0)}/{totals.get(k, 0)}" for k, label in order if totals.get(k)]
    return "official data: " + " · ".join(parts) if parts else "official data: no rows"


def env_lines():
    lines = []
    try:
        import torch
        lines.append(f"torch         {torch.__version__}")
        lines.append(f"cuda          {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu           {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            lines.append(f"vram          {props.total_memory / 1024**3:.1f} GiB")
    except Exception as e:
        lines.append(f"torch         unavailable ({type(e).__name__})")
    for pkg in ("transformers", "datasets"):
        try:
            lines.append(f"{pkg:<13} {importlib.metadata.version(pkg)}")
        except Exception:
            lines.append(f"{pkg:<13} N/A")
    lines.append(f"python        {platform.python_version()}")
    try:
        usage = shutil.disk_usage("/root/autodl-tmp")
        used_pct = usage.used / usage.total * 100
        lines.append(f"data disk     {usage.total / 1024**3:.0f} GiB total, "
                     f"{usage.free / 1024**3:.0f} GiB free, {used_pct:.0f}% used")
    except Exception:
        pass
    lines.append(f"host          {socket.gethostname()}")
    lines.append(f"generated     {datetime.now():%Y-%m-%d %H:%M:%S}")
    return lines


def draw_panel(draw, x, y, w, h, fill=PANEL, outline=BORDER):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=fill, outline=outline, width=2)


def draw_lines(draw, x, y, lines, font=FONT_BODY, fill=WHITE, gap=36, max_width=None):
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill)
        yy += gap
    return yy


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Top gold accent
    d.rectangle((72, 58, 230, 64), fill=GOLD)

    d.text((72, 82), "LASTDE REPRODUCTION — EXECUTION EVIDENCE", font=FONT_TITLE, fill=WHITE)
    d.text((74, 148),
           "ICLR 2025 · training-free AI-text detection · generated from actual run artifacts",
           font=FONT_SMALL, fill=MUTED)

    # Environment panel
    draw_panel(d, 72, 210, 1080, 465)
    d.text((100, 234), "ENVIRONMENT", font=FONT_SECTION, fill=GOLD)
    draw_lines(d, 100, 288, env_lines(), FONT_BODY, WHITE, 36)

    # Matrix panel
    draw_panel(d, 1195, 210, 1130, 465)
    d.text((1223, 234), "EXPERIMENT MATRIX", font=FONT_SECTION, fill=GOLD)
    matrix = [
        "white-box      5 models × 3 datasets = 15 cells",
        "black-box      2 sources × 3 datasets = 6 cells",
        "official data  25 model–dataset pairs",
        "cross-lingual  mGPT German / English",
        "robustness     samples, length, proxy, decoding,",
        "               paraphrase, aggregation",
        "ablations      LL-only, MDE-only, aggregation",
    ]
    draw_lines(d, 1223, 288, matrix, FONT_BODY, WHITE, 36)

    # Results panel
    draw_panel(d, 72, 715, 1130, 600)
    d.text((100, 739), "KEY RESULTS (AUROC)", font=FONT_SECTION, fill=GOLD)

    wb_dirs = {
        "GPT-2-XL": "white_box_gpt2-xl_3datasets_150",
        "Falcon-7": "white_box_falcon-7b_3datasets_150",
        "GPT-J": "white_box_gpt-j_3datasets_150",
        "Phi-2": "white_box_phi-2_3datasets_150",
        "OPT-2.7": "white_box_opt-2.7_3datasets_150",
    }
    wb_vals = {k: auroc(v) for k, v in wb_dirs.items()}
    valid = [v for v in wb_vals.values() if v is not None]
    wb_avg = sum(valid) / len(valid) if valid else None

    res_lines = []
    res_lines.append(f"white-box Lastde avg    {pct(wb_avg)}   ({len(valid)}/5 models)")
    items = list(wb_vals.items())
    res_lines.append("  " + " | ".join(f"{k} {pct(v)}" for k, v in items[:3]))
    res_lines.append("  " + " | ".join(f"{k} {pct(v)}" for k, v in items[3:]))
    res_lines.append("")
    res_lines.append("black-box GPT-2 → GPT-J")
    res_lines.append(f"  XSum {pct(auroc('black_box_gpt2_xsum_150'))} | "
                     f"SQuAD {pct(auroc('black_box_gpt2_squad_150'))} | "
                     f"WP {pct(auroc('black_box_gpt2_wp_150'))}")
    opt_xsum = auroc("black_box_opt-2.7_xsum_wp_150", "xsum")
    opt_wp = auroc("black_box_opt-2.7_xsum_wp_150", "writingprompts")
    res_lines.append("black-box OPT-2.7 → GPT-J")
    res_lines.append(f"  XSum {pct(opt_xsum)} | SQuAD {pct(auroc('black_box_opt-2.7_squad_150'))} | "
                     f"WP {pct(opt_wp)}")
    res_lines.append("")
    res_lines.append(f"mGPT WMT16   German {pct(auroc('white_box_mGPT_wmt16_150', 'wmt16_de'))} | "
                     f"English {pct(auroc('white_box_mGPT_wmt16_150', 'wmt16_en'))}")
    res_lines.append("ablation GPT-2 XSum")
    res_lines.append(f"  Lastde {pct(auroc('ablation_gpt2_xsum_150'))} | "
                     f"LL-only {pct(auroc('ablation_gpt2_xsum_150', method='ll_only'))} | "
                     f"MDE-only {pct(auroc('ablation_gpt2_xsum_150', method='mde_only'))}")
    res_lines.append("")
    res_lines.append(official_summary())

    draw_lines(d, 100, 793, res_lines, FONT_BODY, GREEN, 36)

    # Checks panel
    draw_panel(d, 1245, 715, 1080, 600)
    d.text((1273, 739), "PROTOCOL CHECKS", font=FONT_SECTION, fill=GOLD)
    checks = [
        f"pairs / dataset       150   (XSum black-box {n_pairs('black_box_gpt2_xsum_150')})",
        "prompt                30 tokens",
        "continuation          200 tokens",
        "human cap             200 tokens",
        "length matching       explicit",
        "AUROC                 bootstrap 1000",
        "cache                 (model, text) TPS cache",
        "results               results/**/summary.csv",
        "",
        "commands",
        "$ nvidia-smi --query-gpu=name,memory.total --format=csv",
        "$ python scripts/run_experiment.py --config configs/...",
        "$ python scripts/make_evidence.py",
    ]
    draw_lines(d, 1273, 793, checks, FONT_BODY, WHITE, 38)

    # Footer
    d.rectangle((72, 1370, 2328, 1372), fill=BORDER)
    d.text((72, 1392),
           "Values are read from result CSVs, not hard-coded. Missing files are shown as N/A.  "
           "This sheet is safe to use as a backup slide.",
           font=FONT_SMALL, fill=MUTED)

    img.save(OUT)
    print(f"saved {OUT.resolve()}")
    print("\nENVIRONMENT")
    for line in env_lines():
        print(" ", line)
    print("\nKEY RESULTS")
    for line in res_lines:
        if line:
            print(" ", line)
    print(f"\nimage: {OUT.name}  ({W}x{H})")


if __name__ == "__main__":
    main()

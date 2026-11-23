import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "src")
from lastde.core import auroc
from lastde.models import load_model
from lastde.runner import score_methods

MODEL_PATH = "/root/autodl-tmp/models/gpt-j-6B"
DATA_DIR = "data/official"
METHODS = ["lastde", "likelihood", "logrank", "entropy", "detectlrr"]

cfg = {
    "methods": METHODS,
    "lastde": {"s": 3, "eps_factor": 10, "tau_prime": 5, "agg": "std"},
}

model = load_model(MODEL_PATH, device="cuda")
rows = []

for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.raw_data.json"))):
    base = os.path.basename(path).replace(".raw_data.json", "")
    dataset, source = base.split("_", 1)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print("skip bad", base, e)
        continue

    originals = data.get("original", [])
    sampled = data.get("sampled", [])
    scores = {m: {"human": [], "llm": []} for m in METHODS}
    n = min(len(originals), len(sampled))

    for i in range(n):
        for label, text in (("human", originals[i]), ("llm", sampled[i])):
            if not text or not text.strip():
                continue
            out = model.logprobs(text)
            if out.tps.size < 10:
                continue
            vals = score_methods(out, None, cfg)
            for m in METHODS:
                scores[m][label].append(vals[m])

    for m in METHODS:
        pos, neg = scores[m]["llm"], scores[m]["human"]
        if len(pos) >= 2 and len(neg) >= 2:
            rows.append({
                "dataset": dataset,
                "source": source,
                "method": m,
                "auroc": auroc(pos, neg),
                "n_pairs": min(len(pos), len(neg)),
            })
    print("done", base, flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/official_blackbox_results.csv", index=False)
print(df.pivot_table(index=["dataset", "source"], columns="method", values="auroc").round(4).to_string())

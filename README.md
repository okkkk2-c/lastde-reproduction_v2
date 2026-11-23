# lastde_full

Clean rebuild of the Lastde reproduction. The math in `src/lastde/core.py` is reused from the working prototype; the data/model/orchestration layers are being rebuilt for the full paper matrix.

## Status

- [x] Experiment matrix and GPU plan: `docs/EXPERIMENT_MATRIX.md`
- [x] Model registry: `models.yaml`
- [x] Dataset registry: `datasets.yaml`
- [x] Experiment matrix: `experiments.yaml`
- [x] Matrix orchestrator: `scripts/run_matrix.py` (pair generation + per-model configs + resume)
- [ ] Resume/cache/batching at pair/method granularity (currently skips finished pair files and summary.csv)
- [ ] Reddit ELI5 loader
- [ ] Full white-box 12 x 3
- [ ] Full black-box 12 x 3
- [ ] Sampling/heavy baselines subset
- [ ] Cross-lingual / robustness / ablation / hyperparameters

## Quick start (prototype path)

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_datasets.py --datasets xsum
python scripts/build_pairs.py --datasets xsum --models gpt2 --limit 150
python scripts/run_experiment.py --config configs/xsum_gpt2_150_fast.yaml
```

## Full-matrix runner

The orchestrator is implemented:

```bash
# white-box main: 12 models x XSum/SQuAD/WritingPrompts
python scripts/run_matrix.py --phase white_box_main --models gpt2,opt-2.7,gpt-j

# black-box main: 11 open models x XSum/WritingPrompts/Reddit, GPT-J proxy
python scripts/run_matrix.py --phase black_box_main --models gpt2,opt-2.7

# dry run
python scripts/run_matrix.py --phase white_box_main --models gpt2 --dry-run
```

It resolves specs from `models.yaml` / `datasets.yaml`, generates missing pairs, writes one config per model under `configs/matrix/`, calls `run_experiment.py`, and skips existing pair files / `summary.csv`. It currently runs the cheap methods (`lastde`, `likelihood`, `logrank`, `entropy`, `detectlrr`); sampling methods are added separately.

Closed-source generation (GPT-4-Turbo/GPT-4o/Claude) and Reddit ELI5 are not wired in yet.

## GPU

- 24 GB: GPT-J and <=7B models; 13B needs 4-bit.
- 48 GB: all 12 main models at fp16; recommended for the main table.
- Disk: ~200 GB for all model weights; download per model and delete after use.

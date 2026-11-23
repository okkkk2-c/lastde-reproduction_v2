# AutoDL runbook (48 GB, full matrix first pass)

## 1. Instance

- GPU: **RTX 6000 Ada 48 GB** (recommended) or A100 40/80 GB.
- Image: PyTorch 2.4 / CUDA 12.1 (or the closest available).
- System disk: 30 GB; data disk: **200 GB** (model weights are the big item).
- Region: choose one with fast HuggingFace mirror access.

Why 48 GB: the 13B models (Llama-13, Llama2-13, OPT-13) need ~26 GB just for fp16 weights; 24 GB will OOM.

## 2. Upload and install

```bash
cd /root/autodl-tmp
unzip lastde_full.zip
cd lastde_full
pip install -r requirements.txt
```

Set the mirror and, if you have gated models, your HF token:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=hf_xxx
```

Check the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Data

```bash
python scripts/prepare_datasets.py --datasets xsum,squad,writingprompts
python scripts/prepare_datasets.py --datasets wmt16_de,wmt16_en
```

Reddit ELI5 needs a loader fix first; see `datasets.yaml` (`status: needs_loader_fix`).
Zhihu is not public; put `data/human/zhihu.jsonl` there only if you have it.

## 4. Model download strategy

Do **not** download all 18 models at once. Download one model, run its three datasets, then delete the weights.

```bash
# example: OPT-2.7
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-2.7b')"
```

Gated models (Llama2-13, Llama3-8, Gemma-7) require accepting the license on huggingface.co and a token. The mirror may not proxy gated repos; if it fails, download from the official Hub with your token.

## 5. First run: white-box GPT-2 smoke

Before the full matrix, verify the pipeline on one model and one dataset:

```bash
python scripts/build_pairs.py --datasets xsum --models gpt2 --limit 20
python scripts/run_experiment.py --config configs/xsum_gpt2_150_fast.yaml --limit 20
```

Then run the full 150 pairs.

## 6. Full matrix (target interface)

The matrix runner is being added. The intended calls are:

```bash
# Phase B: white-box main
python scripts/run_matrix.py --phase white_box_main \
  --models gpt2,neo-2.7,opt-2.7,gpt-j,llama-13,llama2-13,llama3-8,opt-13,bloom-7.1,falcon-7,gemma-7,phi-2 \
  --datasets xsum,squad,writingprompts \
  --methods lastde,likelihood,logrank,entropy,detectlrr

# Phase C: black-box main
python scripts/run_matrix.py --phase black_box_main \
  --models gpt2,neo-2.7,opt-2.7,llama-13,llama2-13,llama3-8,opt-13,bloom-7.1,falcon-7,gemma-7,phi-2,gpt-4-turbo \
  --datasets xsum,writingprompts,reddit \
  --proxy gpt-j \
  --methods lastde,likelihood,logrank,entropy,detectlrr
```

## 7. Cost control

- 48 GB card at ~RMB 4-6/hour.
- Main white-box + black-box: 30-50 GPU hours (rough).
- Budget RMB 300-600 including sampling, failed runs, and downloads.
- Stop the instance when not running; model downloads and data prep can be done on CPU.

## 8. Resume and cache

- TPS cache: `(model, text) -> tps/ranks/entropy`.
- Sample cache: `(proxy, text) -> sampled outputs`.
- Run rows: append-only JSONL; rerunning skips completed `(model, dataset, pair, method)`.

## 9. Known blockers

- Gated model access.
- Reddit ELI5 loader.
- Zhihu data.
- 13B on 24 GB (use 48 GB or 4-bit).
- Fast-DetectGPT 10k sampling: use the analytical solution when sampling and scoring are the same model.

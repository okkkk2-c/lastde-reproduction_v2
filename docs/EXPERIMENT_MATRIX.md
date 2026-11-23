# Lastde full reproduction: experiment matrix

## One-line reading

The paper is **not** a 6-dataset x 18-model Cartesian product. It is five experiment groups, each with a different subset. The two main lines are:

- **White-box main:** 12 open-source models x 3 datasets (XSum, SQuAD, WritingPrompts), source model scores itself.
- **Black-box main:** 11 open-source models + GPT-4-Turbo x 3 datasets (XSum, WritingPrompts, Reddit), GPT-J is the sole proxy.

The other groups are cross-lingual (WMT16, Zhihu), robustness (5 sub-experiments), and ablation/hyperparameters.

## Datasets

| Dataset | Domain / language | Used in | Availability |
|---|---|---|---|
| XSum | BBC news / en | white-box main, black-box main | public |
| SQuAD | Wikipedia QA / en | white-box main | public |
| WritingPrompts | story / en | white-box main, black-box main | public |
| Reddit ELI5 | Reddit QA / en | black-box main | public, loader needs fixing |
| WMT16 | translation / en-de | cross-lingual | public |
| Zhihu | Q&A / zh | cross-lingual | not public; supply local JSONL |

Each dataset: 150 human examples; first 30 tokens as prompt; source model continues for about 200 tokens.

## Models

18 source models = 15 open-source + 3 closed-source.

**12 main open models:** GPT-2, Neo-2.7, OPT-2.7, GPT-J, Llama-13, Llama2-13, Llama3-8, OPT-13, BLOOM-7.1, Falcon-7, Gemma-7, Phi-2.

**3 cross-lingual models:** mGPT (German/English), Qwen1.5 (Chinese), Yi-1.5 (Chinese).

**3 closed-source models:** GPT-4-Turbo, GPT-4o, Claude-3-haiku.

GPT-J is both a main source and the black-box proxy. In the black-box main table it is excluded from the source list to avoid using the same model as source and proxy; GPT-4-Turbo takes its place.

## Baselines

Group 1 (score-only): Likelihood, LogRank, Entropy, DetectLRR.
Group 2 (sample-based): DetectGPT, DetectNPR, DNA-GPT, Fast-DetectGPT.

Lastde compares with group 1; Lastde++ compares with group 2.

## Five experiment groups

| Group | Question | Data | Source models | Proxy | Scale |
|---|---|---|---|---|---|
| 1. White-box main | Does Lastde beat group 1? | XSum/SQuAD/WP | 12 open | source itself | 12 x 3 x 8 |
| 2. Black-box main | Does it work without the source model? | XSum/WP/Reddit | 11 open + GPT-4-Turbo | GPT-J | 12 x 3 x 8 |
| 3. Cross-lingual | Does it work in German/Chinese? | WMT16/Zhihu | mGPT/Qwen1.5/Yi-1.5 | GPT-J or language proxy | 2 x 3 x 8 |
| 4. Robustness | Sample count / paraphrase / decoding / length / proxy | XSum/WP | 3-5 representative models | GPT-J | 5 sub-experiments |
| 5. Ablation / hyperparameters | Are LL and DE both needed? Are s/epsilon/tau stable? | XSum/WP | GPT-2/OPT-2.7/GPT-J | GPT-J | 3 variants + 5 aggregations + 3 sweeps |

## Priority

1. Build clean repo, registries, loaders, config generation, cache/resume.
2. White-box main: 12 models x 3 datasets, Lastde + 4 cheap baselines.
3. Black-box main: 12 source models x 3 datasets, GPT-J proxy.
4. Sampling/heavy baselines on a subset: Lastde++ 100, Fast-DetectGPT analytic or 10k, DetectGPT/DetectNPR/DNA-GPT.
5. Cross-lingual, robustness, ablation, hyperparameters.
6. Closed-source generation and final report.

## GPU

- 24 GB: GPT-J and <=7B models; 13B requires 4-bit (not strictly paper-faithful).
- 48 GB: all 12 main models at fp16; recommended.
- Disk: ~200 GB for all weights; download per model and delete after use.
- Estimated main white-box + black-box: 30-50 GPU hours; budget RMB 300-600 including sampling and reruns.

## Known risks

- Gated models: Llama2-13, Llama3-8, Gemma-7 need HF license acceptance and token.
- Reddit ELI5 loader needs a separate fix.
- Zhihu is not public; Chinese cross-lingual cannot be completed without user data.
- Fast-DetectGPT 10k sampling is expensive; use the analytical solution when sampling and scoring use the same model.
- DetectGPT/DetectNPR/DNA-GPT need extra models or rewriting; run on a subset first.
- 13B fp16 needs 40 GB+; 24 GB only works with 4-bit, which changes the probabilities.

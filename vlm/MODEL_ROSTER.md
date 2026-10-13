# VLM / counting-model roster (fixed for the paper)

Checked 2026-10-06 against the OpenRouter models API and the Hugging Face Hub API.
**Use only these models in reported results.** Anything else run during development counts as a harness smoke test and is not reported.

## Selection rule
1. **Frontier upper bound:** the latest flagship vision model from each major lab (OpenAI, Anthropic, Google, Qwen). This answers whether the best available VLMs can do the task.
2. **Open-weight state of the art:** the newest, most-adopted open VLMs that a factory could self-host.
3. **Counting specialists:** models trained to point at or count objects (Molmo2 pointing) and open-world counters (CountGD). Counting is their whole design purpose, so they are the fairest "learned counting" comparison.
4. **No stale models:** nothing older than about 12 months (rules out Llama 3.x, Qwen2.5-VL, InternVL3, SmolVLM and similar).

## Closed frontier models (via OpenRouter, `OPENROUTER_API_KEY` from env)
| Role | OpenRouter id | Released |
|---|---|---|
| OpenAI flagship | `openai/gpt-6.1-sol` | 2026-09-29 |
| Anthropic flagship | `anthropic/claude-fable-5.1` | 2026-09-01 |
| Google, strongest reasoning | `google/gemini-3.1-pro-preview` | 2026-02-19 |
| Google, latest | `google/gemini-3.8-flash` | 2026-09-02 |
| Qwen flagship | `qwen/qwen3.8-max-prime` | 2026-09-23 |

## Open-weight models
| Role | Model | How to run | Released |
|---|---|---|---|
| Newest top open VLM (17k likes) | `Qwen/Qwen3.8-27B` | local MLX `mlx-community/Qwen3.8-27B-4bit` (~16 GB) | 2026-08-05 |
| Google open VLM | `google/gemma-4-26B-A4B-it` | OpenRouter `google/gemma-4-26b-a4b-it` (saves disk) or local `mlx-community/gemma-4-26b-a4b-it-4bit` | 2026-03 |
| Pointing / counting specialist | `allenai/Molmo2-8B` | local MLX `mlx-community/Molmo2-8B-4bit` (~5 GB); point at each yarn on the guide line, then count the points | 2025-12-14 |

## Non-VLM counting baselines
| Model | Notes |
|---|---|
| CountGD (`nikigoli/CountGD`, NeurIPS 2024) | Open-world counting from a text prompt plus visual exemplars (a few clicked yarn crossings). Run if it installs on macOS (GroundingDINO ops); otherwise document why it was skipped. |
| Exemplar counters (GeCo / DAVE) | Optional; pending the literature agent's feasibility check. |

## Our methods
- **2019 pipeline (faithful rebuild):** segmentation, then the dot-map counter.
- **ours-2026:** spectral/lattice estimator with a capture-quality score.
- **CNN:** ImageNet-pretrained, synthetic pretraining, then 5-fold fine-tuning on the 150 images.

## Protocol notes
- Same rectified 1-inch crop as input for every model; one prompt-ablation run on the full photo.
- Prompts: P1 direct JSON, P2 step-by-step, P3 quarter-strip ×4, P4 pointing (Molmo2 and any model that supports points).
- Temperature 0. Log raw outputs. 3 repeats on a 20-image subset to measure answer stability.
- Record model id, provider, date and cost per image.
- Azure OpenAI is not needed: OpenRouter covers OpenAI models with one consistent client.

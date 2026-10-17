# Third-party models, code and services

No third-party model weights, datasets or code are vendored in this repository. The scripts download or call the following; each is used under its own licence and terms of use, which you must check before use. Exact model ids, revisions and access dates are in the run logs (`vlm/logs/`, `vlm/raw/*.jsonl`, `gpu_results/logs/`, `ft_results/logs/`, `genai_sr/logs/`).

| Used for | Component | Where in this repo | Notes |
|---|---|---|---|
| API VLM baselines | OpenRouter API; `openai/gpt-6.1-sol`, `anthropic/claude-fable-5.1`, `google/gemini-3.1-pro-preview`, `google/gemini-3.8-flash`, `qwen/qwen3.8-max-prime`, `qwen/qwen3.8-27b`, `google/gemma-4-26b-a4b-it` | `vlm/openrouter_client.py`, `vlm/MODEL_ROSTER.md` | key from `$OPENROUTER_API_KEY` only; images are sent to third parties; the two open-weight models (`qwen/qwen3.8-27b`, `google/gemma-4-26b-a4b-it`) are also queried via OpenRouter, not run locally |
| Pointing specialist (baseline) | `allenai/Molmo2-8B` (MLX 4-bit and bf16) | `vlm/`, `gpu_results/code/` | |
| Base model of our tuned VLM | `allenai/Molmo2-4B`, which we LoRA-tune on synthetic crops only (Molmo2-4B + synthetic LoRA, one of the paper's two methods; see [MODEL_CARD.md](MODEL_CARD.md)) | `ft_results/code/` | the LoRA adapter is ours; it is not in git and can be attached to a GitHub release; the base model's own licence applies to it |
| Open-world counter | CountGD (`nikigoli/CountGD`, NeurIPS 2024) with GroundingDINO ops | `cnn/countgd/`, `gpu_results/countgd/code/` | run on a GPU server (`gpu_results/countgd/code/`) and, for the legacy crops, also locally on Apple MPS (`cnn/countgd/run_countgd.py`) |
| CNN backbones | ResNet-18, EfficientNet-B0 (torchvision / timm, ImageNet weights) | `cnn/` | the synthetic-only checkpoint (43 MB) can be attached to a GitHub release; checkpoints fine-tuned on pseudo-labels of the real photos are photo-derived and follow the data release |
| Generative enhancement | SwinIR-M x4, Real-ESRGAN x4plus, SeedVR2-3B (Apache-2.0), InvSR (SD-Turbo noise predictor), FLUX.2 [klein] 4B (Apache-2.0), Nano Banana Pro (`google/gemini-3-pro-image` via OpenRouter) | `genai_sr/server/`, `genai_sr/ai_enhance_openrouter.py` | protocol pre-registered in `genai_sr/README.md` |
| Generic IQA (optional) | pyiqa metrics | `paper/tools/iqa_*.py` | |
| Legacy baseline | the Textile Quality Analysis pipeline of an earlier project (IIT Madras CVI club), re-implemented line by line | `methods/legacy_2019/` | credit to the original team (see README, Acknowledgements) |
| Standards cited | IS 1963:1981, ISO 7211-2:2024, ASTM D3775, IS 179, IS 15853, IS 745 | `docs/problem_definition.md` | paraphrased; the standards' texts are not included |

Python dependencies are listed in `requirements.txt` and `requirements-cnn.txt`.

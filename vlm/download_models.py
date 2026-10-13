"""Download the pinned MLX VLM checkpoints listed in models.json into the default
Hugging Face cache (~/.cache/huggingface/hub). Public models only; no token needed.

    python download_models.py            # all models
    python download_models.py qwen35_9b  # one model
"""
import json, os, sys, time
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = json.load(open(os.path.join(HERE, "models.json")))

def main(keys):
    for k in keys:
        spec = MODELS[k]
        t = time.time()
        p = snapshot_download(spec["repo"], revision=spec["revision"])
        print(f"{k}: {spec['repo']}@{spec['revision'][:12]} -> {p} ({time.time()-t:.0f}s)", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or [k for k in MODELS if not k.startswith("_")])

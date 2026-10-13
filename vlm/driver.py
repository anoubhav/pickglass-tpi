"""Run run_vlm.py for several models/sets, restarting the worker process after
Metal GPU errors (on a shared, busy GPU a command-buffer timeout poisons the
process' Metal queue; a fresh process resumes from the JSONL log).

    python driver.py --models qwen35_9b,gemma4_12b --sets legacy,rect150
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ,
           MLX_MAX_OPS_PER_BUFFER=os.environ.get("MLX_MAX_OPS_PER_BUFFER", "10"),
           MLX_MAX_MB_PER_BUFFER=os.environ.get("MLX_MAX_MB_PER_BUFFER", "20"),
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--prompts", default=None)
ap.add_argument("--ids", default=None, help="comma list of image ids (passed to run_vlm.py)")
ap.add_argument("--max-restarts", type=int, default=30)
a = ap.parse_args()
for model in a.models.split(","):
    for st in a.sets.split(","):
        for k in range(a.max_restarts):
            cmd = [sys.executable, os.path.join(HERE, "run_vlm.py"), "--model", model, "--set", st]
            if a.prompts:
                cmd += ["--prompts", a.prompts]
            if a.ids:
                cmd += ["--ids", a.ids]
            rc = subprocess.call(cmd, env=ENV, cwd=HERE)
            print(f"### driver: {model}/{st} exit code {rc} (attempt {k+1})", flush=True)
            if rc == 0:
                break
            time.sleep(15)

"""GPU-hour accounting for the Molmo2-4B synthetic LoRA fine-tune from the server logs/raw records (run on the
server in ft/code).
Training: wall time between the first and last log line x number of GPUs.  Inference: sum of batch
wall times per (model, set) (each batch occupies one GPU)."""
import glob
import json
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
rows = []
for name, log, ngpu in (("smoke (40 steps, 1 GPU)", "ckpt/smoke/train_rank0.log", 1),
                        ("run1 (main, 2 GPUs DDP)", "ckpt/run1/train_rank0.log", 2)):
    p = os.path.join(FT, log)
    if not os.path.exists(p):
        continue
    ts = [l[:8] for l in open(p) if re.match(r"\d\d:\d\d:\d\d ", l)]
    t0, t1 = (datetime.strptime(ts[0], "%H:%M:%S"), datetime.strptime(ts[-1], "%H:%M:%S"))
    h = ((t1 - t0).total_seconds() % 86400) / 3600
    rows.append((name, round(h, 3), ngpu, round(h * ngpu, 3)))
for raw_dir in ("raw", "raw_diag"):
    for p in sorted(glob.glob(os.path.join(FT, raw_dir, "*.jsonl"))):
        seen, tot = set(), 0.0
        for l in open(p):
            r = json.loads(l)
            k = (r["timestamp"], r["batch_seconds"])
            if k not in seen:
                seen.add(k)
                tot += r["batch_seconds"]
        rows.append((f"inference {os.path.basename(p)}", round(tot / 3600, 3), 1, round(tot / 3600, 3)))
print(f"{'item':60s} {'wall_h':>7s} {'gpus':>4s} {'gpu_h':>6s}")
for r in rows:
    print(f"{r[0]:60s} {r[1]:7.2f} {r[2]:4d} {r[3]:6.2f}")
print(f"{'TOTAL':60s} {'':7s} {'':4s} {sum(r[3] for r in rows):6.2f}")
json.dump([dict(item=a, wall_h=b, gpus=c, gpu_h=d) for a, b, c, d in rows],
          open(os.path.join(FT, "results", "gpu_hours.json"), "w"), indent=1)

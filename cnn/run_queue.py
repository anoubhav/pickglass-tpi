#!/usr/bin/env python
"""run_queue.py -- dependency-aware GPU work queue for train_aug.py (Job 2).

One worker per GPU:   CUDA_VISIBLE_DEVICES=<g> python run_queue.py --tasks runs_aug/tasks.json --gpu <g>
Workers claim tasks atomically (runs_aug/_queue/<id>.claim), run them as subprocesses (resumable:
train_aug.py resumes from last.pt and skips runs with DONE), and record .done / .failed.  A task is
started only when its dependencies are done AND the GPU shows no compute process (idle check before
every launch; the worker waits otherwise).  Placeholders {rho_r18} / {rho_b0} resolve to the real
fraction of the V4 run with the lower selection score.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs_aug")
Q = os.path.join(RUNS, "_queue")


def own_pids(files):
    out = set()
    for f in files:
        try:
            out.add(int(open(f).read().split()[0]))
        except Exception:  # noqa: BLE001
            pass
    return out


def gpu_busy(g, ignore=()):
    """True if any compute process other than our own listed ones (``ignore`` PIDs) is on GPU g."""
    try:
        uuid = subprocess.check_output(["nvidia-smi", "-i", str(g), "--query-gpu=uuid", "--format=csv,noheader"],
                                       text=True).strip()
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                       text=True)
        return any(l.split(",")[0].strip() == uuid and int(l.split(",")[1]) not in ignore
                   for l in apps.splitlines() if l.strip())
    except Exception as e:  # noqa: BLE001
        print("nvidia-smi check failed:", e, flush=True)
        return True


def done(tid):
    return os.path.exists(os.path.join(Q, f"{tid}.done"))


def best_score(run):
    p = os.path.join(RUNS, run, "best.json")
    return json.load(open(p))["score"] if os.path.exists(p) else float("inf")


def resolve(cmd):
    subs = {}
    for bb in ("r18", "b0"):
        a, b = best_score(f"V4_{bb}_rho1"), best_score(f"V4_{bb}_rho05")
        subs[f"rho_{bb}"] = "1.0" if a <= b else "0.5"
    return [c.format(**subs) if "{" in c else c for c in cmd]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--own-pid-files", default="", help="comma list of PID files of OUR other GPU jobs (e.g. the "
                                                         "'final' predictions) that may share the GPU")
    a = ap.parse_args()
    os.makedirs(Q, exist_ok=True)
    while True:
        tasks = json.load(open(a.tasks))
        pending = [t for t in tasks if not done(t["id"]) and not os.path.exists(os.path.join(Q, f"{t['id']}.failed"))]
        if not pending:
            print(f"[gpu {a.gpu}] queue empty", flush=True)
            return
        ready = [t for t in pending if all(done(d) for d in t.get("deps", []))
                 and not os.path.exists(os.path.join(Q, f"{t['id']}.claim"))
                 and (not t.get("gpu") or str(t["gpu"]) == str(a.gpu))]
        if not ready:
            if all(os.path.exists(os.path.join(Q, f"{t['id']}.claim")) or
                   any(os.path.exists(os.path.join(Q, f"{d}.failed")) for d in t.get("deps", [])) for t in pending):
                running = [t["id"] for t in pending if os.path.exists(os.path.join(Q, f"{t['id']}.claim"))]
                if not running:
                    print(f"[gpu {a.gpu}] only blocked tasks left", flush=True)
                    return
            time.sleep(20)
            continue
        if gpu_busy(a.gpu, own_pids([f for f in a.own_pid_files.split(",") if f])):
            print(f"[gpu {a.gpu}] busy with another process; waiting", flush=True)
            time.sleep(60)
            continue
        t = ready[0]
        try:
            fd = os.open(os.path.join(Q, f"{t['id']}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"gpu {a.gpu} pid {os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
        cmd = [a.py, os.path.join(HERE, "train_aug.py")] + resolve(t["cmd"])
        print(f"[gpu {a.gpu}] start {t['id']}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        logp = os.path.join(Q, f"{t['id']}.log")
        with open(logp, "a") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
            open(os.path.join(Q, f"{t['id']}.pid"), "w").write(str(p.pid))
            rc = p.wait()
        dt = time.time() - t0
        rec = {"id": t["id"], "gpu": a.gpu, "rc": rc, "seconds": dt, "cmd": cmd[2:], "end": time.ctime()}
        with open(os.path.join(Q, "ledger.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rc == 0:
            open(os.path.join(Q, f"{t['id']}.done"), "w").write(json.dumps(rec))
        else:
            open(os.path.join(Q, f"{t['id']}.failed"), "w").write(json.dumps(rec))
        os.remove(os.path.join(Q, f"{t['id']}.claim"))
        print(f"[gpu {a.gpu}] end {t['id']} rc={rc} {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

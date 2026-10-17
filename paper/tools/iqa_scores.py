"""iqa_scores.py -- generic no-reference IQA scores (pyiqa) for tools/iqa_generic.py and tools/iqa150.py.

Run on a Mac CPU (pyiqa 0.1.16, torch 2.14.1; no GPU). Label-safe: it only
reads images.

Metrics (pyiqa defaults, fp32): brisque, niqe (lower = better), musiq (KonIQ), clipiqa (RN50, 'Good/Bad
photo'), topiq_nr (CFANet, KonIQ) (higher = better).  Q-Align (7B LMM) was not run (no GPU).

Protocols (what image the metric sees):
  fix1024  every image resized to 1024 x 1024: INTER_AREA when larger, INTER_CUBIC when smaller.
           For images >= 1024 px this is exactly genai_sr/server/iq_metrics.py's to_t (W = 1024); the
           MUSIQ / CLIP-IQA values of the clean legacy references match the server's to 1e-4.
  work     the counter's working image: INTER_AREA to max side 1024 when larger, native otherwise
           (what a no-reference gate placed in front of the counter would see). Only differs from
           fix1024 for images < 1024 px, so it is only computed for those.
  view768  the genai_sr viewing copy as stored (768 x 768 JPEG q92, INTER_AREA from the S x S PNG;
           the full-resolution restorations stay on the GPU server); view768_made = the same transform applied
           here to the clean references.
  native   the image as stored (genai_sr 'none' = the LR PNG).

Inputs: the degraded legacy crops are the byte-identical PNGs written by methods/degrade_vlmcrops.py --make
(and methods/degrade_vlmcrops_extra.py) into $DEG_LEGACY13_DIR (default: $PICKGLASS_SCRATCH/deg_legacy13; md5s are
checked against their manifests and, in iqa_generic.py, against tables/degradation_legacy13_per_image.csv).
If that directory was cleaned, regenerate them with  DEG_LEGACY13_DIR=<dir> methods/degrade_vlmcrops.py --make.

    python iqa_scores.py --set legacy13|legacy13_extra|real150|genai_view|genai_refview|genai_full|genai_none \
                         --out RAW.csv [--shard i/n]
    python iqa_scores.py --finalize RAW_DIR      # -> tables/iqa_legacy13_scores.csv, iqa_genai_scores.csv,
                                                 #    iqa150_scores.csv  (numpy/pandas only: runs in .tools/venv)

The raw outputs of the run behind the paper tables (legacy13 as three resumed shards s0-s2; their duplicate rows are identical)
are not part of this repository; the tables in paper/tables were built from them with
    .tools/venv/bin/python -B paper/tools/iqa_scores.py --finalize <raw-output dir>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

# cv2 / torch / pyiqa are imported lazily, so that --finalize runs in the durable .tools/venv (no torch)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TAB = os.path.join(ROOT, "paper", "tables")
SCR = "$PICKGLASS_SCRATCH"
DEG = os.environ.get("DEG_LEGACY13_DIR", os.path.join(SCR, "deg_legacy13"))
GSR = os.path.join(ROOT, "genai_sr")
METRICS = ["brisque", "niqe", "musiq", "clipiqa", "topiq_nr"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def prep(bgr, protocol):
    import cv2
    h, w = bgr.shape[:2]
    if protocol == "fix1024":
        if (h, w) != (1024, 1024):
            interp = cv2.INTER_AREA if min(h, w) >= 1024 else cv2.INTER_CUBIC
            bgr = cv2.resize(bgr, (1024, 1024), interpolation=interp)
    elif protocol == "work":
        s = max(h, w)
        if s > 1024:
            f = 1024 / s
            bgr = cv2.resize(bgr, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    elif protocol in ("view768", "view768_made", "native"):
        pass
    else:
        raise ValueError(protocol)
    return bgr


def view768(bgr):
    """genai_sr/server/make_viewing.py: INTER_AREA to 768^2, JPEG q92 (decoded again)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (768, 768), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def items(which):
    import cv2
    out = []
    if which == "legacy13":
        man = pd.read_csv(os.path.join(DEG, "manifest.csv"))
        for r in man.itertuples():
            prots = ["fix1024"] + (["work"] if max(r.w, r.h) < 1024 else [])
            for p in prots:
                out.append(dict(set="legacy13", image=r.image, kind=r.kind, level=float(r.level), tag=r.tag,
                                path=r.path, protocol=p, md5_expected=r.md5))
    elif which == "legacy13_extra":   # motion / noise / shadow (degrade_vlmcrops_extra.py; ours + log-Gabor only)
        ed = DEG + "_extra"
        man = pd.read_csv(os.path.join(ed, "manifest.csv"))
        for r in man.itertuples():
            out.append(dict(set="legacy13_extra", image=r.image, kind=r.kind, level=float(r.level), tag=r.tag,
                            path=os.path.join(ed, r.tag, f"{r.image}.png"), protocol="fix1024", md5_expected=r.md5))
    elif which == "real150":
        for i in range(150):
            p = os.path.join(ROOT, "annotation_tool", "data", "crops", f"{i}.jpg")
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            prots = ["fix1024"] + (["work"] if max(im.shape[:2]) < 1024 else [])
            for pr in prots:
                out.append(dict(set="real150", id=i, path=p, protocol=pr))
    elif which == "genai_view":
        om = pd.read_csv(os.path.join(GSR, "results", "outputs_manifest.csv"))
        for r in om.itertuples():
            vp = os.path.join(GSR, "viewing", r.restorer, r.cond, r.key + ".jpg")
            out.append(dict(set=r.set, key=r.key, cond=r.cond, restorer=r.restorer, path=vp, protocol="view768"))
    elif which == "genai_refview":   # clean references through the same viewing transform
        m = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv")).drop_duplicates("key")
        for r in m.itertuples():
            out.append(dict(set=r.set, key=r.key, cond="clean", restorer="clean",
                            path=os.path.join(GSR, r.ref_path), protocol="view768_made", md5_expected=r.ref_md5))
    elif which == "genai_full":      # full-res S x S PNGs present on the Mac + clean refs: server protocol
        om = pd.read_csv(os.path.join(GSR, "results", "outputs_manifest.csv"))
        for r in om.itertuples():
            p = os.path.join(GSR, r.png)
            if os.path.exists(p):
                out.append(dict(set=r.set, key=r.key, cond=r.cond, restorer=r.restorer, path=p, protocol="fix1024",
                                md5_expected=r.png_md5))
        m = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv")).drop_duplicates("key")
        for r in m.itertuples():
            out.append(dict(set=r.set, key=r.key, cond="clean", restorer="clean",
                            path=os.path.join(GSR, r.ref_path), protocol="fix1024", md5_expected=r.ref_md5))
    elif which == "genai_none":      # unrestored LR inputs as stored (native size)
        m = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv"))
        for r in m.itertuples():
            out.append(dict(set=r.set, key=r.key, cond=r.cond, restorer="none",
                            path=os.path.join(GSR, r.lr_path), protocol="native", md5_expected=r.lr_md5))
    else:
        raise ValueError(which)
    return out


def finalize(raw_dir, tab=TAB):
    """Collect the raw outputs into the three per-image tables (relative paths only)."""
    import glob
    import json

    def load(pattern):
        fs = sorted(glob.glob(os.path.join(raw_dir, pattern)))
        if not fs:
            raise SystemExit(f"no {pattern} in {raw_dir}")
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

    cols = ["md5", "src_h", "src_w", "in_h", "in_w"] + METRICS
    leg = pd.concat([load("iqa_legacy13_raw*.csv"), load("iqa_legacy13_extra_raw*.csv")], ignore_index=True)
    leg = leg[["set", "image", "kind", "level", "tag", "protocol"] + cols]
    leg = leg.drop_duplicates(["set", "image", "kind", "level", "protocol"])   # resumed shards repeat rows
    assert len(leg) == 416, len(leg)
    leg.sort_values(["set", "image", "kind", "level", "protocol"]).to_csv(
        os.path.join(tab, "iqa_legacy13_scores.csv"), index=False)
    gen = pd.concat([load("iqa_genai_view_s*.csv"), load("iqa_genai_refview.csv"), load("iqa_genai_full.csv"),
                     load("iqa_genai_none.csv")], ignore_index=True)
    gen["file"] = [os.path.relpath(p, GSR) for p in gen.path]
    gen = gen[["set", "key", "cond", "restorer", "protocol", "file"] + cols]
    gen = gen.drop_duplicates(["set", "key", "cond", "restorer", "protocol"])
    gen.sort_values(["set", "key", "cond", "restorer", "protocol"]).to_csv(
        os.path.join(tab, "iqa_genai_scores.csv"), index=False)
    r = load("iqa_real150_raw*.csv").drop_duplicates(["id", "protocol"])
    man = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    meta = pd.DataFrame([dict(id=e["id"], scene_id=e.get("scene_id"), is_scene_rep=bool(e.get("is_scene_rep")),
                              kind=e.get("kind"), S=e["rect_size"][0]) for e in man["images"]]).set_index("id")
    w = None
    for prot in ("fix1024", "work"):
        x = r[r.protocol == prot].set_index("id")
        x = x[METRICS].rename(columns={m: f"{m}_{prot}" for m in METRICS})
        w = x if w is None else w.join(x, how="left")
    base = r[r.protocol == "fix1024"].set_index("id")[["md5", "src_h", "src_w"]]
    w = meta.join(base).join(w)
    for m in METRICS:   # 'work' = 'fix1024' input for crops >= 1024 px
        w[f"{m}_work"] = w[f"{m}_work"].fillna(w[f"{m}_fix1024"])
    o = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    g = pd.read_csv(os.path.join(TAB, "cascade_150_decisions.csv")).set_index("image")
    w["ours_conf"] = o.confidence.reindex(w.index)
    w["escalate_gate"] = g.escalate_gate.reindex(w.index)
    w.index.name = "id"
    w.reset_index().to_csv(os.path.join(tab, "iqa150_scores.csv"), index=False)
    print(f"wrote {len(leg)} legacy, {len(gen)} genai, {len(w)} real-150 rows to {tab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set")
    ap.add_argument("--out")
    ap.add_argument("--finalize", default=None, help="raw-output directory -> the three tables")
    ap.add_argument("--tables", default=TAB, help="--finalize: where to write the tables (default paper/tables)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th item starting at i")
    a = ap.parse_args()
    if a.finalize:
        return finalize(a.finalize, a.tables)
    import cv2
    import pyiqa
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    mets = {m: pyiqa.create_metric(m, device="cpu", as_loss=False) for m in a.metrics.split(",")}
    its = items(a.set)
    si, sn = (int(x) for x in a.shard.split("/"))
    its = its[si::sn]
    done = set()
    rows = []
    if os.path.exists(a.out):   # resume
        prev = pd.read_csv(a.out)
        rows = prev.to_dict("records")
        done = {(r["path"], r["protocol"]) for r in rows}
    t0 = time.time()
    for k, it in enumerate(its):
        if (it["path"], it["protocol"]) in done:
            continue
        bgr = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        it["md5"] = md5(it["path"])
        if it.get("md5_expected") and it["md5"] != it["md5_expected"]:
            raise SystemExit(f"md5 mismatch {it['path']}")
        it["src_h"], it["src_w"] = bgr.shape[:2]
        if it["protocol"] == "view768_made":
            x = view768(bgr)
        else:
            x = prep(bgr, it["protocol"])
        it["in_h"], it["in_w"] = x.shape[:2]
        t = torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).float().div(255).permute(2, 0, 1)[None]
        for name, m in mets.items():
            try:
                it[name] = float(m(t).item())
            except Exception as e:  # noqa: BLE001
                it[name] = np.nan
                it[f"{name}_err"] = repr(e)[:200]
        rows.append(it)
        if (k + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{a.set}: {k + 1}/{len(its)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{a.set}: wrote {len(rows)} rows to {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""train_aug.py -- train the tile CNN with count-preserving augmentation (WVAQ 2027).

Variants (ResNet-18 and EfficientNet-B0, ImageNet init; tile / px-per-inch formulation of cnn_common):
  V1  synthetic only, existing augmentation (cnn_common.TileDataset + photometric_aug), GPU re-run
  V2  synthetic only + verified whitelist augmentation: policy 'whitelist' (tier A), 'whitelist_ext'
      (tier A+B) or 'whitelist_optics' (A+B + the label-exact optics / phone-capture / fabric-geometry
      tier of augment_optics_whitelist.json); lowest selection score
  V3  pseudo-label self-training on real crops only (targets = ours' counts on the high-confidence
      SCENES, pre-declared rule), heavy whitelist augmentation ('whitelist_ext' or 'whitelist_optics')
  V4  synthetic pre-training (the selected V2 checkpoint) -> pseudo-real fine-tuning with the same
      policy (real fraction rho in {1, 0.5} chosen on the selection score)
  V5  V4 + test-time augmentation (flips, rot90 / rot270 with swap, +-5 deg rotations)
Ablation (selection metrics only): one augmentation family dropped in the fine-tuning stage of the V4
recipe (ResNet-18), with 2 extra fine-tuning seeds as the noise band; synthetic-stage drop-one runs of
'whitelist_ext' (trained before the scene split; synthetic only) are re-evaluated ('adopt').

SCENE SPLIT (supersedes an earlier auto-group split, kept in
runs_aug/_superseded_autogroup_split/): the 150 files are 76 scenes (camera_aug/originals_provenance.json,
annotation_tool/scenes_to_annotate.txt; per-file orientation in data/scene_map.csv).  Pseudo-labels,
acceptance and the hold-out are decided per SCENE on its representative file (highest-res real JPEG);
the other files of an accepted training scene (quarter-res copy, notebook-synthetic re-crop) are
training-time variants of the scene with the representative's label mapped to their orientation
(h <-> v when rotated by 90 deg); byte-identical files and PNG copies (same pixels) are not used; the
held-out pseudo-val uses the representative only.  No scene is in both pseudo-train and pseudo-val.

Model selection uses ONLY (pre-declared in runs_aug/PREREG.json):
  synth MAPE   synthetic held-out set data/synth_val (300, exact labels; P = 768, no flips)
  pseudo MAPE  representatives of the held-out 25 % of the accepted scenes, vs ours' counts
               (3 scales x 3 flips, the real-image protocol)
  score        = (synth MAPE + pseudo MAPE) / 2
The 13 legacy images are never read by this script except by 'final', which only writes predictions
(scored once, on the Mac, by score_aug.py).

Subcommands
  prep        scene-level pseudo-labels + hold-out split + PREREG.json   (no model, no legacy)
  ours_synth  frozen classical counter on synth_val_v2 (CPU; validates the pseudo-label rule)
  train       one resumable run -> runs_aug/<name>/{model.pt,best.pt,val.csv,train.log,meta.json,DONE}
  adopt       re-evaluate a synthetic-only checkpoint of the superseded run with the current validator
  select      apply the pre-declared selection rules -> selection.json, final_spec.json
  final       predictions of the final checkpoints (real150, legacy canonical crops, synth_val_v2)
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "2")

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

# OpenCV's thread pool must not be live when the DataLoader forks its workers (fork after a
# multi-threaded cv2 call deadlocks the workers): keep cv2 single-threaded in this process.
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import aug_pipeline as AP  # noqa: E402
import cnn_common as C  # noqa: E402

RUNS = os.path.join(HERE, "runs_aug")
OURS150 = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
GROUPS = os.path.join(HERE, "data", "auto_groups_092.npy")
TOOL_CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
LEGACY_CROPS = os.path.join(ROOT, "vlm", "crops", "legacy")
SYNTH_TRAIN = ",".join(os.path.join(HERE, "data", d) for d in ("synth_train", "synth_train_v2"))
SYNTH_VAL = os.path.join(HERE, "data", "synth_val")
SYNTH_VAL2 = os.path.join(HERE, "data", "synth_val_v2")
OURS_FROZEN = os.path.join(ROOT, "methods", "ours_frozen", "counting_frozen_final.py")
PSEUDO_CSV = os.path.join(RUNS, "pseudo_labels.csv")
SPLIT_SEED = 20261008
SCENE_MAP = os.path.join(HERE, "data", "scene_map.csv")      # scene_map.py (orientation vs representative)
VAL_FRAC = 0.25
SUPERSEDED = os.path.join(RUNS, "_superseded_autogroup_split")


# =============================================================================================
# prep: pseudo-labels + split (pre-declared rule)
# =============================================================================================
def cmd_prep_v0_autogroups(a):
    """SUPERSEDED (auto-group split; kept for the record).  It mixed copies of one scene
    across the per-image acceptance rule; use cmd_prep (scene split)."""
    raise SystemExit("prep_v0_autogroups is superseded by 'prep' (scene split); see "
                     "runs_aug/_superseded_autogroup_split/")
    os.makedirs(RUNS, exist_ok=True)
    d = pd.read_csv(OURS150)
    med = float(d.confidence.median())
    acc = (d.confidence >= med) & (d.f_fused_lattice == 1)
    g = np.load(GROUPS)
    d["group"] = g[d.id.values]
    ids = d.id[acc].tolist()
    ug = sorted(set(d.group[acc]))
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ug)
    target = int(round(0.2 * len(ids)))
    val = []
    for gg in ug:
        mem = d.id[acc & (d.group == gg)].tolist()
        if len(val) >= target:
            break
        if len(val) + len(mem) <= target + 2:
            val += mem
    split = np.where(~acc, "lowconf", np.where(d.id.isin(val), "val", "train"))
    out = pd.DataFrame({"id": d.id, "h": d.h_count, "v": d.v_count, "angle_x": d.angle_x.fillna(0.0),
                        "angle_y": d.angle_y.fillna(0.0), "weave_ours": d.weave,
                        "weave_idx": [C.weave_to_idx(w) for w in d.weave], "confidence": d.confidence,
                        "f_fused_lattice": d.f_fused_lattice, "fusion": d.fusion, "accepted": acc, "split": split,
                        "group": d.group})
    out.to_csv(PSEUDO_CSV, index=False)
    trg = set(out.group[out.split == "train"])
    vag = set(out.group[out.split == "val"])
    assert not (trg & vag)
    info = {"n_accepted": int(acc.sum()), "conf_median_150": med, "n_train": int((split == "train").sum()),
            "n_val": int((split == "val").sum()), "n_lowconf": int((split == "lowconf").sum()),
            "train_groups": len(trg), "val_groups": len(vag), "val_ids": sorted(int(i) for i in val)}
    print(json.dumps(info, indent=1))
    prereg = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "status": "written BEFORE any training / evaluation of this job",
        "pseudo_label_rule": ("accept image i iff ours' confidence >= median confidence over the 150 crops "
                              f"({med:.4f}) AND f_fused_lattice == 1 (on BOTH axes the lattice count was "
                              "confirmed by an independent estimator: fusion fft+profile or fft+tiles). "
                              "Targets = ours' h_count, v_count (guide-line crossing counts) with tilt "
                              "angle_x/angle_y for the label maps; weave = ours' weave ('unknown' -> ignored)."),
        "split": (f"hold out whole auto-groups (cnn/data/auto_groups_092.npy, cos 0.92) of the accepted images, "
                  f"groups shuffled with seed {SPLIT_SEED}, until >= 20 % of the accepted images"),
        "split_info": info,
        "selection_score": ("mean of (a) per-axis MAPE on data/synth_val (300, exact labels, P=768, no flips) and "
                            "(b) per-axis MAPE vs ours' pseudo-labels on the held-out pseudo-val images "
                            "(3 scales x 3 flips). Lower is better."),
        "legacy13": "test only: predictions written by 'final' after all selection; scored once (score_aug.py)",
        "variants": {
            "V1": "synthetic only (synth_train + synth_train_v2, 4800), policy 'existing', 4750 steps, batch 32, "
                  "AdamW 3e-4, cosine, aniso 0.5, lam_ratio 1, rot90 on (= the MPS recipe of "
                  "runs/synth_r18_v2mix_aniso)",
            "V2": "as V1 with policy 'whitelist' (tier A) and 'whitelist_ext' (tier A+B, rotate +-5 deg); the "
                  "policy with the lower selection score is V2 (per backbone)",
            "V3": "ImageNet init, pseudo-train images only, policy 'whitelist_ext', 3000 steps, lr 3e-4, snapshot "
                  "every 250 steps, best snapshot by selection score",
            "V4": "init = V2 'whitelist_ext' checkpoint of the same backbone; fine-tune lr 1e-4, 2000 steps, policy "
                  "'whitelist_ext', real fraction rho in {1.0, 0.5} (rest synthetic); best (rho, snapshot) "
                  "by selection score",
            "V5": "V4 checkpoint + TTA: 3 scales x {id, hflip, vflip, rot90, rot270} x {-5, 0, +5} deg",
            "ablation": "ResNet-18, V4 recipe (pre-train 4750 steps + fine-tune with V4-R18's selected rho) with one "
                        "family dropped in both stages: geometry, photometric, compression_blur, illumination, "
                        "zoom, tiling; plus 2 extra seeds of the full V4-R18 recipe for the noise band. "
                        "Reported: selection metrics only.",
        },
        "report": ("legacy-13 (canonical vlm/crops/legacy PNG, axis-matched, GT via paper/tools/legacy_unified.load_gt): "
                   "MAE, MAPE with image-bootstrap 95 % CI (10k), within 5/10 %, paired vs the MPS synthetic-only CNN "
                   "(cnn:synth_r18, 16.0 %) and vs ours (11.6 %) with legacy_unified.paired (bootstrap dMAPE CI, "
                   "Wilcoxon), Holm over the final variants per comparator."),
        "budget": "~8 GPU-hours (RTX 4090), idle GPUs only",
    }
    with open(os.path.join(RUNS, "PREREG.json"), "w") as f:
        json.dump(prereg, f, indent=1)
    print("wrote", PSEUDO_CSV, "and PREREG.json")


def cmd_prep(a):
    """Scene-level pseudo-labels + scene hold-out + PREREG.json (written before any training of the
    scene-split runs; no model output and no legacy image is read)."""
    os.makedirs(RUNS, exist_ok=True)
    if os.path.exists(PSEUDO_CSV) and not a.force:
        raise SystemExit(f"{PSEUDO_CSV} exists (use --force to rewrite)")
    d = pd.read_csv(OURS150).set_index("id")
    m = pd.read_csv(SCENE_MAP).set_index("id")
    assert len(m) == 150 and m.scene.nunique() == 76
    reps = sorted(int(s) for s in m.scene.unique())
    R = d.loc[reps]
    med = float(R.confidence.median())
    acc = {s: bool(R.confidence[s] >= med and R.f_fused_lattice[s] == 1) for s in reps}
    acc_scenes = [s for s in reps if acc[s]]
    order = list(acc_scenes)
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    n_val = int(math.ceil(VAL_FRAC * len(acc_scenes)))
    val_scenes = sorted(order[:n_val])
    rows = []
    for i in range(150):
        s = int(m.scene[i])
        swap = bool(m.swap[i])
        hr, vr = float(d.h_count[s]), float(d.v_count[s])
        h, v = (vr, hr) if swap else (hr, vr)
        split = "lowconf" if not acc[s] else ("val" if s in val_scenes else "train")
        kind = m.kind[i]
        rows.append({"id": i, "scene": s, "is_rep": i == s, "kind": kind, "swap": swap, "h": h, "v": v,
                     "angle_x": float(np.nan_to_num(d.angle_x[i])), "angle_y": float(np.nan_to_num(d.angle_y[i])),
                     "weave_ours": d.weave[s], "weave_idx": C.weave_to_idx(d.weave[s]),
                     "confidence": float(d.confidence[s]), "f_fused_lattice": float(d.f_fused_lattice[s]),
                     "fusion": d.fusion[s], "accepted": acc[s], "split": split,
                     "use_train": split == "train" and kind in ("real", "nb_synth"),
                     "use_val": split == "val" and i == s,
                     "own_h": float(d.h_count[i]), "own_v": float(d.v_count[i]),
                     "own_confidence": float(d.confidence[i])})
    out = pd.DataFrame(rows)
    out.to_csv(PSEUDO_CSV, index=False)
    trs, vas = set(out.scene[out.use_train]), set(out.scene[out.use_val])
    assert not (trs & vas), "scene in both pseudo-train and pseudo-val"
    assert set(out.scene[out.split == "train"]).isdisjoint(set(out.scene[out.split == "val"]))
    assert not out[out.use_val & (out.kind != "real")].shape[0], "pseudo-val must be real captures"
    info = {"n_scenes": 76, "conf_median_76_reps": med, "n_accepted_scenes": len(acc_scenes),
            "n_val_scenes": len(val_scenes), "n_train_scenes": len(trs),
            "n_train_files": int(out.use_train.sum()),
            "n_train_files_by_kind": out[out.use_train].kind.value_counts().to_dict(),
            "n_lowconf_scenes": 76 - len(acc_scenes), "val_scenes": val_scenes,
            "train_scenes": sorted(int(x) for x in trs)}
    print(json.dumps(info, indent=1))
    prereg = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "status": ("written BEFORE any training / evaluation of the scene-split round; supersedes the "
                   "2026-10-07 auto-group PREREG (runs_aug/_superseded_autogroup_split/PREREG.json), whose "
                   "split could place copies of one scene on both sides of the acceptance rule. Legacy-13 "
                   "predictions of the superseded round were written on the GPU server but never scored."),
        "scene_unit": ("76 scenes of camera_aug/originals_provenance.json; representative = highest-res real "
                       "JPEG (annotation_tool/scenes_to_annotate.txt); per-file orientation vs the "
                       "representative from data/scene_map.csv (NCC of low-passed crops, parity agrees with "
                       "the provenance rot90_k on all 52 representative pairs)"),
        "pseudo_label_rule": ("accept SCENE s iff ours' confidence on its representative >= the median ours' "
                              f"confidence over the 76 representatives ({med:.4f}) AND f_fused_lattice == 1 "
                              "on the representative. Target of every file of s = ours' (h_count, v_count) "
                              "on the representative, h <-> v swapped for files rotated by 90 deg; yarn tilt "
                              "= ours' angle_x / angle_y on the file itself; weave = ours' weave on the "
                              "representative ('unknown' -> ignored)."),
        "training_files": ("files of accepted pseudo-train scenes of kind 'real' (representative, quarter-res "
                           "copies) and 'nb_synth' (notebook re-crops = augmentations of their scene); "
                           "byte-identical duplicates and PNG decodes are skipped (same pixels). One file is "
                           "drawn per sample, so each scene has equal weight."),
        "split": (f"hold out ceil({VAL_FRAC} x n_accepted) accepted scenes, shuffled with seed {SPLIT_SEED}; "
                  "pseudo-val = their representatives only"),
        "split_info": info,
        "selection_score": ("mean of (a) per-axis MAPE on data/synth_val (300, exact labels, P=768, no flips) and "
                            "(b) per-axis MAPE vs ours' pseudo-labels on the held-out scene representatives "
                            "(3 scales x 3 flips). Lower is better."),
        "legacy13": "test only: predictions written by 'final' after all selection; scored once (score_aug.py)",
        "variants": {
            "V1": "synthetic only (synth_train + synth_train_v2, 4800), policy 'existing', 4750 steps, batch 32, "
                  "AdamW 3e-4, cosine, aniso 0.5, lam_ratio 1, rot90 on (= the MPS recipe of "
                  "runs/synth_r18_v2mix_aniso). Trained in the superseded round (synthetic only, no real "
                  "image): adopted unchanged and re-evaluated with this validator.",
            "V2": "as V1 with policy 'whitelist' (V2A), 'whitelist_ext' (V2E; both adopted from the superseded "
                  "round) or 'whitelist_optics' (V2O, new); the policy with the lowest selection score "
                  "(final step) is V2, per backbone",
            "V3": "ImageNet init, pseudo-train scenes only, 3000 steps, lr 3e-4, snapshot every 250 steps; "
                  "policy 'whitelist_ext' (V3E) and 'whitelist_optics' (V3O); best (policy, snapshot) by score",
            "V4": "init = the selected V2 checkpoint of the same backbone; fine-tune with the selected V2 policy, "
                  "lr 1e-4, 2000 steps, snapshot every 250, real fraction rho in {1.0, 0.5} (rest synthetic); "
                  "best (rho, snapshot) by score",
            "V5": "V4 checkpoint + TTA: 3 scales x {id, hflip, vflip, rot90, rot270} x {-5, 0, +5} deg",
            "ablation": ("ResNet-18. Stage 2 (new): the V4-R18 fine-tuning recipe (selected init, policy, rho) "
                         "with one family dropped (the families active in the policy: geometry, photometric, "
                         "compression_blur, illumination, zoom, tiling [+ optics, phone]), plus fine-tuning "
                         "seeds 1 and 2 of the full recipe as the noise band; a family matters only if its "
                         "score lies outside the 3-seed range. Stage 1 (adopted): the superseded round's "
                         "synthetic-only drop-one pre-trainings of 'whitelist_ext' (6 families, seeds 0-2), "
                         "re-evaluated with this validator. Reported: selection metrics only."),
        },
        "report": ("legacy-13 (canonical vlm/crops/legacy PNG, axis-matched, GT via paper/tools/legacy_unified.load_gt): "
                   "MAE, MAPE with image-bootstrap 95 % CI (10k), within 5/10 %, paired vs the MPS synthetic-only CNN "
                   "(cnn:synth_r18, 16.0 %) and vs ours (11.6 %) with legacy_unified.paired (bootstrap dMAPE CI, "
                   "Wilcoxon), Holm over the final variants per comparator."),
        "budget": ("<= 8 GPU-hours (RTX 4090) for the whole job INCLUDING the superseded round (~3.4 h), idle "
                   "GPUs only"),
    }
    with open(os.path.join(RUNS, "PREREG.json"), "w") as f:
        json.dump(prereg, f, indent=1)
    print("wrote", PSEUDO_CSV, "and PREREG.json")


# =============================================================================================
# ours on synth_val_v2 (CPU)
# =============================================================================================
def _ours_job(path):
    import importlib.util
    cv2.setNumThreads(1)
    spec = importlib.util.spec_from_file_location("ours_frozen", OURS_FROZEN)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    crop = cv2.imread(path)
    t0 = time.time()
    res = M.estimate_many(crop, ("ours", "lattice", "profile", "tiles", "fft"))
    e = res["ours"]
    dbg = e["debug"]
    lat = dbg.get("lattice") or {}
    f = dbg.get("conf_features") or {}
    return {"file": os.path.basename(path), "ours_h": e["h_count"], "ours_v": e["v_count"],
            "confidence": e["confidence"], "fusion": "|".join(dbg["fusion"]) if dbg.get("fusion") else None,
            "f_fused_lattice": f.get("fused_lattice", np.nan), "f_agree_prof": f.get("agree_prof", np.nan),
            "angle_x": lat.get("angle_x"), "angle_y": lat.get("angle_y"),
            **{f"{m}_h": res[m]["h_count"] for m in ("lattice", "profile", "tiles", "fft")},
            **{f"{m}_v": res[m]["v_count"] for m in ("lattice", "profile", "tiles", "fft")},
            "runtime_s": time.time() - t0}


def cmd_ours_synth(a):
    from multiprocessing import Pool
    lab = pd.read_csv(os.path.join(a.synth_dir, "labels.csv"))
    paths = [os.path.join(a.synth_dir, f) for f in lab.file]
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(_ours_job, paths, chunksize=2)
    o = pd.DataFrame(rows).merge(lab[["file", "h_count", "v_count", "weave_coarse", "rotation_deg", "r_eff"]], on="file")
    out = os.path.join(RUNS, f"ours_{os.path.basename(os.path.normpath(a.synth_dir))}.csv")
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} images, {time.time() - t0:.0f}s)")


# =============================================================================================
# data
# =============================================================================================
def synth_aug_items(dirs):
    items = []
    for d in str(dirs).split(","):
        d = d.strip()
        if not d:
            continue
        df = pd.read_csv(os.path.join(d, "labels.csv"))
        items += [AP.AugItem(None, os.path.join(d, r.file), float(r.h_count), float(r.v_count),
                             C.weave_to_idx(r.weave_coarse), 1.0, "density", (0.0, 0.0), "synth")
                  for r in df.itertuples()]
    return items


def load_real_crop(i):
    return C.cap_side(cv2.imread(os.path.join(TOOL_CROPS, f"{int(i)}.jpg")), 1024)


def pseudo_items(split="train"):
    """Scene groups: [[AugItem of each training file of scene s], ...] (one entry per scene)."""
    p = pd.read_csv(PSEUDO_CSV)
    p = p[p.use_train] if split == "train" else p[p.use_val]
    out = []
    for _, g in p.groupby("scene"):
        out.append([AP.AugItem(load_real_crop(r.id), None, float(r.h), float(r.v), int(r.weave_idx), 1.0,
                               "crossing", (float(r.angle_x), float(r.angle_y)), f"real{int(r.id)}")
                    for r in g.itertuples()])
    return out


def index_sequence(n_synth_idx, n_real_idx, rho, total, seed):
    """Deterministic sample-index stream: synthetic indices [0, n_synth_idx), real indices after them.
    rho = probability that a sample is real; each pool is walked through successive permutations."""
    rng = np.random.default_rng(seed)

    def stream(n, offset):
        while True:
            for j in rng.permutation(n):
                yield int(j) + offset

    if rho <= 0:
        s = stream(n_synth_idx, 0)
        return np.fromiter((next(s) for _ in range(total)), np.int64, total)
    if rho >= 1:
        s = stream(n_real_idx, n_synth_idx)
        return np.fromiter((next(s) for _ in range(total)), np.int64, total)
    ss, sr = stream(n_synth_idx, 0), stream(n_real_idx, n_synth_idx)
    pick = rng.random(total) < rho
    return np.array([next(sr) if p else next(ss) for p in pick], np.int64)


class SeqSampler(torch.utils.data.Sampler):
    def __init__(self, seq):
        self.seq = seq

    def __iter__(self):
        return iter(self.seq.tolist())

    def __len__(self):
        return len(self.seq)


def worker_init(_):
    cv2.setNumThreads(1)
    np.random.seed(torch.initial_seed() % 2 ** 32)


# =============================================================================================
# selection metrics
# =============================================================================================
class Validator:
    def __init__(self, n_synth=None, log=print):
        df = pd.read_csv(os.path.join(SYNTH_VAL, "labels.csv"))
        if n_synth:
            df = df.iloc[:n_synth]
        self.s_lab = df[["h_count", "v_count"]].to_numpy(float)
        self.s_crops = [C.cap_side(cv2.imread(os.path.join(SYNTH_VAL, f)), 1024) for f in df.file]
        p = pd.read_csv(PSEUDO_CSV)
        p = p[p.use_val]                       # representatives of the held-out scenes only
        assert p.is_rep.all() and p.scene.is_unique
        self.p_ids = p.id.tolist()
        self.p_lab = p[["h", "v"]].to_numpy(float)
        self.p_crops = [load_real_crop(i) for i in self.p_ids]
        log(f"validator: synth_val {len(self.s_crops)}, pseudo-val {len(self.p_crops)} scene representatives")

    @torch.no_grad()
    def __call__(self, model, dev):
        model.eval()
        sp = np.array([[o["h"], o["v"]] for o in (C.predict_crop(model, c, dev, Ps=(768,), flips=False)
                                                   for c in self.s_crops)])
        pp = np.array([[o["h"], o["v"]] for o in (C.predict_crop(model, c, dev) for c in self.p_crops)])
        se = np.abs(sp - self.s_lab) / self.s_lab
        pe = np.abs(pp - self.p_lab) / self.p_lab
        lr_t, lr_p = np.log(self.s_lab[:, 0] / self.s_lab[:, 1]), np.log(sp[:, 0] / sp[:, 1])
        m = {"synth_MAPE": 100 * se.mean(), "synth_acc10": 100 * (se <= .1).mean(),
             "synth_bias": 100 * np.mean(np.log(sp / self.s_lab)), "synth_logratio_r": float(np.corrcoef(lr_t, lr_p)[0, 1]),
             "pseudo_MAPE": 100 * pe.mean(), "pseudo_acc10": 100 * (pe <= .1).mean(),
             "pseudo_bias": 100 * np.mean(np.log(pp / self.p_lab))}
        m["score"] = 0.5 * (m["synth_MAPE"] + m["pseudo_MAPE"])
        return {k: float(v) for k, v in m.items()}


# =============================================================================================
# train
# =============================================================================================
def cmd_train(a):
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"[{a.name}] already done")
        return
    logf = open(os.path.join(out, "train.log"), "a")

    def log(*s):
        msg = f"{datetime.now():%H:%M:%S} " + " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dev = torch.device("cuda" if torch.cuda.is_available() else C.get_device("auto"))
    torch.backends.cudnn.benchmark = True
    C.seed_all(a.seed)
    drop = [d for d in a.drop.split(",") if d]
    s_items = synth_aug_items(a.synth) if a.real_frac < 1 else []
    r_items = pseudo_items("train") if a.real_frac > 0 else []
    items = s_items + r_items
    ds = AP.make_dataset(items, a.policy, a.tpi, seed=a.seed, aniso=a.aniso, drop=drop)
    total = a.steps * a.batch
    seq = index_sequence(len(s_items) * a.tpi, len(r_items) * a.tpi, a.real_frac, total, a.seed)

    model = C.build_model(a.backbone, pretrained=(a.init == "imagenet"))
    if a.init != "imagenet":
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        assert ck["backbone"] == a.backbone, (ck["backbone"], a.backbone)
        model.load_state_dict(ck["state"])
    model.to(dev).train()
    params = [{"params": model.body.parameters(), "lr": a.lr},
              {"params": list(model.count.parameters()) + list(model.weave.parameters()), "lr": a.lr}]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    step = 0
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        log(f"resumed at step {step}")
    meta = {k: v for k, v in vars(a).items() if k != "func"}
    meta.update({"n_synth": len(s_items), "n_real_scenes": len(r_items),
                 "n_real_files": sum(len(g) for g in r_items), "split": "scene", "device": str(dev),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    log(f"[{a.name}] {a.backbone} init={a.init} policy={a.policy} drop={drop} synth={len(s_items)} "
        f"real={len(r_items)} scenes ({sum(len(g) for g in r_items)} files) rho={a.real_frac} steps={a.steps} "
        f"lr={a.lr} on {dev}")
    validator = Validator(a.n_synth_val, log) if (a.eval_every or a.eval_final) else None
    val_rows = pd.read_csv(os.path.join(out, "val.csv")).to_dict("records") if (
        step and os.path.exists(os.path.join(out, "val.csv"))) else []

    def do_eval(st):
        t1 = time.time()
        m = validator(model, dev)
        snap = os.path.join(out, f"snap_{st:05d}.pt")
        C.save_ckpt(model, snap, {"step": st, "name": a.name, **m})
        val_rows.append({"step": st, "snapshot": os.path.basename(snap), **m})
        pd.DataFrame(val_rows).to_csv(os.path.join(out, "val.csv"), index=False)
        log(f"  eval step {st}: synth MAPE {m['synth_MAPE']:.2f} (acc10 {m['synth_acc10']:.1f}, r {m['synth_logratio_r']:.2f})"
            f"  pseudo MAPE {m['pseudo_MAPE']:.2f} (acc10 {m['pseudo_acc10']:.1f}, bias {m['pseudo_bias']:+.1f})"
            f"  score {m['score']:.2f}  ({time.time() - t1:.0f}s)")
        model.train()

    if step < a.steps:
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=SeqSampler(seq[step * a.batch:]),
                                         num_workers=a.workers, drop_last=True, worker_init_fn=worker_init,
                                         persistent_workers=False, prefetch_factor=4, pin_memory=True)
        base = [a.lr for _ in opt.param_groups]       # (not the resumed optimizer's scheduled lr)
        t0 = time.time()
        run_c = run_w = None
        n0 = step
        for x, y, wv, sw in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            wv, sw = wv.to(dev, non_blocking=True), sw.to(dev, non_blocking=True)
            sc = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
            for g_, b in zip(opt.param_groups, base):
                g_["lr"] = b * sc
            pc, pw = model(x)
            loss, lc, lw = C.loss_fn(pc, pw, y, wv, sw, a.lam_weave, a.lam_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_c = float(lc) if run_c is None else 0.95 * run_c + 0.05 * float(lc)
            run_w = float(lw) if run_w is None else 0.95 * run_w + 0.05 * float(lw)
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                log(f"  step {step}/{a.steps} count-loss {run_c:.4f} weave-loss {run_w:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                    f"{(step - n0) / el:.1f} it/s")
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
            if a.eval_every and step % a.eval_every == 0 and step < a.steps:
                do_eval(step)
            if step >= a.steps:
                break
        log(f"trained {step - n0} steps in {(time.time() - t0) / 60:.1f} min")
    model.eval()
    C.save_ckpt(model, os.path.join(out, "model.pt"), {**meta, "steps_done": step})
    if validator is not None and not any(r["step"] == step for r in val_rows):
        do_eval(step)
    if val_rows:
        v = pd.DataFrame(val_rows)
        b = v.loc[v.score.idxmin()]
        import shutil
        shutil.copyfile(os.path.join(out, b.snapshot), os.path.join(out, "best.pt"))
        json.dump({"best_step": int(b.step), **{k: float(b[k]) for k in v.columns if k not in ("step", "snapshot")}},
                  open(os.path.join(out, "best.json"), "w"), indent=1)
        log(f"best snapshot step {int(b.step)} score {b.score:.2f}")
    open(os.path.join(out, "DONE"), "w").write(datetime.now().isoformat())
    log("done")


def _val(run, which="best"):
    """Selection metrics of a finished run: 'best' snapshot (best.json) or the 'final' step."""
    d = os.path.join(RUNS, run)
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    v = pd.read_csv(os.path.join(d, "val.csv"))
    r = v.loc[v.score.idxmin()] if which == "best" else v.loc[v.step.idxmax()]
    ck = os.path.join(d, "best.pt" if which == "best" else "model.pt")
    return {"run": run, "ckpt": os.path.relpath(ck, HERE), "step": int(r.step),
            **{k: float(r[k]) for k in v.columns if k not in ("step", "snapshot")}}


V2_POLICY = {"V2A": "whitelist", "V2E": "whitelist_ext", "V2O": "whitelist_optics"}
POLICY_FAMILIES = {"whitelist": ("geometry", "photometric", "compression_blur", "illumination", "zoom", "tiling"),
                   "whitelist_ext": ("geometry", "photometric", "compression_blur", "illumination", "zoom", "tiling"),
                   "whitelist_optics": ("geometry", "photometric", "compression_blur", "illumination", "zoom",
                                        "tiling", "optics", "phone")}


def select_v2(bb):
    c2 = [x for x in (_val(f"{k}_{bb}", "final") for k in V2_POLICY) if x]
    best = min(c2, key=lambda x: x["score"])
    return best, V2_POLICY[best["run"].split("_")[0]], c2


def cmd_plan2(a):
    """Phase-2 tasks (after V1-V3 and the adopted runs are DONE): V4 from the selected V2 of each
    backbone with the same policy, and the stage-2 ablation of ResNet-18 -> runs_aug/tasks_phase2.json."""
    tasks = []
    chosen = {}
    for bb, backbone in (("r18", "resnet18"), ("b0", "efficientnet_b0")):
        best, pol, c2 = select_v2(bb)
        chosen[bb] = {"run": best["run"], "policy": pol, "score": best["score"],
                      "candidates": {x["run"]: round(x["score"], 3) for x in c2}}
        init = f"runs_aug/{best['run']}/model.pt"
        for rho, tag in ((1.0, "rho1"), (0.5, "rho05")):
            tasks.append({"id": f"V4_{bb}_{tag}", "deps": [],
                          "cmd": ["train", "--name", f"V4_{bb}_{tag}", "--backbone", backbone, "--workers", str(a.workers),
                                  "--init", init, "--policy", pol, "--real-frac", str(rho), "--steps", "2000",
                                  "--lr", "1e-4", "--eval-every", "250"]})
    pol = chosen["r18"]["policy"]
    init = f"runs_aug/{chosen['r18']['run']}/model.pt"
    # PREREG amendment_1: 1000-step stage-2 ablation (with its own seed-0 reference) for 'whitelist_optics'
    ab_steps = "1000" if pol == "whitelist_optics" else "2000"
    base = ["--backbone", "resnet18", "--workers", str(a.workers), "--init", init, "--policy", pol,
            "--real-frac", "{rho_r18}", "--steps", ab_steps, "--lr", "1e-4", "--eval-every", "250"]
    seeds = (0, 1, 2) if ab_steps == "1000" else (1, 2)
    chosen["ablation_stage2"] = {"steps": int(ab_steps), "seeds": list(seeds), "families": list(POLICY_FAMILIES[pol])}
    for sd in seeds:
        tasks.append({"id": f"ABLFT_seed{sd}_r18", "deps": ["V4_r18_rho1", "V4_r18_rho05"],
                      "cmd": ["train", "--name", f"ABLFT_seed{sd}_r18"] + base + ["--seed", str(sd)]})
    for fam in POLICY_FAMILIES[pol]:
        tasks.append({"id": f"ABLFT_{fam}_r18", "deps": ["V4_r18_rho1", "V4_r18_rho05"],
                      "cmd": ["train", "--name", f"ABLFT_{fam}_r18"] + base + ["--drop", fam]})
    json.dump(tasks, open(os.path.join(RUNS, "tasks_phase2.json"), "w"), indent=1)
    json.dump(chosen, open(os.path.join(RUNS, "v2_choice.json"), "w"), indent=1)
    print(json.dumps(chosen, indent=1))
    print(f"wrote {len(tasks)} tasks -> runs_aug/tasks_phase2.json")


def cmd_select(a):
    """Apply the pre-declared selection rules (PREREG.json) -> selection.json + final_spec.json."""
    sel, spec = {}, []
    for bb in ("r18", "b0"):
        sel[f"V1_{bb}"] = {**_val(f"V1_{bb}", "final"), "candidates": ["V1 (fixed recipe)"]}
        best2, pol2, c2 = select_v2(bb)
        sel[f"V2_{bb}"] = {**best2, "policy": pol2, "candidates": c2}
        c3 = [x for x in (_val(f"V3E_{bb}", "best"), _val(f"V3O_{bb}", "best")) if x]
        sel[f"V3_{bb}"] = {**min(c3, key=lambda x: x["score"]), "candidates": c3}
        c4 = [x for x in (_val(f"V4_{bb}_rho1", "best"), _val(f"V4_{bb}_rho05", "best")) if x]
        sel[f"V4_{bb}"] = {**min(c4, key=lambda x: x["score"]), "candidates": c4}
        sel[f"V5_{bb}"] = {**sel[f"V4_{bb}"], "tta": True, "candidates": ["= V4 + TTA (no selection)"]}
        for k in ("V1", "V2", "V3", "V4", "V5"):
            spec.append({"name": f"{k}_{bb}", "ckpt": sel[f"{k}_{bb}"]["ckpt"], "tta": k == "V5"})
    spec.append({"name": "REF_mps_r18", "ckpt": "runs/synth_r18_v2mix_aniso/model.pt", "tta": False,
                 "note": "existing MPS checkpoint re-predicted on the GPU: parity check on real150 only"})
    json.dump(sel, open(os.path.join(RUNS, "selection.json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(RUNS, "final_spec.json"), "w"), indent=1)
    rows = []
    for k, v in sel.items():
        rows.append({"variant": k, "run": v["run"], "ckpt": v["ckpt"], "step": v["step"],
                     **{m: round(v[m], 3) for m in ("synth_MAPE", "synth_acc10", "synth_logratio_r", "pseudo_MAPE",
                                                    "pseudo_acc10", "pseudo_bias", "score")}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RUNS, "selection_table.csv"), index=False)
    print(t.to_string(index=False))


def cmd_adopt(a):
    """Adopt a SYNTHETIC-ONLY run of the superseded auto-group split (its training never read a real image, so the
    split change cannot affect it): copy model.pt / meta / log and evaluate the final step with the
    current (scene-split) validator -> runs_aug/<name>/{model.pt, val.csv, best.json, DONE}."""
    import shutil
    src = os.path.join(SUPERSEDED, "runs", a.name)
    out = os.path.join(RUNS, a.name)
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"[adopt] {a.name} done")
        return
    meta = json.load(open(os.path.join(src, "meta.json")))
    assert float(meta.get("real_frac", 0.0)) == 0.0 and meta.get("init", "imagenet") == "imagenet", meta
    assert os.path.exists(os.path.join(src, "DONE")), src
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(os.path.join(src, "model.pt"), os.path.join(out, "model.pt"))
    shutil.copyfile(os.path.join(src, "train.log"), os.path.join(out, "train_superseded_round.log"))
    meta.update({"adopted_from": os.path.relpath(src, HERE), "adopted": datetime.now().isoformat(timespec="seconds")})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else C.get_device("auto"))
    model = C.load_ckpt(os.path.join(out, "model.pt"), dev)
    m = Validator(a.n_synth_val)(model, dev)
    step = int(meta.get("steps", -1))
    pd.DataFrame([{"step": step, "snapshot": "model.pt", **m}]).to_csv(os.path.join(out, "val.csv"), index=False)
    json.dump({"best_step": step, **m}, open(os.path.join(out, "best.json"), "w"), indent=1)
    open(os.path.join(out, "DONE"), "w").write(datetime.now().isoformat())
    print(a.name, json.dumps({k: round(v, 3) for k, v in m.items()}))


def cmd_evalonly(a):
    """Selection metrics of an existing checkpoint (e.g. the MPS synthetic-only model) -> runs_aug/<name>/val.csv"""
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else C.get_device("auto"))
    model = C.load_ckpt(a.ckpt, dev)
    m = Validator(a.n_synth_val)(model, dev)
    pd.DataFrame([{"step": -1, "snapshot": a.ckpt, **m}]).to_csv(os.path.join(out, "val.csv"), index=False)
    json.dump({"best_step": -1, **m}, open(os.path.join(out, "best.json"), "w"), indent=1)
    open(os.path.join(out, "DONE"), "w").write(datetime.now().isoformat())
    print(a.name, json.dumps(m))


# =============================================================================================
# final predictions (after ALL selection)
# =============================================================================================
@torch.no_grad()
def cmd_final(a):
    spec = json.load(open(a.spec))
    dev = torch.device("cuda" if torch.cuda.is_available() else C.get_device("auto"))
    od = os.path.join(RUNS, "final")
    os.makedirs(od, exist_ok=True)
    ps = pd.read_csv(PSEUDO_CSV)
    s2 = pd.read_csv(os.path.join(SYNTH_VAL2, "labels.csv"))
    s2_crops = [C.cap_side(cv2.imread(os.path.join(SYNTH_VAL2, f)), 1024) for f in s2.file]
    real = [load_real_crop(i) for i in range(150)]
    leg_names = sorted(os.path.splitext(f)[0] for f in os.listdir(LEGACY_CROPS) if f.endswith(".png"))
    leg = [C.cap_side(cv2.imread(os.path.join(LEGACY_CROPS, f"{n}.png")), 1024) for n in leg_names]
    for v in spec:
        name, ckpt, tta = v["name"], v["ckpt"], v.get("tta", False)
        vd = os.path.join(od, name)
        if os.path.exists(os.path.join(vd, "DONE")):
            print(f"[final] {name} done")
            continue
        os.makedirs(vd, exist_ok=True)
        model = C.load_ckpt(ckpt, dev)
        pred = (lambda c: AP.predict_crop_tta(model, c, dev)) if tta else (lambda c: C.predict_crop(model, c, dev))
        t0 = time.time()
        rows = []
        for i, c in enumerate(real):
            o = pred(c)
            rows.append(dict(image=i, pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                             pred_weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))]))
        r = pd.DataFrame(rows).merge(ps.rename(columns={
            "id": "image", "own_h": "ours_h", "own_v": "ours_v", "own_confidence": "ours_conf", "h": "pseudo_h",
            "v": "pseudo_v", "confidence": "scene_conf"})[
            ["image", "scene", "is_rep", "kind", "split", "use_train", "use_val", "ours_h", "ours_v", "ours_conf",
             "pseudo_h", "pseudo_v", "scene_conf"]], on="image")
        r.to_csv(os.path.join(vd, "real150_predictions.csv"), index=False)
        rows = []
        for n, c in zip(leg_names, leg):
            o = pred(c)
            rows.append(dict(name=n, pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                             pred_weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))]))
        pd.DataFrame(rows).to_csv(os.path.join(vd, "legacy_predictions_vlm.csv"), index=False)
        rows = []
        pred_s2 = pred if (a.synth_tta or not tta) else (lambda c: C.predict_crop(model, c, dev))
        for f, c, h, vv in zip(s2.file, s2_crops, s2.h_count, s2.v_count):
            o1 = C.predict_crop(model, c, dev, Ps=(768,), flips=False)
            o2 = pred_s2(c)
            rows.append(dict(file=f, h_count=h, v_count=vv, pred_h_single=o1["h"], pred_v_single=o1["v"],
                             pred_h=o2["h"], pred_v=o2["v"]))
        pd.DataFrame(rows).to_csv(os.path.join(vd, "synth_val_v2_predictions.csv"), index=False)
        json.dump({**v, "synth_val_v2_tta": bool(tta and a.synth_tta), "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                  open(os.path.join(vd, "meta.json"), "w"), indent=1)
        open(os.path.join(vd, "DONE"), "w").write(datetime.now().isoformat())
        print(f"[final] {name}: {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prep)
    p = sub.add_parser("prep_v0_autogroups")
    p.set_defaults(func=cmd_prep_v0_autogroups)
    p = sub.add_parser("ours_synth")
    p.add_argument("--synth-dir", default=SYNTH_VAL2)
    p.add_argument("--procs", type=int, default=16)
    p.set_defaults(func=cmd_ours_synth)
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--init", default="imagenet", help="'imagenet' or a checkpoint path")
    p.add_argument("--policy", default="whitelist_ext", choices=AP.POLICIES)
    p.add_argument("--drop", default="", help="comma list of families to drop (" + ",".join(AP.FAMILIES) + ")")
    p.add_argument("--synth", default=SYNTH_TRAIN)
    p.add_argument("--real-frac", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=4750)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tpi", type=int, default=4, help="tiles per image (index multiplicity)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--aniso", type=float, default=0.5)
    p.add_argument("--lam-ratio", type=float, default=1.0)
    p.add_argument("--lam-weave", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-final", type=int, default=1)
    p.add_argument("--n-synth-val", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("select")
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("plan2")
    p.add_argument("--workers", type=int, default=24)
    p.set_defaults(func=cmd_plan2)
    p = sub.add_parser("adopt")
    p.add_argument("--name", required=True)
    p.add_argument("--n-synth-val", type=int, default=None)
    p.set_defaults(func=cmd_adopt)
    p = sub.add_parser("evalonly")
    p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-synth-val", type=int, default=None)
    p.set_defaults(func=cmd_evalonly)
    p = sub.add_parser("final")
    p.add_argument("--spec", required=True, help="JSON list of {name, ckpt, tta}")
    p.add_argument("--synth-tta", type=int, default=0,
                   help="1: TTA also on synth_val_v2 for TTA variants (costly); 0: the standard 3 scales x 3 flips")
    p.set_defaults(func=cmd_final)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

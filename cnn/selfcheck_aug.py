#!/usr/bin/env python
"""selfcheck_aug.py -- self-check of the augmentation / pseudo-label training job (scene-split round).

Checks (each prints PASS / FAIL / INFO; the JSON goes to runs_aug/report/selfcheck.json):
  A  scene grouping   pseudo_labels.csv covers the 150 files in the 76 provenance scenes; every file of
                      a scene carries the scene's split; no scene in both pseudo-train and pseudo-val;
                      pseudo-val = real-capture representatives only; notebook-synthetic files are
                      never held out; orientation parity agrees with the provenance rot90_k
  B  no legacy leakage (code) train_aug.py reads the legacy crops only inside cmd_final; neither
                      train_aug.py nor aug_pipeline.py imports the legacy GT (legacy_unified / load_gt)
  C  no legacy leakage (data) no legacy crop is a copy of any of the 150 crops (band-pass NCC of the
                      crops, max over the 8 dihedral transforms, vs the same-scene distribution)
  D  selection is legacy-free: selection.json is reproduced from the runs' val.csv files alone (the
                      selection metrics), and the legacy predictions were written after it
  E  fabric-level near-duplicates across splits (INFO): different provenance scenes whose crops look
                      alike (band-pass NCC > 0.9) -- none may straddle pseudo-train / pseudo-val
  F  the frozen counter methods/results/counting_frozen_final.py is byte-identical (md5)

    <venv_cnn>/bin/python selfcheck_aug.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs_aug")
CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
LEGACY = os.path.join(ROOT, "vlm", "crops", "legacy")
FROZEN = os.path.join(ROOT, "methods", "results", "counting_frozen_final.py")
FROZEN_MD5 = "3d0520d6bec8a016543ce7e5b353d784"
OUT = os.path.join(RUNS, "report", "selfcheck.json")
RES = {}


def say(tag, ok, msg, **kw):
    st = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{st}] {tag}: {msg}")
    RES[tag] = {"status": st, "msg": msg, **kw}


def bp_thumb(path, n=128):
    """Band-pass (removes the yarn texture AND the smooth pick-glass vignetting) normalised thumbnail."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    bp = cv2.GaussianBlur(im, (0, 0), 512 / 100) - cv2.GaussianBlur(im, (0, 0), 512 / 12)
    t = cv2.resize(bp, (n, n), interpolation=cv2.INTER_AREA)[12:-12, 12:-12]
    return (t - t.mean()) / (t.std() + 1e-6)


def ncc8(a, b):
    return max(float((np.rot90(np.fliplr(a) if f else a, k) * b).mean()) for k in range(4) for f in (False, True))


def main():
    # ------------------------------------------------------------------ A scene grouping
    prov = json.load(open(os.path.join(ROOT, "camera_aug", "originals_provenance.json")))
    p = pd.read_csv(os.path.join(RUNS, "pseudo_labels.csv"))
    pre = json.load(open(os.path.join(RUNS, "PREREG.json")))
    groups = {frozenset(g) for g in prov["groups"]}
    mine = {frozenset(g.id) for _, g in p.groupby("scene")}
    say("A1_scenes_match_provenance", len(p) == 150 and mine == groups and len(mine) == 76,
        f"{len(p)} files, {len(mine)} scenes, identical to originals_provenance.json groups: {mine == groups}")
    one_split = p.groupby("scene").split.nunique().max() == 1
    say("A2_one_split_per_scene", bool(one_split), "every file of a scene carries the scene's split")
    tr, va = set(p.scene[p.use_train]), set(p.scene[p.use_val])
    say("A3_train_val_disjoint", not (tr & va), f"{len(tr)} pseudo-train scenes, {len(va)} pseudo-val scenes, "
        f"overlap {sorted(tr & va)}")
    v = p[p.use_val]
    say("A4_val_reps_real_only", bool(v.is_rep.all() and (v.kind == "real").all() and v.scene.is_unique),
        f"pseudo-val = {len(v)} representatives, kinds {v.kind.value_counts().to_dict()}")
    nb = p[p.kind == "nb_synth"]
    say("A5_nb_synth_never_heldout", not nb.use_val.any(),
        f"notebook-synthetic files: {len(nb)}; used as training variants of their scene: {int(nb.use_train.sum())}; "
        "held out: 0")
    cut = pre["split_info"]["conf_median_76_reps"]
    o = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    reps = sorted(p.scene.unique())
    acc = {s for s in reps if o.confidence[s] >= cut and o.f_fused_lattice[s] == 1}
    say("A6_confidence_rule_reproduced", acc == set(p.scene[p.accepted]) and abs(np.median(o.confidence[reps]) - cut) < 1e-12,
        f"pre-declared cut = median ours' confidence over the 76 representatives = {cut:.4f}; "
        f"{len(acc)} accepted scenes")
    m = pd.read_csv(os.path.join(HERE, "data", "scene_map.csv"))
    pk = m[m.prov_rot_k.notna()]
    agree = int((pk.prov_rot_k.astype(int) % 2 == pk.rot_k % 2).sum())
    say("A7_orientation_parity", agree == len(pk), f"h/v swap parity vs provenance rot90_k: {agree}/{len(pk)}")

    # ------------------------------------------------------------------ B code paths
    src = open(os.path.join(HERE, "train_aug.py")).read()
    fn = {}
    for mt in re.finditer(r"^def (\w+)\(.*?(?=^def |\Z)", src, re.S | re.M):
        fn[mt.group(1)] = mt.group(0)
    users = sorted(k for k, body in fn.items() if "LEGACY_CROPS" in body)
    pat = re.compile(r"^\s*(import|from)\s+[\w.]*legacy_unified|load_gt\(|legacy13_gt|legacy_labels", re.M)
    gt_import = [f for f in ("train_aug.py", "aug_pipeline.py", "augment.py", "augment_optics.py", "cnn_common.py",
                             "run_queue.py") if pat.search(open(os.path.join(HERE, f)).read())]
    say("B1_legacy_only_in_final", users == ["cmd_final"] and not gt_import,
        f"functions reading the legacy crops: {users}; legacy GT imported by training code: {gt_import}")

    # ------------------------------------------------------------------ C image overlap legacy vs 150
    real = {i: bp_thumb(os.path.join(CROPS, f"{i}.jpg")) for i in range(150)}
    same = [ncc8(real[r.id], real[r.scene]) for r in m.itertuples() if r.id != r.scene]
    leg = {}
    for f in sorted(os.listdir(LEGACY)):
        if f.endswith(".png"):
            a = bp_thumb(os.path.join(LEGACY, f))
            s, i = max((ncc8(a, b), i) for i, b in real.items())
            leg[f] = (round(s, 3), i)
    mx = max(s for s, _ in leg.values())
    say("C1_no_legacy_copy_in_150", mx < min(same),
        f"max band-pass NCC legacy vs 150 = {mx:.3f} (best match {max(leg.items(), key=lambda kv: kv[1][0])}); "
        f"same-scene pairs: min {min(same):.3f}, median {np.median(same):.3f}", per_legacy=leg)

    # ------------------------------------------------------------------ D selection reproducible from val.csv only
    sp = os.path.join(RUNS, "selection.json")
    if os.path.exists(sp):
        sel = json.load(open(sp))

        def val(run, which):
            d = pd.read_csv(os.path.join(RUNS, run, "val.csv"))
            r = d.loc[d.score.idxmin()] if which == "best" else d.loc[d.step.idxmax()]
            return float(r.score), run

        ok = True
        for bb in ("r18", "b0"):
            want = {"V2": min(val(f"{k}_{bb}", "final") for k in ("V2A", "V2E", "V2O"))[1],
                    "V3": min(val(f"{k}_{bb}", "best") for k in ("V3E", "V3O"))[1],
                    "V4": min(val(f"V4_{bb}_{t}", "best") for t in ("rho1", "rho05"))[1]}
            for k, run in want.items():
                ok &= sel[f"{k}_{bb}"]["run"] == run
        fin = os.path.join(RUNS, "final")
        t_sel = os.path.getmtime(sp)
        legs = [os.path.join(fin, d, "legacy_predictions_vlm.csv") for d in os.listdir(fin)] if os.path.isdir(fin) else []
        after = all(os.path.getmtime(f) >= t_sel for f in legs if os.path.exists(f))
        say("D1_selection_from_val_only", bool(ok and after),
            f"selection.json reproduced from val.csv: {bool(ok)}; all {len(legs)} legacy prediction files written "
            f"after selection.json: {after}")
    else:
        say("D1_selection_from_val_only", None, "selection.json not yet written")

    # ------------------------------------------------------------------ E fabric-level near duplicates
    R = {i: real[i] for i in reps}
    pairs = []
    for a, b in itertools.combinations(reps, 2):
        s = ncc8(R[a], R[b])
        if s > 0.9:
            sa, sb = p.loc[p.id == a, "split"].iloc[0], p.loc[p.id == b, "split"].iloc[0]
            pairs.append({"scene_a": a, "scene_b": b, "ncc": round(s, 3), "split_a": sa, "split_b": sb,
                          "ours_a": [round(o.h_count[a], 1), round(o.v_count[a], 1)],
                          "ours_b": [round(o.h_count[b], 1), round(o.v_count[b], 1)]})
    straddle = [q for q in pairs if {q["split_a"], q["split_b"]} == {"train", "val"}]
    say("E1_fabric_near_duplicates_not_straddling", not straddle,
        f"{len(pairs)} pairs of different scenes with band-pass NCC > 0.9 (same fabric re-photographed?); "
        f"straddling pseudo-train / pseudo-val: {len(straddle)}", pairs=pairs)
    vmax = {}
    for s in sorted(va):
        vmax[int(s)] = round(max(ncc8(real[s], real[t]) for t in p.id[p.use_train]), 3)
    say("E2_val_vs_train_similarity", None, f"max band-pass NCC of each held-out representative to any training "
        f"file: {vmax} (same-scene 5th percentile {np.percentile(same, 5):.3f})")

    # ------------------------------------------------------------------ F frozen counter
    md5 = hashlib.md5(open(FROZEN, "rb").read()).hexdigest()
    say("F1_frozen_counter_unchanged", md5 == FROZEN_MD5, f"md5 {md5}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(RES, open(OUT, "w"), indent=1, default=str)
    n_fail = sum(r["status"] == "FAIL" for r in RES.values())
    print(f"{len(RES)} checks, {n_fail} FAIL -> {OUT}")


if __name__ == "__main__":
    main()

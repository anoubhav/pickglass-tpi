"""scenes76_labelfree.py -- label-free statistics of the paper re-computed per SCENE.

The 150 files of PickGlass are 76 distinct scenes (camera_aug/originals_provenance.json;
annotation_tool/data/manifest.json `scenes`).  Every label-free number that the paper quotes "on the
150 photos" is re-computed here on the 76 scene representatives (one row per scene, the rep =
highest-resolution real JPEG), next to the old per-file value.  No thread label is read anywhere.

Inputs (read-only):
  annotation_tool/data/manifest.json             scene fields, kinds, reps, IAA v2
  paper/tables/strata150.csv                     S, side ratio, saturation per file
  methods/results/all150_degrade.csv             28 controlled degradations (ours, fft, ...)
  methods/results/all150_crops.csv               legacy-pipeline segmentation crops
  methods/results/ruler/{calib150,count150}.csv  ruler calibration and downstream counts
  paper/tables/cascade_150_decisions.csv         frozen cascade decisions on the clean photos
  camera_aug/results/label_audit_pairs.csv       real full-res / quarter-res pairs (same exposure)
Outputs:
  paper/tables/scenes76_labelfree.json           every number (per scene, and per file for reference)
  paper/tables/quality76.csv                     Tab. 3 cells per scene (ours, fft) + per file
  paper/tables/fabric_groups_candidate.csv       scene -> candidate fabric group (13 suspected re-shoot pairs,
                                                 from cnn/runs_aug/report/selfcheck.json E1), for a sensitivity analysis

    <venv2>/bin/python tools/scenes76_labelfree.py
Deterministic (bootstrap seed 0, 10,000 resamples; Wilson intervals for proportions).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
B = 10000
RNG = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def boot_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    idx = RNG.integers(0, len(x), (B, len(x)))
    v = np.median(x[idx], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def main():
    man = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    imgs = pd.DataFrame(man["images"])
    reps = sorted(int(i) for i in man["scenes"]["reps"])
    assert len(reps) == 76
    out = {"source": "paper/tools/scenes76_labelfree.py", "n_files": int(len(imgs)), "n_scenes": len(reps)}

    # ------------------------------------------------------------------ dataset structure
    kinds = imgs.kind.value_counts().to_dict()
    out["kinds"] = {k: int(v) for k, v in kinds.items()}
    imgs["long"] = imgs[["orig_w", "orig_h"]].max(axis=1)
    rep = imgs[imgs.id.isin(reps)].copy()
    rep["full"] = rep["long"] >= 3000
    st = pd.read_csv(os.path.join(PAPER, "tables", "strata150.csv"))
    rep = rep.merge(st[["id", "S", "side_ratio", "sat240"]], on="id")
    out["reps_full"] = int(rep.full.sum())
    out["reps_quarter_only"] = int((~rep.full).sum())
    for nm, g in (("full", rep[rep.full]), ("quarter", rep[~rep.full])):
        out[f"S_{nm}"] = {"min": float(g.S.min()), "median": float(g.S.median()), "max": float(g.S.max())}
    out["side_ratio_max_reps"] = float(rep.side_ratio.max())
    out["side_ratio_gt105_reps"] = int((rep.side_ratio > 1.05).sum())
    out["sat240_gt1pct_reps"] = int((rep.sat240 > 0.01).sum())
    out["sat240_max_reps"] = float(rep.sat240.max())
    real = imgs[imgs.kind != "notebook_synthetic"]
    out["real_files"] = int(len(real))
    sizes = real.groupby(["orig_w", "orig_h"]).size()
    out["real_file_sizes"] = {f"{w}x{h}": int(n) for (w, h), n in sizes.items()}
    mem = imgs.groupby("scene_id").size()
    out["scene_size_hist"] = {int(k): int(v) for k, v in mem.value_counts().sort_index().items()}
    # scenes that have both a full-res and a quarter-res copy (the controlled resolution pairs)
    q = imgs[imgs.kind == "quarter_res_copy"]
    out["scenes_with_quarter_copy"] = int(q.scene_id.nunique())
    out["swap_hv_real_copies"] = int(imgs[(~imgs.is_scene_rep) & (imgs.kind != "notebook_synthetic") & imgs.scene_swap_hv].shape[0])
    out["swap_hv_notebook"] = int(imgs[(imgs.kind == "notebook_synthetic") & imgs.scene_swap_hv].shape[0])
    out["iaa_v2"] = man["iaa_subset"]
    iaa = rep[rep.id.isin(man["iaa_subset"])]
    out["iaa_v2_full"] = int(iaa.full.sum())
    out["iaa_v2_quarter_only"] = int((~iaa.full).sum())

    # ------------------------------------------------------------------ Tab. 3: self-consistency
    deg = pd.read_csv(os.path.join(ROOT, "methods", "results", "all150_degrade.csv"))
    if "error" in deg:
        deg = deg[deg.error.isna()]
    keys = ["image", "crop", "preproc", "glare", "estimator"]
    clean = deg[deg.kind == "clean"][keys + ["h", "v", "conf"]].rename(columns={"h": "h0", "v": "v0", "conf": "conf0"})
    d = deg[deg.kind != "clean"].merge(clean, on=keys)
    d["absdrift"] = np.fmax(np.abs(d.h / d.h0 - 1), np.abs(d.v / d.v0 - 1)).fillna(np.inf)
    d["over2"] = (np.fmax(d.h / d.h0, d.v / d.v0) > 2).fillna(False)
    rows = []
    for (est, kind, level), g in d[d.estimator.isin(["ours", "fft"])].groupby(["estimator", "kind", "level"]):
        gr = g[g.image.isin(reps)]
        k = int((gr.absdrift <= 0.05).sum())
        hi = gr[gr.image.isin(rep[rep.full].id)]
        lo = gr[gr.image.isin(rep[~rep.full].id)]
        rows.append(dict(estimator=est, kind=kind, level=level,
                         files_n=len(g), files_within5_pct=100 * float((g.absdrift <= .05).mean()),
                         scenes_n=len(gr), scenes_within5_pct=100 * k / len(gr),
                         scenes_within5_ci_lo=wilson(k, len(gr))[0], scenes_within5_ci_hi=wilson(k, len(gr))[1],
                         full_n=len(hi), full_within5_pct=100 * float((hi.absdrift <= .05).mean()),
                         quarter_n=len(lo), quarter_within5_pct=100 * float((lo.absdrift <= .05).mean()),
                         full_over2=int(hi.over2.sum()), quarter_over2=int(lo.over2.sum())))
    q76 = pd.DataFrame(rows)
    q76.to_csv(os.path.join(PAPER, "tables", "quality76.csv"), index=False, float_format="%.2f")
    out["tab3_max_abs_shift_files_vs_scenes_pp"] = float((q76.files_within5_pct - q76.scenes_within5_pct).abs().max())

    # confidence as a quality score (ours, pooled over the 28 conditions)
    o = d[(d.estimator == "ours") & d.image.isin(reps)].copy()
    ad = np.where(np.isfinite(o.absdrift), o.absdrift, 1e3)
    out["conf_spearman_pooled_scenes"] = float(stats.spearmanr(o.conf, ad)[0])
    o_all = d[d.estimator == "ours"]
    out["conf_spearman_pooled_files"] = float(stats.spearmanr(o_all.conf, np.where(np.isfinite(o_all.absdrift), o_all.absdrift, 1e3))[0])
    per = []
    for (kind, level), g in o.groupby(["kind", "level"]):
        a_ = np.where(np.isfinite(g.absdrift), g.absdrift, 1e3)
        if np.std(g.conf) > 0:
            per.append(float(stats.spearmanr(g.conf, a_)[0]))
    out["conf_spearman_within_condition_median_scenes"] = float(np.nanmedian(per))
    out["conf_spearman_within_condition_range_scenes"] = [float(np.nanmin(per)), float(np.nanmax(per))]
    qt = pd.qcut(o.conf.rank(method="first"), 5, labels=False)
    w5 = (o.absdrift <= .05)
    out["conf_quintile_within5_pct_scenes"] = [100 * float(w5[qt == i].mean()) for i in range(5)]
    out["n_degraded_crops_scenes"] = int(len(o))

    # ------------------------------------------------------------------ ruler, per scene
    cal = pd.read_csv(os.path.join(ROOT, "methods", "results", "ruler", "calib150.csv"))
    cnt = pd.read_csv(os.path.join(ROOT, "methods", "results", "ruler", "count150.csv"))
    seg = pd.read_csv(os.path.join(ROOT, "methods", "results", "all150_crops.csv"))
    seg = seg[(seg.estimator == "ours") & (seg.kind == "clean")]
    ref = seg[seg.crop == "gt"][["image", "h", "v"]].rename(columns={"h": "h0", "v": "v0"})
    sg = seg[seg.crop == "seg_final"].merge(ref, on="image")
    sg["absdrift_seg"] = np.fmax((sg.h / sg.h0 - 1).abs(), (sg.v / sg.v0 - 1).abs()).fillna(np.inf)
    cr = cal[cal.id.isin(reps)]
    ok = cr[cr.ok.astype(bool)]
    kc = cnt[cnt.id.isin(reps)].merge(sg[["image", "absdrift_seg"]], left_on="id", right_on="image")
    a5 = kc.absdrift.fillna(np.inf) <= .05
    s5 = kc.absdrift_seg <= .05
    out["ruler_scenes"] = {
        "n": int(len(cr)), "calibrated": int(len(ok)),
        "corner_err_median_mm": float(ok.cerr_mean.median()), "corner_err_ci": boot_median(ok.cerr_mean),
        "within5_pct": 100 * float(a5.mean()), "within5_ci": wilson(int(a5.sum()), len(kc)),
        "seg_within5_pct": 100 * float(s5.mean()), "seg_within5_ci": wilson(int(s5.sum()), len(kc)),
        "mcnemar_ruler_vs_seg": mcnemar(a5, s5),
        "pitch_ratio_median": float(ok.r10_free.median()), "pitch_ratio_ci": boot_median(ok.r10_free),
        "W_div_median": float(ok.W_free.median()), "W_div_ci": boot_median(ok.W_free),
    }

    # ------------------------------------------------------------------ frozen cascade on the clean photos
    cas = pd.read_csv(os.path.join(PAPER, "tables", "cascade_150_decisions.csv"))
    cs = cas[cas.image.isin(reps)]
    out["cascade_clean_escalated_scenes"] = [int(cs.escalate_gate.sum()), int(len(cs))]
    out["cascade_clean_escalated_files"] = [int(cas.escalate_gate.sum()), int(len(cas))]

    # ------------------------------------------------------------------ controlled resolution pairs
    pr = pd.read_csv(os.path.join(ROOT, "camera_aug", "results", "label_audit_pairs.csv"))
    pr["ad"] = pr.drift.abs()
    sc = pr.scene.unique()
    by = {s: pr[pr.scene == s] for s in sc}
    med, gt10 = [], []
    for _ in range(B):
        pick = RNG.choice(sc, len(sc))
        x = pd.concat([by[s] for s in pick])
        med.append(np.median(x.ad))
        gt10.append(100 * (x.ad > .10).mean())
    out["pairs"] = {"n_pairs": int(pr[["full_res", "quarter_res"]].drop_duplicates().shape[0]),
                    "n_scenes": int(len(sc)), "n_counts": int(len(pr)),
                    "median_absdrift_pct": 100 * float(pr.ad.median()),
                    "median_absdrift_scene_ci": [100 * float(np.percentile(med, 2.5)), 100 * float(np.percentile(med, 97.5))],
                    "gt10_counts": int((pr.ad > .10).sum()),
                    "gt10_pct": 100 * float((pr.ad > .10).mean()),
                    "gt10_scene_ci": [float(np.percentile(gt10, 2.5)), float(np.percentile(gt10, 97.5))],
                    "median_signed_pct": 100 * float(pr.drift.median()),
                    "ppt_quarter_median": float(pr.ppt_quarter.median()),
                    "S_quarter_median": float(pr.S_quarter.median()), "S_full_median": float(pr.S_full.median())}
    # ------------------------------------------------------------------ candidate fabric groups (label-free)
    # 13 pairs of DIFFERENT scenes that look like the same fabric re-photographed (band-pass NCC > 0.9;
    # cnn/selfcheck_aug.py check E1). Union-find over them gives a stricter, fabric-level grouping for a
    # sensitivity analysis against the human counts (never used to choose anything).
    sc = json.load(open(os.path.join(ROOT, "cnn", "runs_aug", "report", "selfcheck.json")))
    pairs = [(int(q["scene_a"]), int(q["scene_b"])) for q in sc["E1_fabric_near_duplicates_not_straddling"]["pairs"]]
    par = {s: s for s in reps}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a_, b_ in pairs:
        par[find(a_)] = find(b_)
    fg = pd.DataFrame({"scene_id": reps, "fabric_group": [min(s for s in reps if find(s) == find(r)) for r in reps]})
    fg.to_csv(os.path.join(PAPER, "tables", "fabric_groups_candidate.csv"), index=False)
    out["fabric_pairs"] = pairs
    out["fabric_groups"] = int(fg.fabric_group.nunique())
    with open(os.path.join(PAPER, "tables", "scenes76_labelfree.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    show = q76[["estimator", "kind", "level", "files_within5_pct", "scenes_within5_pct", "full_within5_pct",
                "quarter_within5_pct", "full_over2", "quarter_over2"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()

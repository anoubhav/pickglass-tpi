"""Score every method on the 150 photos against the human counts -> paper tables.

Fills Table 4 (tab:main150), the resolution / quality-flag stratification (tab:strat150)
and the coverage-accuracy curve (fig:coverage) once the weekend annotations exist.

Inputs (all read-only):
  --labels   one or more annotation-tool CSV exports (annotation_tool "Export CSV"),
             comma-separated or repeated. Records with done=1 and nonwoven=0 are used.
             Counts: h_tpi / v_tpi (tool v2, normalised by the rotated guide-line length)
             when present, else h_count / v_count.
  predictions, from the other jobs:
    methods/predictions_ours_150.csv            ours + fft/profile/tiles/gabor/lattice columns
    vlm/results/rect150_P1_predictions_wide.csv  7 API VLMs, prompt P1
        NOTE: VLM "<model>__v" = VERTICAL yarns = our h; "<model>__h" = horizontal = our v.
    cnn/runs/synth_r18_v2mix_aniso/eval_final/real150_predictions.csv   CNN zero-shot
    cnn/runs/cv_real/predictions.csv  (optional; 5-fold CV, written by cnn/train_cv.py)
    --extra-wide PATH[::keys]          (optional) other wide CSVs in the VLM format, e.g.
        gpu_results/countgd/outputs/results/rect150_predictions_wide.csv::countgd,countgd@CG-W-ex3-lat

Axes are unambiguous on the 150: annotators count on the same rectified crop, so
h (vertical yarns crossing the horizontal guide line) is compared with each method's h.

Ground truth per image = mean over annotators (default), or --blind-only (records whose
proposals were never shown), or --annotator NAME.

SCENES (Oct 8 2026, before any human count): the 150 files are 76 scenes (annotation_tool manifest
fields scene_id / is_scene_rep / kind / scene_rot_k; camera_aug/originals_provenance.json). Only the
scene representative is annotated and annotation_tool/propagate_scene_labels.py copies its labels to
the other files (h <-> v for copies stored turned 90 / 270 deg).
  --unit scene (DEFAULT): PRIMARY evaluation, one row per scene = the rep's photo (76 rows when all
        are labelled); main150.csv, main150_rows.tex, strat150.csv, coverage_ours.csv as before.
        SECONDARY per-file evaluation is always written to main150_perfile.csv: every real file of a
        labelled scene (139 = 150 minus the 11 notebook_synthetic files, which are augmentations and
        never test items), scene-clustered bootstrap CI, plus per-resolution and per-kind rows.
  --unit image: the pre-Oct-8 behaviour (one row per labelled id, labels not propagated; --dedup
        then drops the copies listed in methods/results/ruler/duplicate_groups.json).
IAA membership ("blind IAA subset" in human_agreement.csv) is taken from the manifest's current
iaa_subset (version 2 = 20 reps of distinct scenes), not from the per-record flag.

Pre-registered on Oct 7 2026 (Job 6), before any human count existed: main150.csv also carries
  both5 / both10  share of IMAGES with both counts (h and v) within 5% / 10% of the reference,
  bias_med        median signed relative error over the 2n counts (%), and spread_mad = 1.4826 MAD of it,
and --dedup scores one photo per group of copies (methods/results/ruler/duplicate_groups.json: the 150 files
hold 121 distinct exposures; the lowest id of each group is kept). main150_rows.tex keeps its 6 columns.

Outputs (default --out paper/tables): main150.csv, main150_rows.tex, strat150.csv,
coverage_ours.csv, human_agreement.csv, data150_summary.json and figs/fig_coverage.pdf (with --fig).

Run with the shared venv2 python (pandas, numpy, matplotlib):
  <venv2>/bin/python tools/score150.py --labels ../annotation_tool/annotations_A.csv,../annotation_tool/annotations_B.csv --fig
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)

P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_VLM = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")
P_CNN0 = os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso", "eval_final", "real150_predictions.csv")
P_CNNCV = os.path.join(ROOT, "cnn", "runs", "cv_real", "predictions.csv")
MANIFEST = os.path.join(ROOT, "annotation_tool", "data", "manifest.json")
P_DUP = os.path.join(ROOT, "methods", "results", "ruler", "duplicate_groups.json")


def scene_module(manifest_path=MANIFEST):
    """propagate_scene_labels.py of the annotation tool that owns the manifest (<tool>/data/manifest.json)."""
    import importlib.util
    tool = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    spec = importlib.util.spec_from_file_location("propagate_scene_labels", os.path.join(tool, "propagate_scene_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

VLM_NAMES = {
    "gpt-6.1-sol": "GPT-6.1-sol", "claude-fable-5.1": "Claude Fable 5.1",
    "gemini-3.1-pro": "Gemini 3.1 Pro", "gemini-3.8-flash": "Gemini 3.8 Flash",
    "qwen3.8-max-prime": "Qwen3.8-Max-Prime", "qwen3.8-27b": "Qwen3.8-27B (open)",
    "gemma-4-26b-a4b": "Gemma-4-26B-A4B (open)", "molmo2_8b": "Molmo2-8B (local)",
}
FLAGS = ["glare", "blur", "shadow", "low_res", "perspective", "occluded"]


def coarse_weave(w):
    if not isinstance(w, str) or not w:
        return None
    w = w.lower()
    if w.startswith("twill"):
        return "twill"
    if w in ("plain", "satin"):
        return w
    if w in ("unsure", "unknown", "none", "nan", "-1"):
        return None
    return "other"


def load_labels(paths, blind_only=False, annotator=None):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["src"] = os.path.basename(p)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # annotation tool v2: <ax>_tpi = dots / length (in) of the possibly rotated guide line inside
    # the 1-inch window (== <ax>_count for unrotated lines); use it, else the raw count (v1 CSVs)
    for ax in ("h", "v"):
        if f"{ax}_tpi" in d:
            d[f"{ax}_count_raw"] = d[f"{ax}_count"]
            d[f"{ax}_count"] = pd.to_numeric(d[f"{ax}_tpi"], errors="coerce").fillna(d[f"{ax}_count"])
        # a line the annotator cleared ("yarns impossible to see") has 0 dots: missing, not 0 TPI
        d[f"{ax}_count"] = pd.to_numeric(d[f"{ax}_count"], errors="coerce").where(lambda s: s > 0)
    d = d[(d["done"].astype(int) == 1) & (d.get("nonwoven", 0).astype(int) == 0)].copy()
    if annotator:
        d = d[d["annotator"] == annotator]
    if blind_only:
        d = d[d["proposals_seen"].astype(int) == 0]
    # one record per (image, annotator): keep the newest
    d = d.sort_values("timestamp").drop_duplicates(["id", "annotator"], keep="last")
    d["weave_c"] = d["weave"].map(coarse_weave)
    for f in FLAGS:
        col = "flag_" + f
        d[col] = d[col].astype(int) if col in d else 0
    return d


def gt_table(rec):
    g = rec.groupby("id")
    gt = pd.DataFrame({
        "h": g["h_count"].mean(), "v": g["v_count"].mean(),
        "n_ann": g["annotator"].nunique(),
        "weave": g["weave_c"].agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size else None),
        "iaa": g["iaa"].max(),
    })
    for f in FLAGS:
        gt["flag_" + f] = g["flag_" + f].max()
    return gt


def load_predictions():
    preds = {}
    o = pd.read_csv(P_OURS).set_index("id")
    preds["Ours (lattice + fusion)"] = pd.DataFrame({"h": o.h_count, "v": o.v_count, "weave": o.weave.map(coarse_weave),
                                                     "conf": o.confidence})
    for col, name in [("lattice", "Lattice only"), ("fft", "FFT max near-axis peak"), ("tiles", "FFT tile voting"),
                      ("profile", "Deskewed profile + YIN"), ("gabor", "Log-Gabor bank")]:
        if f"h_{col}" in o:
            preds[name] = pd.DataFrame({"h": o[f"h_{col}"], "v": o[f"v_{col}"]})
    if os.path.exists(P_VLM):
        w = pd.read_csv(P_VLM).set_index("image_id")
        for key, name in VLM_NAMES.items():
            if f"{key}__v" in w:
                preds[name + ", P1"] = pd.DataFrame({"h": w[f"{key}__v"], "v": w[f"{key}__h"],
                                                     "weave": w[f"{key}__weave"].map(coarse_weave)})
    if os.path.exists(P_CNN0):
        c = pd.read_csv(P_CNN0).set_index("image")
        preds["CNN, synthetic only (zero-shot)"] = pd.DataFrame({"h": c.pred_h, "v": c.pred_v,
                                                                 "weave": c.pred_weave.map(coarse_weave)})
    if os.path.exists(P_CNNCV):
        cv = pd.read_csv(P_CNNCV)
        for (bb, ex), g in cv.groupby(["backbone", "experiment"]):
            g = g.set_index("image")
            preds[f"CNN 5-fold CV: {bb}, {ex}"] = pd.DataFrame({"h": g.pred_h, "v": g.pred_v})
    return preds


def load_extra_wide(specs):
    """--extra-wide PATH[::key1,key2]: any other wide CSV in the VLM format (image_id,
    <key>__v, <key>__h[, <key>__weave]), e.g. the remote-GPU CountGD / Molmo2-bf16 runs in
    gpu_results/. As for the VLMs, <key>__v = VERTICAL yarns = our h. Without ::keys every
    key in the file is added."""
    out = {}
    for spec in specs or []:
        path, _, keys = spec.partition("::")
        w = pd.read_csv(path).set_index("image_id")
        avail = [c[:-3] for c in w.columns if c.endswith("__v") and c[:-3] + "__h" in w.columns]
        for k in ([x.strip() for x in keys.split(",") if x.strip()] or avail):
            if k not in avail:
                raise SystemExit(f"--extra-wide: no columns {k}__v/{k}__h in {path}")
            d = pd.DataFrame({"h": w[f"{k}__v"], "v": w[f"{k}__h"]})
            if f"{k}__weave" in w and w[f"{k}__weave"].notna().any():
                d["weave"] = w[f"{k}__weave"].map(coarse_weave)
            out[k] = d
    return out


def metrics(gt, pr, n_boot=2000, seed=0, cluster=None):
    """cluster (optional, Series aligned with gt.index): bootstrap whole clusters (scenes) instead of rows."""
    j = gt[["h", "v"]].join(pr[["h", "v"]], rsuffix="_p", how="left")
    have = j[["h_p", "v_p"]].notna().all(axis=1)
    jj = j[have]
    e = np.abs(np.r_[jj.h_p - jj.h, jj.v_p - jj.v])
    pe = 100 * e / np.r_[jj.h, jj.v]
    out = dict(n_gt=len(gt), n_scored=int(have.sum()), coverage=round(100 * have.mean(), 1))
    if len(jj) == 0:
        return out
    out.update(MAE=e.mean(), MAPE=pe.mean(), MdAPE=np.median(pe), w5=100 * np.mean(pe <= 5), w10=100 * np.mean(pe <= 10))
    # pre-registered Oct 7 (before labels): image-level tolerance (both counts) and signed bias
    pe_h = (100 * np.abs(jj.h_p - jj.h) / jj.h).to_numpy()
    pe_v = (100 * np.abs(jj.v_p - jj.v) / jj.v).to_numpy()
    se = 100 * np.r_[(jj.h_p - jj.h) / jj.h, (jj.v_p - jj.v) / jj.v]
    out.update(both5=100 * np.mean((pe_h <= 5) & (pe_v <= 5)), both10=100 * np.mean((pe_h <= 10) & (pe_v <= 10)),
               bias_med=float(np.median(se)), spread_mad=float(1.4826 * np.median(np.abs(se - np.median(se)))))
    # image-level bootstrap of MAPE (cluster bootstrap when clusters are given)
    rng = np.random.default_rng(seed)
    per_img = 0.5 * (100 * np.abs(jj.h_p - jj.h) / jj.h + 100 * np.abs(jj.v_p - jj.v) / jj.v).to_numpy()
    if cluster is None:
        bs = [per_img[rng.integers(0, len(per_img), len(per_img))].mean() for _ in range(n_boot)]
    else:
        cl = pd.Series(cluster).reindex(jj.index).to_numpy()
        groups = [np.where(cl == c)[0] for c in pd.unique(cl)]
        out["n_clusters"] = len(groups)
        bs = []
        for _ in range(n_boot):
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            bs.append(per_img[idx].mean())
    out.update(MAPE_lo=np.percentile(bs, 2.5), MAPE_hi=np.percentile(bs, 97.5))
    if "weave" in pr and "weave" in gt:
        k = gt["weave"].notna() & pr["weave"].reindex(gt.index).notna()
        if k.any():
            out["weave_acc"] = 100 * np.mean(gt.loc[k, "weave"] == pr["weave"].reindex(gt.index)[k])
            out["weave_n"] = int(k.sum())
    return out


def loo_mean(gt):
    s_h, s_v, n = gt.h.sum(), gt.v.sum(), len(gt)
    return pd.DataFrame({"h": (s_h - gt.h) / (n - 1), "v": (s_v - gt.v) / (n - 1)}, index=gt.index)


def human_agreement(rec):
    rows = []
    for subset, r in [("all common images", rec), ("blind IAA subset", rec[(rec.iaa.astype(int) == 1) &
                                                                          (rec.proposals_seen.astype(int) == 0)])]:
        ann = sorted(r.annotator.unique())
        for i in range(len(ann)):
            for k in range(i + 1, len(ann)):
                a = r[r.annotator == ann[i]].set_index("id")
                b = r[r.annotator == ann[k]].set_index("id")
                c = a.index.intersection(b.index)
                if len(c) == 0:
                    continue
                d = np.r_[a.loc[c, "h_count"] - b.loc[c, "h_count"], a.loc[c, "v_count"] - b.loc[c, "v_count"]]
                m = np.r_[(a.loc[c, "h_count"] + b.loc[c, "h_count"]) / 2, (a.loc[c, "v_count"] + b.loc[c, "v_count"]) / 2]
                rel = 100 * np.abs(d) / m
                rows.append(dict(subset=subset, pair=f"{ann[i]} vs {ann[k]}", n_images=len(c), MAE=np.abs(d).mean(),
                                 bias=d.mean(), loa_lo=d.mean() - 1.96 * d.std(ddof=1), loa_hi=d.mean() + 1.96 * d.std(ddof=1),
                                 w5=100 * np.mean(rel <= 5), w10=100 * np.mean(rel <= 10)))
    return pd.DataFrame(rows)


def tex_escape(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


def tex_row(name, m):
    name = tex_escape(name)
    if m.get("n_scored", 0) == 0:
        return f"{name} & \\TBD & \\TBD & \\TBD & \\TBD & \\TBD \\\\"
    ci = f"{m['MAPE']:.1f} [{m['MAPE_lo']:.1f}, {m['MAPE_hi']:.1f}]"
    wv = f"{m['weave_acc']:.0f}" if "weave_acc" in m else "--"
    cov = "" if m["coverage"] >= 99.9 else f"$^{{{m['n_scored']}/{m['n_gt']}}}$"
    return f"{name}{cov} & {m['MAE']:.1f} & {ci} & {m['w5']:.0f} & {m['w10']:.0f} & {wv} \\\\"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--blind-only", action="store_true")
    ap.add_argument("--annotator", default=None)
    ap.add_argument("--unit", choices=("scene", "image"), default="scene",
                    help="scene (default): one row per scene = its representative photo, labels propagated "
                         "from annotation_tool/propagate_scene_labels.py, plus main150_perfile.csv; "
                         "image: one row per labelled id (pre-Oct-8 behaviour)")
    ap.add_argument("--dedup", action="store_true",
                    help="--unit image only: one photo per group of copies (121 exposures, older grouping)")
    ap.add_argument("--out", default=os.path.join(PAPER, "tables"))
    ap.add_argument("--manifest", default=MANIFEST, help="annotation-tool manifest (scene fields, IAA subset, crop sizes)")
    ap.add_argument("--fig", action="store_true", help="write figs/fig_coverage.pdf")
    ap.add_argument("--fig-dir", default=os.path.join(PAPER, "figs"))
    ap.add_argument("--groups", default=None,
                    help="--unit scene only (Job P, Oct 8): CSV scene_id,fabric_group (e.g. tables/fabric_groups_candidate.csv); "
                         "also writes main150_fabric.csv with the bootstrap clustered by fabric group (sensitivity "
                         "analysis; default outputs unchanged)")
    ap.add_argument("--extra-wide", action="append", default=[],
                    help="PATH[::key1,key2] extra wide prediction CSV in the VLM format (e.g. gpu_results CountGD)")
    a = ap.parse_args()
    paths = [p for arg in a.labels for s in arg.split(",") for p in sorted(glob.glob(s.strip()))]
    if not paths:
        raise SystemExit("no label files found")
    os.makedirs(a.out, exist_ok=True)
    import json
    man = json.load(open(a.manifest))
    rec = load_labels(paths, a.blind_only, a.annotator)
    if man.get("iaa_subset"):   # current subset (v2 re-draw), not the flag stored when a record was created
        rec["iaa"] = rec["id"].astype(int).isin(set(man["iaa_subset"])).astype(int)
    gt = gt_table(rec)
    gt.index = gt.index.astype(int)
    perfile = None
    if a.unit == "scene":
        if a.dedup:
            print("--dedup is ignored with --unit scene (a scene is already one row)")
        PS = scene_module(a.manifest)
        scenes = PS.load_scenes(man)
        perfile, gt_scene = PS.propagate(gt, scenes)          # h/v swapped into each file's / the rep's frame
        n_own = len(gt)
        gt = gt_scene.drop(columns=["label_src_kind"], errors="ignore")
        gt.index = gt.index.astype(int)
        perfile = perfile[perfile.is_test_item.astype(bool)]  # notebook_synthetic: augmentations, never test items
        print(f"--unit scene: {n_own} labelled files -> {len(gt)} scenes (primary, rep photo) and "
              f"{len(perfile)} real files (secondary); scenes labelled from a non-rep file: "
              f"{sorted(int(i) for i in gt.index[gt.label_src_id != gt.index])}")
    elif a.dedup:
        drop = {i for g in json.load(open(P_DUP))["groups"] for i in sorted(g)[1:]}
        gt = gt[~gt.index.astype(int).isin(drop)]
        print(f"--dedup: dropped {len(drop)} copies listed in {os.path.relpath(P_DUP, ROOT)}")
    print(f"labels: {len(rec)} records from {len(paths)} file(s); {len(gt)} {a.unit}s; "
          f"annotators {sorted(rec.annotator.unique())}")

    side = {im["id"]: im["rect_size"][0] for im in man["images"]}
    gt["S"] = [side.get(int(i)) for i in gt.index]

    preds = load_predictions()
    preds.update(load_extra_wide(a.extra_wide))
    preds["Leave-one-out mean (reference)"] = loo_mean(gt)
    rows, tex = [], []
    for name, pr in preds.items():
        pr = pr.copy()
        pr.index = pr.index.astype(int)
        m = metrics(gt, pr)
        rows.append(dict(method=name, **m))
        tex.append(tex_row(name, m))
    pd.DataFrame(rows).round(2).to_csv(os.path.join(a.out, "main150.csv"), index=False)
    if a.groups and a.unit == "scene":   # Job P: fabric-level sensitivity (13 suspected re-shoot pairs merged)
        fg = pd.read_csv(a.groups).set_index("scene_id")["fabric_group"]
        fg.index = fg.index.astype(int)
        cl = pd.Series([fg.get(int(i), int(i)) for i in gt.index], index=gt.index)
        frows = []
        for name, pr in preds.items():
            pr = pr.copy()
            pr.index = pr.index.astype(int)
            frows.append(dict(method=name, **metrics(gt, pr, cluster=cl)))
        pd.DataFrame(frows).round(2).to_csv(os.path.join(a.out, "main150_fabric.csv"), index=False)
    if perfile is not None:   # SECONDARY: every real file of a labelled scene, scene-clustered CI
        pf = perfile[["h", "v", "scene_id", "kind", "rect_size"]].copy()
        pf.index = pf.index.astype(int)
        subsets = {"all real files": pf.index, "S<1000 (~490 px/in)": pf.index[pf.rect_size < 1000],
                   "S>=1000 (~1890 px/in)": pf.index[pf.rect_size >= 1000]}
        for kind in ("rep", "quarter_res_copy", "byte_identical", "png_copy"):
            subsets[f"kind {kind}"] = pf.index[pf.kind == kind]
        prow = []
        for name, pr in preds.items():
            if name.startswith("Leave-one-out"):
                continue
            pr = pr.copy()
            pr.index = pr.index.astype(int)
            for sname, idx in subsets.items():
                if len(idx) == 0:
                    continue
                g = pf.loc[idx]
                prow.append(dict(method=name, subset=sname, n_files=len(g), n_scenes=int(g.scene_id.nunique()),
                                 **metrics(g, pr, n_boot=500, cluster=g.scene_id)))
        pd.DataFrame(prow).round(2).to_csv(os.path.join(a.out, "main150_perfile.csv"), index=False)
    open(os.path.join(a.out, "main150_rows.tex"), "w").write(
        "% AUTO-GENERATED by tools/score150.py -- columns: method & MAE & MAPE [95% CI] & <=5% & <=10% & weave acc\n"
        + "\n".join(tex) + "\n")

    # stratification for the main methods
    strat = []
    groups = {"S<1000 (~490 px/in)": gt.S < 1000, "S>=1000 (~1890 px/in)": gt.S >= 1000}
    for f in FLAGS:
        groups[f"flag {f}"] = gt["flag_" + f] == 1
        groups[f"no flag {f}"] = gt["flag_" + f] == 0
    keep = [k for k in preds if k.startswith(("Ours", "FFT max", "GPT", "Gemini 3.8", "CNN"))]
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        for name in keep:
            pr = preds[name].copy()
            pr.index = pr.index.astype(int)
            strat.append(dict(group=gname, method=name, **metrics(gt[mask], pr, n_boot=500)))
    pd.DataFrame(strat).round(2).to_csv(os.path.join(a.out, "strat150.csv"), index=False)

    # selective prediction for ours: keep images with confidence >= t
    o = preds["Ours (lattice + fusion)"].copy()
    o.index = o.index.astype(int)
    cov = []
    for t in np.round(np.arange(0, 1.0, 0.05), 2):
        k = o.index[o.conf >= t]
        m = metrics(gt.loc[gt.index.intersection(k)], o, n_boot=200)
        m.pop("coverage", None)
        cov.append(dict(threshold=t, kept_pct=100 * len(gt.index.intersection(k)) / len(gt), **m))
    cov = pd.DataFrame(cov)
    cov.round(2).to_csv(os.path.join(a.out, "coverage_ours.csv"), index=False)

    # dataset statistics for Table 1 (tab:data)
    summ = dict(unit=a.unit, n_images=int(len(gt)), n_records=int(len(rec)), annotators=sorted(rec.annotator.unique().tolist()),
                n_files_labelled_propagated=None if perfile is None else int(len(perfile)),
                n_images_2plus_annotators=int((gt.n_ann >= 2).sum()),
                n_iaa_blind_records=int(((rec.iaa.astype(int) == 1) & (rec.proposals_seen.astype(int) == 0)).sum()),
                weave=gt.weave.fillna("unsure").value_counts().to_dict(),
                h_min_med_max=[float(gt.h.min()), float(gt.h.median()), float(gt.h.max())],
                v_min_med_max=[float(gt.v.min()), float(gt.v.median()), float(gt.v.max())],
                flags={f: int(gt["flag_" + f].sum()) for f in FLAGS},
                median_time_s=float(rec.time_spent_s.median()) if "time_spent_s" in rec else None)
    import json as _json
    with open(os.path.join(a.out, "data150_summary.json"), "w") as f:
        _json.dump(summ, f, indent=1)

    ha = human_agreement(rec)
    ha.round(2).to_csv(os.path.join(a.out, "human_agreement.csv"), index=False)

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        ax.plot(cov.kept_pct, cov.w10, "-o", ms=3, label="within 10%")
        ax.plot(cov.kept_pct, cov.w5, "-s", ms=3, label="within 5%")
        ax.set_xlabel("coverage (% of images kept, by confidence)")
        ax.set_ylabel("% of counts")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(a.fig_dir, exist_ok=True)
        fig.savefig(os.path.join(a.fig_dir, "fig_coverage.pdf"))
        fig.savefig(os.path.join(a.fig_dir, "fig_coverage.png"), dpi=200)
    print(pd.DataFrame(rows)[["method", "n_scored", "MAE", "MAPE", "w5", "w10", "both5", "bias_med"]].round(1)
          .to_string(index=False))
    if len(ha):
        print(ha.round(2).to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

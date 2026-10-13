"""Score the remote-GPU baselines (Molmo2-8B bf16 and CountGD, RTX 4090 on gpu-server) on the 13
hand-counted legacy windows, with the VLM harness' own code, and export the 150-crop predictions.

    <venv2>/bin/python gpu_results/score_gpu_baselines.py      # -> gpu_results/scored/, gpu_results/tables/

Inputs (all synced from gpu-server by rsync, md5-verified):
  raw/molmo2_8b_bf16__{legacy,rect150}[__stability].jsonl   Molmo2-8B bf16 (transformers), repeat 0 = 644 px,
                                                            stability repeats 1/2 = input pre-resized to 588/700 px
  raw/molmo2_8b_bf16_w4mlx__{legacy,rect150}.jsonl          same model with the Mac's MLX 4-bit LM weights dequantised
  ../vlm/raw/molmo2_8b__legacy.jsonl                         the Mac MLX 4-bit run (PARTIAL: P1 20/20, others 3-4 images)
  countgd/outputs/raw/countgd__legacy__shard0of1.jsonl       CountGD detections (score > 0.10 stored)
  countgd/outputs/results/{legacy_predictions,stageA_legacy_metrics,countgd_bootstrap_ci,
                           countgd_threshold_sensitivity}.csv  GPU-side summaries, re-verified here

Conventions, imported from wvaq2027/vlm (not re-implemented):
  * every Molmo2 answer is RE-PARSED from its raw text with vlm/parse_output.py (parse_counts for
    P1-P3, parse_points for P4/P4q); the 'parsed' field written on the GPU is only compared;
  * per-image estimates = vlm/evaluate_vlm.estimates (P3 = 4 x central quarter patch, P3s = sum of the
    4 diagonal quarter patches, P4 = number of points on the two whole-window guide lines,
    P4q = 4 x points on the central quarter patch);
  * metrics = vlm/evaluate_vlm.metrics / errors (axis-matched: vertical<->warp, horizontal<->weft;
    best assignment: sorted pair vs sorted pair); failures are not imputed (n_ok shown);
  * GT = vlm/gt_legacy.legacy_gt (2019 spreadsheet, the 2019 harness' own load_gt), 13 images.
Added here:
  * within-5 % counts;
  * the paper's CI convention (paper/sections/5_experiments.tex, paper/tools/score150.py): 2,000
    image-level bootstrap resamples, numpy default_rng(0); statistic = mean over images of the
    per-image mean % error (MAPE) or mean abs error (MAE) of its two counts;
  * paired bootstrap (same convention) of per-image differences between two methods.
CountGD: counts are recomputed from the raw detections (score > 0.23 and owned by the tile whose
centre is nearest, as pre-registered in run_countgd.py) and the whole-crop lattice TPI from the
detected cell centres, independently of the GPU-side evaluate_countgd.py, then compared with it.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # wvaq2027/gpu_results
ROOT = os.path.dirname(HERE)                                 # wvaq2027
VLM = os.path.join(ROOT, "vlm")
sys.path.insert(0, VLM)
import evaluate_vlm as EV                                    # noqa: E402  (the VLM harness' scorer)
from gt_legacy import legacy_gt                              # noqa: E402
from parse_output import parse_counts, parse_points          # noqa: E402

OUT = os.path.join(HERE, "scored")
TAB = os.path.join(HERE, "tables")
BF16, W4, MLX = "molmo2_8b_bf16", "molmo2_8b_bf16_w4mlx", "molmo2_8b"
LABEL = {BF16: "Molmo2-8B bf16 (HF transformers, RTX 4090)",
         W4: "Molmo2-8B bf16 runtime + MLX 4-bit LM weights (RTX 4090)",
         MLX: "Molmo2-8B 4-bit MLX (M2 Max; partial run)"}
METHODS = ["P1", "P2", "P3", "P3s", "P4", "P4q"]
LEGACY_NAMES = EV.LEGACY_NAMES
N_BOOT, SEED = 2000, 0
CG_THR = 0.23
CG_PRIMARY = "CG-B-yarn-ex3"


# ============================================================================ helpers
def write_csv(path, rows):
    """EV.write_csv layout, but 6 significant digits (EV uses 4, which double-rounds e.g. 53.846 -> 53.85 -> 53.9)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def boot_ci(x):
    """Paper convention: 2,000 image-level resamples, default_rng(0), percentile 95 % CI of the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return None, None
    rng = np.random.default_rng(SEED)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_image(rows, gt, mode):
    """{img: (mean abs err, mean % err)} over the parsed GT images, with EV.errors."""
    out = {}
    for img, g in gt.items():
        e = rows.get(img)
        if e is None or not e["ok"]:
            continue
        err, t = EV.errors(e["v"], e["h"], g, mode)
        ae = np.abs(np.array(err, float))
        out[img] = (float(ae.mean()), float((ae / np.array(t, float) * 100).mean()))
    return out


def score(rows, gt, mode):
    """EV.metrics + within-5 % + bootstrap CIs (paper convention)."""
    m = EV.metrics(rows, gt, mode)
    pe_all = []
    for img, g in gt.items():
        e = rows.get(img)
        if e is None or not e["ok"]:
            continue
        err, t = EV.errors(e["v"], e["h"], g, mode)
        pe_all += list(np.abs(np.array(err, float)) / np.array(t, float) * 100)
    m["within5"] = int((np.array(pe_all) <= 5).sum()) if pe_all else None
    pi = per_image(rows, gt, mode)
    m["mae_ci"] = boot_ci([v[0] for v in pi.values()])
    m["mape_ci"] = boot_ci([v[1] for v in pi.values()])
    return m


def paired(rows_a, rows_b, gt, mode, stat=0):
    """Paired bootstrap of per-image (a - b) abs error (stat 0) or % error (stat 1)."""
    pa, pb = per_image(rows_a, gt, mode), per_image(rows_b, gt, mode)
    imgs = [i for i in gt if i in pa and i in pb]
    if len(imgs) < 3:
        return dict(n_images=len(imgs))
    d = np.array([pa[i][stat] - pb[i][stat] for i in imgs])
    rng = np.random.default_rng(SEED)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return dict(n_images=len(imgs), diff=float(d.mean()), ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)), frac_boot_a_better=float(np.mean(bs < 0)),
                images=",".join(imgs))


def spearman_best(rows, gt):
    from scipy.stats import spearmanr
    p, t = [], []
    for img, g in gt.items():
        e = rows.get(img)
        if e is None or not e["ok"]:
            continue
        p += sorted([e["v"], e["h"]])
        t += sorted([g["warp"], g["weft"]])
    if len(set(p)) < 2:
        return None
    return float(spearmanr(p, t).correlation)


def f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ci_s(ci, nd=1):
    return "–" if ci is None or ci[0] is None else f"{ci[0]:.{nd}f}–{ci[1]:.{nd}f}"


# ============================================================================ Molmo2
def load_molmo():
    """All Molmo2 records, re-parsed from raw text; returns (records, reparse_check)."""
    srcs = [(BF16, sorted(glob.glob(os.path.join(HERE, "raw", f"{BF16}__*.jsonl")))),
            (W4, sorted(glob.glob(os.path.join(HERE, "raw", f"{W4}__*.jsonl")))),
            (MLX, sorted(glob.glob(os.path.join(VLM, "raw", f"{MLX}__*.jsonl"))))]
    recs, chk = [], Counter()
    for key, files in srcs:
        for fn in files:
            tag = os.path.basename(fn)[:-len(".jsonl")].split("__")[2:]
            if tag and tag[0] in ("smoke",) or (tag and tag[0].startswith("pilot")):
                continue
            for line in open(fn):
                r = json.loads(line)
                r["short"], r["cost_usd"] = key, 0.0
                r["repeat"] = int(r.get("repeat", 0)) if key != MLX else 0
                r["_file"] = os.path.relpath(fn, ROOT)
                st = r.get("parsed") or {}
                if r["prompt_id"].startswith("P4"):
                    new = parse_points(r.get("raw") or "")
                    same = (new["n_points"] == st.get("n_points") and bool(new["truncated"]) == bool(st.get("truncated")))
                else:
                    new = parse_counts(r.get("raw") or "")
                    same = all(new.get(k) == st.get(k) for k in ("vertical", "horizontal", "weave", "ok"))
                chk[(key, "agree" if same else "DIFFER")] += 1
                r["parsed"] = new
                recs.append(r)
    best = {}                                   # exactly evaluate_vlm.load_records' de-duplication
    for r in recs:
        k = (r["short"], r["set"], r["repeat"], str(r["image_id"]), r["view"], r["prompt_id"])
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return list(best.values()), {f"{a}|{b}": n for (a, b), n in sorted(chk.items())}


def rows_for(est, model, st, method, rep=0):
    return {img: e for (m, s, rp, me, img), e in est.items() if m == model and s == st and rp == rep and me == method}


def stage_a_row(label, model, method, rows, gt, crop, gt_plain):
    ax, be = score(rows, gt, "axis"), score(rows, gt, "best")
    bp = EV.metrics(rows, gt_plain, "best")
    gt_rows = [rows[i] for i in gt if i in rows]
    secs = [r["sec"] for r in gt_rows if r.get("sec") is not None and not math.isnan(r["sec"])]
    return dict(
        model=model, label=label, method=method, crop=crop, n_gt=len(gt), n_run=len(gt_rows),
        n_ok=ax.get("n_ok", 0), n_values=ax.get("n_values"),
        mae_axis=ax.get("mae"), mae_axis_ci_lo=(ax.get("mae_ci") or (None,))[0], mae_axis_ci_hi=(ax.get("mae_ci") or (None, None))[1],
        mape_axis=ax.get("mape"), mape_axis_ci_lo=(ax.get("mape_ci") or (None,))[0], mape_axis_ci_hi=(ax.get("mape_ci") or (None, None))[1],
        within5_axis=ax.get("within5"), within10_axis=ax.get("within10"), bias_axis=ax.get("bias"),
        medae_axis=ax.get("medae"), mae_warp=ax.get("mae_warp"), mae_weft=ax.get("mae_weft"),
        mae_best=be.get("mae"), mae_best_ci_lo=(be.get("mae_ci") or (None,))[0], mae_best_ci_hi=(be.get("mae_ci") or (None, None))[1],
        mape_best=be.get("mape"), mape_best_ci_lo=(be.get("mape_ci") or (None,))[0], mape_best_ci_hi=(be.get("mape_ci") or (None, None))[1],
        within5_best=be.get("within5"), within10_best=be.get("within10"), bias_best=be.get("bias"), medae_best=be.get("medae"),
        mae_best_plain=bp.get("mae"), n_ok_plain=bp.get("n_ok", 0), n_plain=len(gt_plain),
        spearman_best=spearman_best(rows, gt),
        n_distinct_answers_gt=len({(rows[i]["v"], rows[i]["h"]) for i in gt if i in rows and rows[i]["ok"]}),
        sec_per_img=float(np.mean(secs)) if secs else None)


def pointing_diag(recs):
    rows = []
    for r in recs:
        if not r["prompt_id"].startswith("P4") or r.get("error"):
            continue
        pts = r["parsed"].get("points", [])
        axis = 0 if r["view"].endswith("_h") else 1
        xs = np.sort(np.array([p[axis] for p in pts], float)) if pts else np.array([])
        gaps = np.diff(xs) if len(xs) > 2 else np.array([])
        cv = float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else None
        rows.append(dict(model=r["short"], set=r["set"], repeat=r["repeat"], image=str(r["image_id"]), view=r["view"],
                         n_points=len(pts), spacing_cv=cv,
                         median_gap_per1000=float(np.median(gaps)) if len(gaps) else None,
                         span_per1000=float(xs[-1] - xs[0]) if len(xs) > 1 else None,
                         mean_offset_from_line_per1000=float(np.mean([abs(p[1 - axis] - 500) for p in pts])) if pts else None,
                         truncated=r["parsed"].get("truncated"), finish_reason=r.get("finish_reason"),
                         generation_tokens=r.get("generation_tokens"), max_tokens=r.get("max_tokens"),
                         seconds=r.get("seconds")))
    return rows


def pointing_summary(pdg, model, st):
    P = [x for x in pdg if x["model"] == model and x["set"] == st and x["repeat"] == 0]
    if not P:
        return None
    cvs = np.array([x["spacing_cv"] for x in P if x["spacing_cv"] is not None])
    full = [x for x in P if not x["view"].startswith("qc")]
    return dict(model=model, set=st, n_lists=len(P), n_with_3plus_points=int(len(cvs)),
                frac_cv_lt_0p05=float(np.mean(cvs < 0.05)) if len(cvs) else None,
                frac_cv_lt_0p10=float(np.mean(cvs < 0.10)) if len(cvs) else None,
                median_spacing_cv=float(np.median(cvs)) if len(cvs) else None,
                median_points_whole_line=float(np.median([x["n_points"] for x in full])) if full else None,
                median_span_whole_line_per1000=float(np.median([x["span_per1000"] for x in full if x["span_per1000"] is not None])) if full else None,
                n_truncated=sum(1 for x in P if x["truncated"]),
                n_finish_length=sum(1 for x in P if x["finish_reason"] == "length"))


def stability(est, gt):
    out = []
    for st in ("legacy", "rect150"):
        for meth in ("P1", "P4"):
            reps = [rows_for(est, BF16, st, meth, rp) for rp in (0, 1, 2)]
            if not all(reps):
                continue
            imgs = [i for i in reps[1] if i in reps[0] and i in reps[2]]
            full = [i for i in imgs if all(reps[k][i]["ok"] for k in range(3))]
            sds, rng, relr, ident = [], [], [], 0
            for i in full:
                for c in ("v", "h"):
                    a = np.array([reps[k][i][c] for k in range(3)], float)
                    sds.append(a.std(ddof=1))
                    rng.append(np.ptp(a))
                    relr.append(np.ptp(a) / max(a.mean(), 1e-9) * 100)
                ident += int(all(reps[k][i]["v"] == reps[0][i]["v"] and reps[k][i]["h"] == reps[0][i]["h"] for k in range(3)))
            row = dict(model=BF16, set=st, method=meth, input_px="644 / 588 / 700", n_images=len(imgs),
                       n_images_all_parsed=len(full), mean_sd_tpi=float(np.mean(sds)) if sds else None,
                       mean_range_tpi=float(np.mean(rng)) if rng else None, max_range_tpi=float(np.max(rng)) if rng else None,
                       mean_range_pct=float(np.mean(relr)) if relr else None, identical_all3=ident)
            if st == "legacy":
                for k, side in zip(range(3), (644, 588, 700)):
                    rows = {i: reps[k][i] for i in imgs}
                    ax, be = score(rows, gt, "axis"), score(rows, gt, "best")
                    row.update({f"mae_axis_{side}": ax.get("mae"), f"mape_axis_{side}": ax.get("mape"),
                                f"within10_axis_{side}": ax.get("within10"), f"mae_best_{side}": be.get("mae"),
                                f"mape_best_{side}": be.get("mape"), f"within10_best_{side}": be.get("within10"),
                                f"n_ok_{side}": ax.get("n_ok")})
            out.append(row)
    return out


def method_of(view, pid):
    return {"crop": pid, "qc": "P3", "guide_h": "P4", "guide_v": "P4", "qc_guide_h": "P4q",
            "qc_guide_v": "P4q"}.get(view, "P3s" if view.startswith("qd") else None)


def quantisation(recs, est, gt):
    contrasts = [("weights_4bit", BF16, W4, "4-bit LM weights only (same HF runtime, preprocessing, prompt, inputs)"),
                 ("backend", W4, MLX, "MLX runtime (fp16 activations/kernels, mlx-vlm preprocessing) with the same 4-bit weights"),
                 ("total", BF16, MLX, "reference bf16 HF run vs the Mac MLX 4-bit run")]
    by = defaultdict(dict)
    for r in recs:
        if r["repeat"] == 0 and not r.get("error"):
            by[(r["set"], str(r["image_id"]), r["view"], r["prompt_id"])][r["short"]] = r
    out = []
    for name, A, B, what in contrasts:
        for st in ("legacy", "rect150"):
            for meth in METHODS:
                pr = [(k, d[A], d[B]) for k, d in by.items() if k[0] == st and method_of(k[2], k[3]) == meth and A in d and B in d]
                if not pr:
                    continue
                diffs = []
                for _, a, b in pr:
                    if meth in ("P4", "P4q"):
                        diffs.append(abs(a["parsed"]["n_points"] - b["parsed"]["n_points"]))
                    elif a["parsed"]["ok"] and b["parsed"]["ok"]:
                        diffs.append((abs(a["parsed"]["vertical"] - b["parsed"]["vertical"]) +
                                      abs(a["parsed"]["horizontal"] - b["parsed"]["horizontal"])) / 2)
                ra, rb = rows_for(est, A, st, meth), rows_for(est, B, st, meth)
                imgs = sorted({k[1] for k, _, _ in pr})
                both = [i for i in imgs if i in ra and i in rb and ra[i]["ok"] and rb[i]["ok"]]
                row = dict(contrast=name, isolates=what, model_a=A, model_b=B, set=st, method=meth,
                           n_calls_paired=len(pr), n_images=len(imgs),
                           identical_text=sum(a["raw"].strip() == b["raw"].strip() for _, a, b in pr),
                           mean_abs_diff_per_call=float(np.mean(diffs)) if diffs else None,
                           n_identical_estimates=sum(1 for i in both if ra[i]["v"] == rb[i]["v"] and ra[i]["h"] == rb[i]["h"]),
                           n_images_both_parsed=len(both),
                           n_distinct_answers_a=len({(ra[i]["v"], ra[i]["h"]) for i in both}),
                           n_distinct_answers_b=len({(rb[i]["v"], rb[i]["h"]) for i in both}))
                if st == "legacy":
                    g2 = {i: gt[i] for i in both if i in gt}
                    row["n_gt_images"] = len(g2)
                    if g2:
                        for tag, rows in (("a", ra), ("b", rb)):
                            for mode in ("axis", "best"):
                                m = EV.metrics({i: rows[i] for i in g2}, g2, mode)
                                row[f"mae_{mode}_{tag}"] = m.get("mae")
                                row[f"mape_{mode}_{tag}"] = m.get("mape")
                                row[f"within10_{mode}_{tag}"] = m.get("within10")
                        for mode in ("axis", "best"):
                            p = paired(ra, rb, g2, mode)
                            row[f"mae_diff_{mode}_a_minus_b"] = p.get("diff")
                            row[f"mae_diff_{mode}_ci_lo"] = p.get("ci_lo")
                            row[f"mae_diff_{mode}_ci_hi"] = p.get("ci_hi")
                out.append(row)
    return out


def rect150_exports(est):
    manifest = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    side = {str(im["id"]): im["rect_size"][0] for im in manifest["images"]}
    keys = [(BF16, "P1", BF16), (BF16, "P3", f"{BF16}@P3"), (BF16, "P3s", f"{BF16}@P3s"),
            (BF16, "P4", f"{BF16}@P4"), (W4, "P1", W4), (W4, "P4", f"{W4}@P4")]
    wide, longp, summ = defaultdict(dict), [], []
    for model, meth, key in keys:
        rows = rows_for(est, model, "rect150", meth)
        if not rows:
            continue
        for img in sorted(rows, key=int):
            e = rows[img]
            r0 = e["recs"][0]
            longp.append(dict(model=key, method=meth, image_id=int(img), native_crop_px=side.get(img),
                              vertical=e["v"], horizontal=e["h"], weave=e["weave"], parse_ok=e["ok"],
                              truncated=any(x["parsed"].get("truncated") for x in e["recs"]),
                              finish_reason=",".join(sorted({str(x.get("finish_reason")) for x in e["recs"]})),
                              seconds=e["sec"], model_repo=r0.get("repo"), revision=r0.get("revision"),
                              raw_file=r0.get("_file")))
            wide[int(img)][f"{key}__v"] = e["v"]
            wide[int(img)][f"{key}__h"] = e["h"]
            wide[int(img)][f"{key}__weave"] = e["weave"]
        ok = [e for e in rows.values() if e["ok"]]
        vals = np.array([[e["v"], e["h"]] for e in ok], float)
        c = Counter((e["v"], e["h"]) for e in ok)
        small = [np.mean([e["v"], e["h"]]) for i, e in rows.items() if e["ok"] and side.get(i, 0) < 1000]
        big = [np.mean([e["v"], e["h"]]) for i, e in rows.items() if e["ok"] and side.get(i, 0) >= 1000]
        secs = [e["sec"] for e in rows.values() if e["sec"] is not None and not math.isnan(e["sec"])]
        summ.append(dict(key=key, model=model, method=meth, n_run=len(rows), n_parsed=len(ok),
                         n_distinct_answers=len(c), most_common_answer=f"{c.most_common(1)[0][0][0]:g} x {c.most_common(1)[0][0][1]:g}" if c else None,
                         most_common_share=c.most_common(1)[0][1] / len(ok) if ok else None,
                         median_v=float(np.median(vals[:, 0])) if len(vals) else None,
                         median_h=float(np.median(vals[:, 1])) if len(vals) else None,
                         p05=float(np.percentile(vals, 5)) if len(vals) else None,
                         p95=float(np.percentile(vals, 95)) if len(vals) else None,
                         mean_tpi_small_crops=float(np.mean(small)) if small else None, n_small=len(small),
                         mean_tpi_large_crops=float(np.mean(big)) if big else None, n_large=len(big),
                         sec_per_img=float(np.mean(secs)) if secs else None))
    wrows = [dict(image_id=i, native_crop_px=side.get(str(i)), **wide[i]) for i in sorted(wide)]
    return longp, wrows, summ


# ============================================================================ CountGD
def cg_load():
    recs = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "countgd", "outputs", "raw", "countgd__legacy__shard*of*.jsonl"))):
        for line in open(fn):
            r = json.loads(line)
            k = (r["image_id"], r["method"], r["part"])
            if k not in recs or (recs[k].get("error") and not r.get("error")):
                recs[k] = r
    return recs


def cg_strip_count(r, thr=CG_THR, band=False):
    """Detections with score > thr owned by their tile; band: centre within +-p_perp/2 of the strip centre line."""
    S, n = r["side_px"], 0
    p_perp = r["p_y"] if r["part"] == "h" else r["p_x"]
    for x0, y0, x1, y1, s, owned in r["detections"]:
        if s > thr and owned:
            if band:
                cc = (y0 + y1) / 2 if r["part"] == "h" else (x0 + x1) / 2
                if abs(cc - S / 2) > p_perp / 2:
                    continue
            n += 1
    return n


def cg_lattice(r, thr=CG_THR):
    """Per-axis TPI from detected cell centres: S / median nearest-neighbour spacing within +-30 deg of
    the axis, among each centre's 12 nearest neighbours (rule stated in run_countgd.py)."""
    from scipy.spatial import cKDTree
    pts = np.array([((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in r["detections"] if d[4] > thr and d[5]], float)
    if len(pts) < 4:
        return None, None, len(pts)
    d, j = cKDTree(pts).query(pts, k=min(len(pts), 13))
    tan = math.tan(math.radians(30))
    dx, dy = [], []
    for i in range(len(pts)):
        bx = by = None
        for dd, jj in zip(d[i, 1:], j[i, 1:]):
            if not np.isfinite(dd) or dd <= 0:
                continue
            vx, vy = pts[jj] - pts[i]
            if bx is None and abs(vy) <= tan * abs(vx):
                bx = dd
            if by is None and abs(vx) <= tan * abs(vy):
                by = dd
        if bx is not None:
            dx.append(bx)
        if by is not None:
            dy.append(by)
    S = r["side_px"]
    return (S / np.median(dx) if dx else None), (S / np.median(dy) if dy else None), len(pts)


def cg_estimates(recs, thr=CG_THR):
    """At the pre-registered threshold the RUNTIME values are used (strip 'count' field and stored lattice
    TPI, computed from full-precision scores/boxes); the stored detections are rounded (score 4 dp, box
    0.1 px), so the recount from them is a check (chk) and is what other thresholds / the band rule use."""
    by = defaultdict(dict)
    for (img, m, part), r in recs.items():
        by[img][(m, part)] = r
    out, chk = {}, defaultdict(list)
    for img, d in by.items():
        r = d.get(("PER", "both"))
        if r:
            out[("PER", img)] = dict(v=r["count_v"], h=r["count_h"], ok=True, weave=None, sec=r["seconds"])
        for m in sorted({mm for (mm, _) in d if mm != "PER"}):
            if m.startswith("CG-W"):
                r = d[(m, "whole")]
                if r.get("error"):
                    out[(m + "-lat", img)] = dict(v=None, h=None, ok=False, weave=None, sec=r["seconds"])
                    continue
                le, lp, n = cg_lattice(r, thr)
                if thr == CG_THR:
                    if n != r["count"]:
                        chk["W cells: runtime N != recount"].append(f"{img}/{m}: {r['count']} vs {n}")
                    for a, b in ((le, r.get("lat_ends")), (lp, r.get("lat_picks"))):
                        if a is not None and b is not None:
                            chk["W lattice |recount - runtime| TPI"].append(abs(a - b))
                    le, lp = r.get("lat_ends"), r.get("lat_picks")
                out[(m + "-lat", img)] = dict(v=le, h=lp, ok=le is not None and lp is not None, weave=None, sec=r["seconds"])
                continue
            rh, rv = d.get((m, "h")), d.get((m, "v"))
            if rh is None or rv is None:
                continue
            per_s = d[("PER", "both")]["seconds"] if ("PER", "both") in d else 0.0
            sec = rh["seconds"] + rv["seconds"] - per_s
            if rh.get("error") or rv.get("error"):
                for mm in (m, m + "@band"):
                    out[(mm, img)] = dict(v=None, h=None, ok=False, weave=None, sec=sec)
                continue
            if thr == CG_THR:
                for rr in (rh, rv):
                    if cg_strip_count(rr) != rr["count"]:
                        chk["strip: runtime count != recount"].append(f"{img}/{m}/{rr['part']}: {rr['count']} vs {cg_strip_count(rr)}")
                    chk["strip records checked"].append(1)
                v, h = float(rh["count"]), float(rv["count"])
            else:
                v, h = float(cg_strip_count(rh, thr)), float(cg_strip_count(rv, thr))
            out[(m, img)] = dict(v=v, h=h, ok=True, weave=None, sec=sec)
            out[(m + "@band", img)] = dict(v=float(cg_strip_count(rh, thr, True)), h=float(cg_strip_count(rv, thr, True)),
                                           ok=True, weave=None, sec=sec)
    s = {}
    for k, v in chk.items():
        if k.startswith("W lattice"):
            s[k] = dict(n=len(v), median=float(np.median(v)), max=float(np.max(v)))
        elif k == "strip records checked":
            s[k] = len(v)
        else:
            s[k] = v
    return out, s


def read_csv(path):
    return list(csv.DictReader(open(path)))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def countgd(gt, gt_plain):
    recs = cg_load()
    est, chk = cg_estimates(recs)
    meths = sorted({m for (m, _) in est})
    res = dict(n_records=len(recs), n_error_records=sum(1 for r in recs.values() if r.get("error")),
               error_records=sorted({f"{r['method']}/{r['part']}" for r in recs.values() if r.get("error")}),
               recount_checks=chk)
    # inputs = canonical VLM crops: side_px must equal the canonical crop size
    import cv2
    sides = {}
    for (img, m, part), r in recs.items():
        sides.setdefault(img, set()).add(r["side_px"])
    bad = []
    for img, s in sides.items():
        c = cv2.imread(os.path.join(VLM, "crops", "legacy", f"{img}.png"))
        if s != {c.shape[0]} or c.shape[0] != c.shape[1]:
            bad.append(img)
    res["side_px_equals_canonical_crop"] = f"{len(sides) - len(bad)}/{len(sides)}"
    # compare with the GPU-side per-image CSV
    theirs = read_csv(os.path.join(HERE, "countgd", "outputs", "results", "legacy_predictions.csv"))
    n_cmp, ok_mis, gt_mis, beyond = 0, 0, 0, []
    for t in theirs:
        e = est.get((t["method"], t["image"]))
        if e is None:
            ok_mis += 1
            continue
        n_cmp += 1
        ok_mis += int((t["ok"] == "True") != bool(e["ok"]))
        if e["ok"]:
            for a, b in ((fnum(t["vertical"]), e["v"]), (fnum(t["horizontal"]), e["h"])):
                if abs(a - b) > 5e-4 * abs(b) + 1e-9:          # beyond 4-significant-digit rounding
                    beyond.append(f"{t['image']}/{t['method']}: gpu {a:g} vs here {b:.4g}")
        if t["image"] in gt:
            gt_mis += int(fnum(t["gt_warp"]) != gt[t["image"]]["warp"] or fnum(t["gt_weft"]) != gt[t["image"]]["weft"])
    res["vs_gpu_legacy_predictions"] = dict(n_rows=len(theirs), n_compared=n_cmp, ok_flag_or_missing_mismatch=ok_mis,
                                            gt_mismatch=gt_mis, differ_beyond_rounding=beyond,
                                            note="GPU CSV: 4 significant digits; its strip counts are recounted from the "
                                                 "rounded stored scores, ours use the runtime count field")
    # metrics: recomputed vs GPU-side stageA
    rows_out = []
    for m in meths:
        rows = {img: e for (mm, img), e in est.items() if mm == m}
        rows_out.append(stage_a_row(f"CountGD {m}" if m != "PER" else "Periodicity only (CountGD exemplar generator)",
                                    "countgd", m, rows, gt, "VLM trimmed window, native px", gt_plain))
    theirs_a = {r["method"]: r for r in read_csv(os.path.join(HERE, "countgd", "outputs", "results", "stageA_legacy_metrics.csv"))}
    worst = {}
    for r in rows_out:
        t = theirs_a.get(r["method"])
        if t is None:
            worst.setdefault("missing_in_gpu_csv", []).append(r["method"])
            continue
        for k in ("mae_axis", "mape_axis", "within10_axis", "mae_best", "mape_best", "within10_best", "n_ok"):
            a, b = r[k], fnum(t[k])
            if a is None and b is None:
                continue
            rel = abs(a - b) / max(abs(b), 1e-9) if (a is not None and b is not None) else float("inf")
            worst[k] = max(worst.get(k, 0.0), rel)
    res["vs_gpu_stageA_max_rel_diff"] = worst
    # bootstrap: reproduce the GPU-side CI (B=10000, idx matrix, MAE best) as a check
    keys = sorted(gt)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), size=(10000, len(keys)))
    rep = {}
    for m in (CG_PRIMARY, CG_PRIMARY + "@band", "PER", "CG-W-ex3-lat", "CG-W-tx-lat"):
        pi = per_image({i: e for (mm, i), e in est.items() if mm == m}, gt, "best")
        arr = np.array([pi[k][0] if k in pi else np.nan for k in keys])
        bs = np.nanmean(arr[idx], axis=1)
        rep[m] = (float(np.nanmean(arr)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
    gpu_ci = {r["method"]: (fnum(r["mae_best"]), fnum(r["ci95_lo"]), fnum(r["ci95_hi"]))
              for r in read_csv(os.path.join(HERE, "countgd", "outputs", "results", "countgd_bootstrap_ci.csv"))}
    res["vs_gpu_bootstrap_ci_max_abs_diff"] = max(abs(a - b) for m in rep for a, b in zip(rep[m], gpu_ci[m]))
    # threshold sensitivity, recomputed from raw detections
    sens, theirs_s = [], read_csv(os.path.join(HERE, "countgd", "outputs", "results", "countgd_threshold_sensitivity.csv"))
    maxd = 0.0
    for thr in (0.15, 0.20, 0.23, 0.30, 0.40, 0.50):
        e_t, _ = cg_estimates(recs, thr)
        for m in (CG_PRIMARY, CG_PRIMARY + "@band"):
            rows = {i: e for (mm, i), e in e_t.items() if mm == m}
            ax, be = EV.metrics(rows, gt, "axis"), EV.metrics(rows, gt, "best")
            sens.append(dict(method=m, threshold=thr, mae_axis=ax.get("mae"), mape_axis=ax.get("mape"),
                             within10_axis=ax.get("within10"), mae_best=be.get("mae"), bias_best=be.get("bias")))
            t = [x for x in theirs_s if x["method"] == m and abs(float(x["threshold"]) - thr) < 1e-9]
            if t:
                maxd = max(maxd, abs(fnum(t[0]["mae_best"]) - be["mae"]) / be["mae"])
    res["vs_gpu_threshold_sensitivity_max_rel_diff"] = maxd
    # paired: does CountGD add anything over its own exemplar generator?
    per = {img: e for (mm, img), e in est.items() if mm == "PER"}
    res["paired_vs_PER"] = {}
    for m in (CG_PRIMARY, CG_PRIMARY + "@band", "CG-W-ex3-lat"):
        rows = {img: e for (mm, img), e in est.items() if mm == m}
        res["paired_vs_PER"][m] = {mode: paired(rows, per, gt, mode, stat=1) for mode in ("axis", "best")}
    # whole-crop cells vs GT ends x picks (plain weave should give N = ends x picks)
    cells = []
    for img, g in gt.items():
        r = recs.get((img, "CG-W-ex3", "whole"))
        if r and not r.get("error"):
            cells.append(dict(image=img, weave=g["weave_name"], n_cells=r["count"], gt_ends_x_picks=g["warp"] * g["weft"],
                              ratio=r["count"] / (g["warp"] * g["weft"])))
    res["whole_cells_ratio_median_plain"] = float(np.median([c["ratio"] for c in cells if c["weave"] == "plain"]))
    res["whole_cells_ratio_median_twill"] = float(np.median([c["ratio"] for c in cells if c["weave"] == "twill"]))
    # primary over-count factor vs periodicity
    prim = [(est[(CG_PRIMARY, i)], est[("PER", i)]) for i in LEGACY_NAMES if (CG_PRIMARY, i) in est and est[(CG_PRIMARY, i)]["ok"]]
    res["primary_over_PER_median"] = float(np.median([a[c] / b[c] for a, b in prim for c in ("v", "h")]))
    return est, rows_out, sens, cells, res


# ============================================================================ references (same convention)
def reference_rows(gt, gt_plain):
    out = []
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_per_image.csv")).set_index("image")
    rows = {i: dict(v=float(d.loc[i, "ours_h"]), h=float(d.loc[i, "ours_v"]), ok=True, weave=None, sec=float("nan"))
            for i in gt if i in d.index}
    out.append(stage_a_row("Ours (lattice + fusion), frozen", "ours", "P1 whole crop", rows, gt, "methods P1 whole crop", gt_plain))
    v = [r for r in read_csv(os.path.join(VLM, "results", "legacy_predictions.csv"))
         if r["model"] == "gpt-6.1-sol" and r["method"] == "P1" and r["set"] == "legacy" and r["repeat"] == "0"]
    rows = {r["image"]: dict(v=fnum(r["vertical"]), h=fnum(r["horizontal"]), ok=r["ok"] == "True", weave=r["weave"],
                             sec=fnum(r["seconds"])) for r in v}
    out.append(stage_a_row("GPT-6.1-sol (API)", "gpt-6.1-sol", "P1", rows, gt, "VLM trimmed window, 1536 px", gt_plain))
    keys = list(gt)
    loo_ax = {k: dict(v=np.mean([gt[o]["warp"] for o in keys if o != k]), h=np.mean([gt[o]["weft"] for o in keys if o != k]),
                      ok=True, weave=None, sec=float("nan")) for k in keys}
    r_ax = stage_a_row("Leave-one-out mean predictor", "loo_mean", "-", loo_ax, gt, "-", gt_plain)
    loo_be = {k: dict(v=np.mean([min(gt[o]["warp"], gt[o]["weft"]) for o in keys if o != k]),
                      h=np.mean([max(gt[o]["warp"], gt[o]["weft"]) for o in keys if o != k]), ok=True, weave=None, sec=float("nan"))
              for k in keys}
    r_be = stage_a_row("Leave-one-out mean predictor", "loo_mean", "-", loo_be, gt, "-", gt_plain)
    for k in r_be:                               # best-assignment columns from the sorted-mean predictor (as EV does)
        if "best" in k:
            r_ax[k] = r_be[k]
    out.append(r_ax)
    return out


# ============================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    gt = legacy_gt()
    gt_plain = {k: g for k, g in gt.items() if g["weave_name"] == "plain"}
    S = dict(gt_images=sorted(gt), n_gt=len(gt), bootstrap=f"{N_BOOT} image-level resamples, default_rng({SEED})")

    # ---------------- Molmo2
    recs, reparse = load_molmo()
    S["molmo2_reparse_vs_stored"] = reparse
    S["molmo2_n_records"] = dict(Counter(r["short"] for r in recs))
    est = EV.estimates(recs)
    stage = []
    for m in (BF16, W4, MLX):
        for meth in METHODS:
            rows = rows_for(est, m, "legacy", meth)
            if rows:
                crop = "VLM trimmed window, 644 px" + (" (Mac MLX)" if m == MLX else " (RTX 4090)")
                stage.append(stage_a_row(LABEL[m], m, meth, rows, gt, crop, gt_plain))
    refs = reference_rows(gt, gt_plain)
    cg_est, cg_rows, cg_sens, cg_cells, cg_res = countgd(gt, gt_plain)
    allrows = refs + stage + cg_rows
    write_csv(os.path.join(OUT, "legacy13_stageA_gpu_baselines.csv"), allrows)
    S["stageA"] = allrows

    # P4 vs P1 (and the other prompts vs P1), paired, bf16
    p4 = []
    r1 = rows_for(est, BF16, "legacy", "P1")
    for meth in ("P2", "P3", "P3s", "P4", "P4q"):
        rows = rows_for(est, BF16, "legacy", meth)
        for mode in ("axis", "best"):
            for stat, nm in ((0, "MAE"), (1, "MAPE")):
                p = paired(rows, r1, gt, mode, stat)
                p4.append(dict(model=BF16, a=meth, b="P1", mode=mode, stat=nm, **p))
    write_csv(os.path.join(OUT, "molmo2_bf16_paired_vs_P1.csv"), p4)
    S["paired_vs_P1"] = p4

    # per-image legacy predictions (GPU models; all 20 images, GT where it exists)
    per = []
    for (m, st, rep, meth, img), e in sorted(est.items()):
        if st != "legacy":
            continue
        g = gt.get(img)
        per.append(dict(model=m, repeat=rep, input_px={0: 644, 1: 588, 2: 700}[rep], method=meth, image=img,
                        vertical=e["v"], horizontal=e["h"], weave=e["weave"], ok=e["ok"],
                        gt_warp=g["warp"] if g else None, gt_weft=g["weft"] if g else None,
                        gt_weave=g["weave_name"] if g else None, seconds=e["sec"]))
    for (m, img), e in sorted(cg_est.items()):
        g = gt.get(img)
        per.append(dict(model="countgd", repeat=0, input_px="native", method=m, image=img, vertical=e["v"],
                        horizontal=e["h"], weave=None, ok=e["ok"], gt_warp=g["warp"] if g else None,
                        gt_weft=g["weft"] if g else None, gt_weave=g["weave_name"] if g else None, seconds=e["sec"]))
    write_csv(os.path.join(OUT, "legacy_predictions_gpu.csv"), per)

    # stability, quantisation, pointing
    stab = stability(est, gt)
    write_csv(os.path.join(OUT, "molmo2_bf16_stability_resize.csv"), stab)
    S["stability_resize"] = stab
    q = quantisation(recs, est, gt)
    write_csv(os.path.join(OUT, "molmo2_quantisation.csv"), q)
    S["quantisation"] = q
    dq = json.load(open(os.path.join(HERE, "logs", "w4mlx_dequant_check.json")))
    rels = [x["rel_fro_err"] for x in dq["layers"]]
    S["w4mlx_dequant"] = dict(n_patched=dq["n_patched"], rel_fro_err_median=float(np.median(rels)), rel_fro_err_max=float(np.max(rels)))
    pdg = pointing_diag(recs)
    write_csv(os.path.join(OUT, "molmo2_pointing_diagnostics.csv"), pdg)
    S["pointing"] = [x for x in (pointing_summary(pdg, m, s) for m in (BF16, W4, MLX) for s in ("legacy", "rect150")) if x]
    write_csv(os.path.join(OUT, "molmo2_pointing_summary.csv"), S["pointing"])
    hist = {}
    for m in (BF16, W4, MLX):
        for st in ("legacy", "rect150"):
            for meth in ("P1", "P2", "P3"):
                r = rows_for(est, m, st, meth)
                if r:
                    c = Counter((e["v"], e["h"]) for e in r.values() if e["ok"])
                    hist[f"{m}|{st}|{meth}"] = dict(n=sum(c.values()), answers={f"{k[0]:g}x{k[1]:g}": v for k, v in c.most_common(6)})
    S["answer_histograms"] = hist

    # rect150 exports (no GT yet)
    longp, wrows, summ = rect150_exports(est)
    write_csv(os.path.join(OUT, "molmo2_rect150_predictions_long.csv"), longp)
    write_csv(os.path.join(OUT, "molmo2_rect150_predictions_wide.csv"), wrows)
    write_csv(os.path.join(OUT, "molmo2_rect150_summary.csv"), summ)
    S["rect150"] = summ

    # CountGD
    write_csv(os.path.join(OUT, "countgd_threshold_sensitivity_recomputed.csv"), cg_sens)
    write_csv(os.path.join(OUT, "countgd_whole_cells_vs_gt.csv"), cg_cells)
    S["countgd"] = cg_res

    # paired comparisons against the reference rows (per-image % error, paired bootstrap, same convention)
    def rows_of(model, method):
        if model == "countgd":
            return {i: e for (m, i), e in cg_est.items() if m == method}
        return rows_for(est, model, "legacy", method)
    keys = list(gt)
    loo = {k: dict(v=np.mean([gt[o]["warp"] for o in keys if o != k]), h=np.mean([gt[o]["weft"] for o in keys if o != k]),
                   ok=True) for k in keys}
    import pandas as pd
    dd = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_per_image.csv")).set_index("image")
    ours = {i: dict(v=float(dd.loc[i, "ours_h"]), h=float(dd.loc[i, "ours_v"]), ok=True) for i in gt}
    pr = []
    for model, meth in [(BF16, "P1"), (BF16, "P3"), (BF16, "P4"), (BF16, "P4q"), ("countgd", CG_PRIMARY),
                        ("countgd", "CG-W-ex3-lat"), ("countgd", "PER")]:
        for ref_name, ref in (("loo_mean", loo), ("ours", ours)):
            p = paired(rows_of(model, meth), ref, gt, "axis", stat=1)
            pr.append(dict(model=model, method=meth, ref=ref_name, mode="axis", stat="MAPE", **p))
    write_csv(os.path.join(OUT, "paired_vs_references.csv"), pr)
    S["paired_vs_references"] = pr

    # paper-schema rows (paper/tables/legacy13_all.csv columns + CI columns)
    sel = [(BF16, m) for m in METHODS] + [(W4, m) for m in ("P1", "P4")] + \
          [("countgd", m) for m in (CG_PRIMARY, CG_PRIMARY + "@band", "CG-W-ex3-lat", "PER")]
    prow = []
    for r in allrows:
        if (r["model"], r["method"]) not in sel:
            continue
        group = "vlm-gpu" if r["model"].startswith("molmo2") else "specialist"
        name = f"{r['model']} {r['method']}"
        for assign, sfx in (("axis", "axis"), ("sorted", "best")):
            prow.append(dict(group=group, method=name, crop=r["crop"], assign=assign, axis="both", n=r["n_values"],
                             MAE=r[f"mae_{sfx}"], MAPE=r[f"mape_{sfx}"], n_w5=r[f"within5_{sfx}"], n_w10=r[f"within10_{sfx}"],
                             medAE=r[f"medae_{sfx}"], n_images=r["n_ok"], MAPE_ci_lo=r[f"mape_{sfx}_ci_lo"],
                             MAPE_ci_hi=r[f"mape_{sfx}_ci_hi"], MAE_ci_lo=r[f"mae_{sfx}_ci_lo"], MAE_ci_hi=r[f"mae_{sfx}_ci_hi"]))
    write_csv(os.path.join(TAB, "legacy13_gpu_rows.csv"), prow)

    # LaTeX rows in the format of the paper's legacy table (tab:legacy: axis-matched, MAE, MAPE, n<=5%, n<=10% of 26)
    R = {(r["model"], r["method"]): r for r in allrows}

    def cell(*ks, key):
        return " / ".join(f(R[k][key]) if isinstance(R[k][key], float) else str(R[k][key]) for k in ks)
    tex = ["% generated by gpu_results/score_gpu_baselines.py -- axis-matched, legacy-13 (26 counts), same crops as the VLM rows",
           "% Molmo2-8B bf16 (RTX 4090) replaces the partial 4-bit MLX row; CountGD on the same VLM crops at native resolution"]
    for name, ks in (("Molmo2-8B (bf16), P1 / P4q", [(BF16, "P1"), (BF16, "P4q")]),
                     ("CountGD, yarn strips (pre-registered)", [("countgd", CG_PRIMARY)]),
                     ("CountGD, cells $\\to$ lattice (post hoc)$^c$", [("countgd", "CG-W-ex3-lat")])):
        tex.append(f"{name} & VLM & {cell(*ks, key='mae_axis')} & {cell(*ks, key='mape_axis')} & "
                   f"{cell(*ks, key='within5_axis')} & {cell(*ks, key='within10_axis')} \\\\")
    open(os.path.join(TAB, "tab_legacy_gpu_rows.tex"), "w").write("\n".join(tex) + "\n")
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    report(S)


def report(S):
    print(f"\nre-parse check: {S['molmo2_reparse_vs_stored']}\nrecords: {S['molmo2_n_records']}")
    print("\n| Model | Method | ok | MAE ax [CI] | MAPE ax [CI] | ≤5 ax | ≤10 ax | MAE best | MAPE best [CI] | ≤5 b | ≤10 b | bias | distinct | s/img |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in S["stageA"]:
        if r["model"] == "countgd" and r["method"] not in (CG_PRIMARY, CG_PRIMARY + "@band", "CG-W-ex3-lat", "CG-W-tx-lat", "PER"):
            continue
        print(f"| {r['label']} | {r['method']} | {r['n_ok']}/{r['n_run']} | {f(r['mae_axis'])} [{ci_s((r['mae_axis_ci_lo'], r['mae_axis_ci_hi']))}] | "
              f"{f(r['mape_axis'])} [{ci_s((r['mape_axis_ci_lo'], r['mape_axis_ci_hi']))}] | {r['within5_axis']} | {r['within10_axis']} | "
              f"{f(r['mae_best'])} | {f(r['mape_best'])} [{ci_s((r['mape_best_ci_lo'], r['mape_best_ci_hi']))}] | {r['within5_best']} | "
              f"{r['within10_best']} | {f(r['bias_best'])} | {r['n_distinct_answers_gt']} | {f(r['sec_per_img'])} |")
    print("\npaired vs P1 (bf16):")
    for p in S["paired_vs_P1"]:
        if p.get("diff") is not None:
            print(f"  {p['a']}-P1 {p['mode']:4s} {p['stat']:4s} n={p['n_images']} diff={p['diff']:.1f} [{p['ci_lo']:.1f}, {p['ci_hi']:.1f}]")
    print("\nstability:")
    for r in S["stability_resize"]:
        print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nquantisation:")
    for r in S["quantisation"]:
        print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "isolates"})
    print("\nw4 dequant:", S["w4mlx_dequant"])
    print("\npointing:")
    for r in S["pointing"]:
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nanswer histograms:")
    for k, v in S["answer_histograms"].items():
        print(" ", k, v)
    print("\nrect150:")
    for r in S["rect150"]:
        print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("\ncountgd checks:")
    for k, v in S["countgd"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()

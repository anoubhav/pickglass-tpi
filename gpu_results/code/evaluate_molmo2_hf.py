"""Score the bf16 (GPU, transformers) Molmo2-8B run and compare it with the Mac MLX 4-bit run.

All scoring code is the Mac harness' own (code/mac_vlm/evaluate_vlm.py: estimates, metrics,
errors, loo_mean_baseline, weave_acc, write_csv), imported verbatim, so the numbers are
directly comparable with <repo root>/vlm/results/*.csv.  GT = legacy spreadsheet via the legacy
harness' load_gt (13 usable images).

Inputs
  raw/molmo2_8b_bf16__*.jsonl                       this run (model key 'molmo2_8b_bf16')
  mac_mlx_snapshot/molmo2_8b__*.jsonl               read-only copy of the Mac MLX-4bit records
Outputs (results/)
  stageA_legacy_metrics.csv     same columns (and order) as the Mac file + extras at the end; GPU
                                models only (molmo2_8b_bf16, molmo2_8b_bf16_w4mlx) -> append-safe
  stageA_with_mlx_snapshot.csv  the same table incl. molmo2_8b (Mac MLX 4-bit, whatever exists)
  legacy_predictions.csv        same columns as the Mac file (GPU models only; *_with_mlx_snapshot.csv: all)
  stability_P1_legacy.csv       Mac layout; the 3 'repeats' are the 3 input resizes (greedy decoding)
  rect150_P1_summary.csv        Mac layout (P1 rows)
  rect150_P1_predictions_long.csv / _wide.csv   same columns as the Mac files (P1 only)
  rect150_all_methods_long.csv  every method (P1, P3, P3s, P4) on the 150 crops
  stability_resize.csv          3 input resizes (644 / 588 / 700 px) x 20 legacy + 20 rect150 crops
  molmo2_pointing_diagnostics.csv  same columns as the Mac file (+ model, set, repeat)
  quant_paired_records.csv      every (image, view, prompt) answered by both backends
  quant_summary.csv             agreement + paired metrics, bf16 vs MLX 4-bit
  bootstrap_ci_legacy.csv       image-level bootstrap 95% CI of MAE / %err (methods with all 13 GT parsed)
  bootstrap_paired_diff_legacy.csv  paired bootstrap of the MAE difference bf16 - w4mlx / bf16 - MLX
  summary.json
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as C                                           # noqa: E402
import evaluate_vlm as EV                                        # noqa: E402  (verbatim Mac scorer)
from vlm_common import parse_points, parse_counts                # noqa: E402

ROOT = C.ROOT
RAW = os.path.join(ROOT, "raw")
MLX_DIR = os.path.join(ROOT, "mac_mlx_snapshot")
OUT = os.path.join(ROOT, "results")
BF16, W4, MLX = "molmo2_8b_bf16", "molmo2_8b_bf16_w4mlx", "molmo2_8b"
LABEL = {BF16: "Molmo2-8B bf16 (transformers, RTX 4090)",
         W4: "Molmo2-8B bf16 + MLX 4-bit LM weights (transformers, RTX 4090)",
         MLX: "Molmo2-8B 4-bit (MLX, M2 Max)"}
METHODS = ["P1", "P2", "P3", "P3s", "P4", "P4q"]


def load_records():
    recs = []
    for key in (BF16, W4):
        for f in sorted(glob.glob(os.path.join(RAW, f"{key}__*.jsonl"))):
            tag = os.path.basename(f)[:-len(".jsonl")].split("__")[2:]
            if tag and tag[0] == "smoke":
                continue
            for line in open(f):
                r = json.loads(line)
                r["short"], r["cost_usd"] = key, 0.0
                r["repeat"] = int(r.get("repeat", 0))
                recs.append(r)
    for f in sorted(glob.glob(os.path.join(MLX_DIR, f"{MLX}__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            r["short"], r["repeat"], r["cost_usd"] = MLX, 0, 0.0
            if r["prompt_id"].startswith("P4") and not r.get("error"):
                r["parsed"] = parse_points(r["raw"])      # as evaluate_vlm does
            recs.append(r)
    best = {}
    for r in recs:
        k = (r["short"], r["set"], r["repeat"], str(r["image_id"]), r["view"], r["prompt_id"])
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return list(best.values())


def rows_for(est, model, st, method, rep=0):
    return {img: e for (m, s, rp, me, img), e in est.items() if m == model and s == st and rp == rep and me == method}


def corr(rows, gt, mode):
    """Spearman / Pearson between predicted and GT values (2 per parsed GT image)."""
    from scipy.stats import pearsonr, spearmanr
    p, t = [], []
    for img, g in gt.items():
        e = rows.get(img)
        if e is None or not e["ok"]:
            continue
        if mode == "axis":
            p += [e["v"], e["h"]]; t += [g["warp"], g["weft"]]
        else:
            p += sorted([e["v"], e["h"]]); t += sorted([g["warp"], g["weft"]])
    if len(p) < 4 or len(set(p)) < 2:
        return None, None
    return float(spearmanr(p, t).correlation), float(pearsonr(p, t)[0])


def stageA(est, gt, models=(BF16, W4, MLX)):
    """Columns: exactly the Mac stageA_legacy_metrics.csv columns (same order), then extras."""
    out = []
    for m in models:
        for meth in METHODS:
            rows = rows_for(est, m, "legacy", meth)
            if not rows:
                continue
            ax, be = EV.metrics(rows, gt, "axis"), EV.metrics(rows, gt, "best")
            gt_plain = {k: g for k, g in gt.items() if g["weave_name"] == "plain"}
            bp, ap = EV.metrics(rows, gt_plain, "best"), EV.metrics(rows, gt_plain, "axis")
            gt_rows = [rows[i] for i in gt if i in rows]
            all20 = [rows[i] for i in C.LEGACY_NAMES if i in rows]
            secs = [r["sec"] for r in gt_rows if not math.isnan(r["sec"])]
            wc, wn = EV.weave_acc(rows, gt)
            out.append(dict(
                model=m, label=LABEL[m], method=meth, n_gt=len(gt),
                n_ok=ax.get("n_ok", 0), parse_fail_gt=ax.get("n_fail"), not_run_gt=ax.get("n_missing", 0),
                parse_fail_all20=sum(1 for r in all20 if not r["ok"]), n_run_all20=len(all20),
                mae_axis=ax.get("mae"), mape_axis=ax.get("mape"), within10_axis=ax.get("within10"),
                bias_axis=ax.get("bias"), mae_warp=ax.get("mae_warp"), mae_weft=ax.get("mae_weft"),
                mae_best=be.get("mae"), mape_best=be.get("mape"), within10_best=be.get("within10"),
                bias_best=be.get("bias"), medae_best=be.get("medae"), n_values=be.get("n_values"),
                weave_correct=wc, weave_n=wn,
                n_plain=len(gt_plain), n_ok_plain=bp.get("n_ok", 0), mae_best_plain=bp.get("mae"),
                mape_best_plain=bp.get("mape"), within10_best_plain=bp.get("within10"),
                mae_axis_plain=ap.get("mae"), within10_axis_plain=ap.get("within10"),
                sec_per_img=float(np.mean(secs)) if secs else None, usd_per_img=0.0,
                n_gt_run=len(gt_rows),
                spearman_best=corr(rows, gt, "best")[0], pearson_best=corr(rows, gt, "best")[1],
                spearman_axis=corr(rows, gt, "axis")[0],
                n_distinct_answers_gt=len({(rows[i]["v"], rows[i]["h"]) for i in gt if i in rows and rows[i]["ok"]})))
    return out


def legacy_predictions(est, gt, models=(BF16, W4, MLX)):
    rows = []
    for (m, st, rep, meth, img), e in sorted(est.items()):
        if st != "legacy" or m not in models:
            continue
        g = gt.get(img)
        rows.append(dict(model=m, set=st, repeat=rep, method=meth, image=img,
                         vertical=e["v"], horizontal=e["h"], weave=e["weave"], ok=e["ok"],
                         gt_warp=g["warp"] if g else None, gt_weft=g["weft"] if g else None,
                         gt_weave=g["weave_name"] if g else None, seconds=e["sec"], usd=e["cost"],
                         input_side=e["recs"][0].get("resize_side", e["recs"][0]["input_wh"][0])))
    return rows


def rect150(est):
    manifest = json.load(open(C.MANIFEST))
    side = {str(im["id"]): im["rect_size"][0] for im in manifest["images"]}
    longp, wide, allm = [], defaultdict(dict), []
    for (m, st, rep, meth, img), e in sorted(est.items(), key=lambda x: (x[0][0], x[0][3], int(x[0][4]) if x[0][1] == "rect150" else 0)):
        if st != "rect150" or rep != 0:
            continue
        r = e["recs"][0]
        trunc = any(x["parsed"].get("truncated") for x in e["recs"])
        allm.append(dict(model=m, method=meth, image_id=int(img), native_crop_px=side.get(img),
                         vertical=e["v"], horizontal=e["h"], weave=e["weave"], parse_ok=e["ok"],
                         truncated=trunc, seconds=e["sec"]))
        if meth != "P1":
            continue
        longp.append(dict(model=m, image_id=int(img), native_crop_px=side.get(img), vertical=e["v"],
                          horizontal=e["h"], weave=e["weave"], parse_ok=e["ok"], seconds=e["sec"], usd=e["cost"],
                          finish_reason=r.get("finish_reason"), provider=r.get("provider", "local"),
                          model_repo=r.get("repo"), revision=r.get("revision"), raw_file_key=r.get("image_path")))
        wide[int(img)][f"{m}__v"] = e["v"]
        wide[int(img)][f"{m}__h"] = e["h"]
        wide[int(img)][f"{m}__weave"] = e["weave"]
    wrows = [dict(image_id=i, native_crop_px=side.get(str(i)), **wide[i]) for i in sorted(wide)]
    # summary per method (label-free on the 150)
    summ = []
    for m in (BF16, W4, MLX):
        for meth in ("P1", "P3", "P3s", "P4"):
            L = [r for r in allm if r["model"] == m and r["method"] == meth]
            if not L:
                continue
            ok = [r for r in L if r["parse_ok"]]
            vals = np.array([[r["vertical"], r["horizontal"]] for r in ok], float) if ok else np.zeros((0, 2))
            small = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] < 1000]
            big = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] >= 1000]
            secs = [r["seconds"] for r in L if r["seconds"] is not None and not math.isnan(r["seconds"])]
            uniq = len({(r["vertical"], r["horizontal"]) for r in ok})
            summ.append(dict(model=m, method=meth, n_run=len(L), n_parsed=len(ok),
                             n_truncated=sum(1 for r in L if r["truncated"]),
                             n_distinct_answers=uniq,
                             median_v=float(np.median(vals[:, 0])) if len(vals) else None,
                             median_h=float(np.median(vals[:, 1])) if len(vals) else None,
                             p05=float(np.percentile(vals, 5)) if len(vals) else None,
                             p95=float(np.percentile(vals, 95)) if len(vals) else None,
                             mean_tpi_small_crops=float(np.mean(small)) if small else None, n_small=len(small),
                             mean_tpi_large_crops=float(np.mean(big)) if big else None, n_large=len(big),
                             sec_per_img=float(np.mean(secs)) if secs else None))
    return longp, wrows, allm, summ


def stability(est, gt):
    """3 input resizes (repeat 0 = 644 px, 1 = 588 px, 2 = 700 px); P1 and P4."""
    out = []
    for st in ("legacy", "rect150"):
        for meth in ("P1", "P4"):
            reps = [rows_for(est, BF16, st, meth, rp) for rp in (0, 1, 2)]
            if not all(reps):
                continue
            imgs = [i for i in reps[1] if i in reps[0] and i in reps[2]]
            full = [i for i in imgs if all(reps[k][i]["ok"] for k in range(3))]
            sds, rng, ident, relr = [], [], 0, []
            for i in full:
                vs = np.array([reps[k][i]["v"] for k in range(3)], float)
                hs = np.array([reps[k][i]["h"] for k in range(3)], float)
                sds += [vs.std(ddof=1), hs.std(ddof=1)]
                rng += [np.ptp(vs), np.ptp(hs)]
                relr += [np.ptp(vs) / max(vs.mean(), 1e-9) * 100, np.ptp(hs) / max(hs.mean(), 1e-9) * 100]
                ident += int(np.ptp(vs) == 0 and np.ptp(hs) == 0)
            row = dict(model=BF16, set=st, method=meth, resizes_px="644/588/700", n_images=len(imgs),
                       n_images_all_parsed=len(full),
                       mean_sd_tpi=float(np.mean(sds)) if sds else None,
                       mean_range_tpi=float(np.mean(rng)) if rng else None,
                       max_range_tpi=float(np.max(rng)) if rng else None,
                       mean_range_pct=float(np.mean(relr)) if relr else None,
                       identical_all3=ident)
            if st == "legacy":
                for k, side in zip(range(3), (644, 588, 700)):
                    rows = {i: reps[k][i] for i in imgs}
                    mb, ma = EV.metrics(rows, gt, "best"), EV.metrics(rows, gt, "axis")
                    row[f"mae_best_{side}"] = mb.get("mae")
                    row[f"within10_best_{side}"] = mb.get("within10")
                    row[f"mae_axis_{side}"] = ma.get("mae")
                    row[f"n_ok_{side}"] = mb.get("n_ok")
            out.append(row)
    return out


def pointing_diag(recs):
    rows = []
    for r in recs:
        if not r["prompt_id"].startswith("P4") or r.get("error"):
            continue
        pts = r["parsed"].get("points", [])
        axis = 0 if r["view"].endswith("_h") else 1
        cross = 1 - axis
        xs = np.sort(np.array([p[axis] for p in pts], float)) if pts else np.array([])
        gaps = np.diff(xs) if len(xs) > 2 else np.array([])
        cv = float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else None
        off = float(np.mean([abs(p[cross] - 500) for p in pts])) if pts else None
        span = float(xs[-1] - xs[0]) if len(xs) > 1 else None
        # Mac molmo2_pointing_diagnostics.csv columns first (same order), then extras
        rows.append(dict(image=str(r["image_id"]), view=r["view"], n_points=len(pts), spacing_cv=cv,
                         truncated=r["parsed"].get("truncated"), max_tokens=r.get("max_tokens"),
                         mean_offset_from_line_per1000=off, seconds=r.get("seconds"),
                         model=r["short"], set=r["set"], repeat=r["repeat"],
                         generation_tokens=r.get("generation_tokens"), finish_reason=r.get("finish_reason"),
                         span_per1000=span, resize_side=r.get("resize_side")))
    return rows


def stability_mac_format(est, gt):
    """stability_P1_legacy.csv in the Mac column layout.  With greedy decoding a repeat at the same
    input is deterministic, so the three 'repeats' are the three input resizes (644/588/700 px)."""
    out = []
    for m in (BF16,):
        reps = [rows_for(est, m, "legacy", "P1", rp) for rp in (0, 1, 2)]
        if not all(reps):
            continue
        sds, ident, maxdiff, n_img = [], 0, [], 0
        for img in C.LEGACY_NAMES:
            if not all(img in reps[k] and reps[k][img]["ok"] for k in range(3)):
                continue
            n_img += 1
            vs = np.array([reps[k][img]["v"] for k in range(3)], float)
            hs = np.array([reps[k][img]["h"] for k in range(3)], float)
            sds += [vs.std(ddof=1), hs.std(ddof=1)]
            maxdiff += [vs.max() - vs.min(), hs.max() - hs.min()]
            ident += int(np.ptp(vs) == 0 and np.ptp(hs) == 0)
        out.append(dict(model=m, repeats=3, n_images_all_parsed=n_img,
                        mean_sd_tpi=float(np.mean(sds)) if sds else None,
                        mean_range_tpi=float(np.mean(maxdiff)) if maxdiff else None,
                        max_range_tpi=float(np.max(maxdiff)) if maxdiff else None,
                        identical_all3=ident,
                        mae_best_per_repeat=[EV.metrics(reps[k], gt, "best").get("mae") for k in range(3)],
                        perturbation="greedy decoding; repeats = input resized to 644/588/700 px"))
    return out


def _per_image_err(rows, gt, img, mode):
    e, t = EV.errors(rows[img]["v"], rows[img]["h"], gt[img], mode)
    return float(np.mean(np.abs(e))), float(np.mean(np.abs(np.array(e)) / np.array(t) * 100))


def bootstrap(est, gt, B=10000, seed=0):
    """Image-level percentile bootstrap (B resamples of the GT images) of MAE / mean % error for every
    method with all GT images parsed, plus paired bootstrap of the MAE difference between models."""
    rng = np.random.default_rng(seed)
    out, paired = [], []
    for m in (BF16, W4, MLX):
        for meth in METHODS:
            rows = rows_for(est, m, "legacy", meth)
            imgs = [i for i in gt if i in rows and rows[i]["ok"]]
            if len(imgs) < len(gt):
                continue
            for mode in ("best", "axis"):
                ae = np.array([_per_image_err(rows, gt, i, mode)[0] for i in imgs])
                pe = np.array([_per_image_err(rows, gt, i, mode)[1] for i in imgs])
                idx = rng.integers(0, len(imgs), size=(B, len(imgs)))
                bm, bp = ae[idx].mean(1), pe[idx].mean(1)
                out.append(dict(model=m, method=meth, mode=mode, n_images=len(imgs), mae=float(ae.mean()),
                                mae_ci_lo=float(np.percentile(bm, 2.5)), mae_ci_hi=float(np.percentile(bm, 97.5)),
                                mape=float(pe.mean()), mape_ci_lo=float(np.percentile(bp, 2.5)),
                                mape_ci_hi=float(np.percentile(bp, 97.5)), B=B))
    for a_, b_ in ((BF16, W4), (BF16, MLX)):
        for meth in METHODS:
            ra, rb = rows_for(est, a_, "legacy", meth), rows_for(est, b_, "legacy", meth)
            imgs = [i for i in gt if i in ra and i in rb and ra[i]["ok"] and rb[i]["ok"]]
            if len(imgs) < 3:
                continue
            for mode in ("best", "axis"):
                d = np.array([_per_image_err(ra, gt, i, mode)[0] - _per_image_err(rb, gt, i, mode)[0] for i in imgs])
                idx = rng.integers(0, len(imgs), size=(B, len(imgs)))
                bd = d[idx].mean(1)
                paired.append(dict(model_a=a_, model_b=b_, method=meth, mode=mode, n_images=len(imgs),
                                   mae_a_minus_b=float(d.mean()), ci_lo=float(np.percentile(bd, 2.5)),
                                   ci_hi=float(np.percentile(bd, 97.5)),
                                   frac_boot_a_better=float(np.mean(bd < 0)), B=B, images=",".join(imgs)))
    return out, paired


CONTRASTS = [  # (name, model A, model B, what the difference isolates)
    ("weights_4bit", BF16, W4, "4-bit MLX weights only (same HF backend, preprocessing, prompt, bf16 activations)"),
    ("backend", W4, MLX, "MLX runtime: fp16 activations/kernels, PIL-antialiased low-res view, mlx-vlm prompt"),
    ("total", BF16, MLX, "reference bf16 HF vs the Mac MLX-4bit run"),
]


def _method_of(view, pid):
    if view == "crop":
        return pid
    if view == "qc":
        return "P3"
    if view in ("guide_h", "guide_v"):
        return "P4"
    if view in ("qc_guide_h", "qc_guide_v"):
        return "P4q"
    if view.startswith("qd"):
        return "P3s"
    return None


def quantisation(recs, est, gt):
    """Pairwise comparison on identical (set, image, view, prompt) inputs, repeat 0."""
    by = defaultdict(dict)
    for r in recs:
        if r["repeat"] != 0 or r.get("error"):
            continue
        by[(r["set"], str(r["image_id"]), r["view"], r["prompt_id"])][r["short"]] = r
    paired, summ = [], []
    for cname, A_, B_, what in CONTRASTS:
        P = []
        for (st, img, view, pid), d in sorted(by.items()):
            if A_ not in d or B_ not in d:
                continue
            a, b = d[A_], d[B_]
            row = dict(contrast=cname, model_a=A_, model_b=B_, set=st, image=img, view=view, prompt_id=pid,
                       method=_method_of(view, pid),
                       prompt_version_a=a["prompt_version"], prompt_version_b=b["prompt_version"],
                       max_tokens_a=a["max_tokens"], max_tokens_b=b["max_tokens"],
                       prompt_tokens_a=a["prompt_tokens"], prompt_tokens_b=b["prompt_tokens"],
                       gen_tokens_a=a["generation_tokens"], gen_tokens_b=b["generation_tokens"],
                       seconds_a=a["seconds"], seconds_b=b["seconds"],
                       identical_text=a["raw"].strip() == b["raw"].strip())
            if pid.startswith("P4"):
                na, nb = a["parsed"].get("n_points", 0), b["parsed"].get("n_points", 0)
                row.update(value_a=na, value_b=nb, abs_diff=abs(na - nb),
                           rel_diff_pct=abs(na - nb) / max((na + nb) / 2, 1e-9) * 100,
                           truncated_a=a["parsed"].get("truncated"), truncated_b=b["parsed"].get("truncated"))
            else:
                pa, pb = a["parsed"], b["parsed"]
                row.update(value_a=f"{pa.get('vertical')}x{pa.get('horizontal')}/{pa.get('weave')}",
                           value_b=f"{pb.get('vertical')}x{pb.get('horizontal')}/{pb.get('weave')}")
                if None not in (pa.get("vertical"), pa.get("horizontal"), pb.get("vertical"), pb.get("horizontal")):
                    row["abs_diff"] = (abs(pa["vertical"] - pb["vertical"]) + abs(pa["horizontal"] - pb["horizontal"])) / 2
            P.append(row)
        paired += P
        for st in ("legacy", "rect150"):
            for meth in METHODS:
                pr = [p for p in P if p["set"] == st and p["method"] == meth]
                if not pr:
                    continue
                ra, rb = rows_for(est, A_, st, meth), rows_for(est, B_, st, meth)
                imgs = sorted({p["image"] for p in pr})
                okb = [i for i in imgs if i in ra and i in rb and ra[i]["ok"] and rb[i]["ok"]]
                diffs = [p["abs_diff"] for p in pr if p.get("abs_diff") is not None]
                row = dict(contrast=cname, isolates=what, model_a=A_, model_b=B_, set=st, method=meth,
                           n_records=len(pr), n_images=len(imgs),
                           identical_text=sum(p["identical_text"] for p in pr),
                           mean_abs_diff_per_call=float(np.mean(diffs)) if diffs else None,
                           median_abs_diff_per_call=float(np.median(diffs)) if diffs else None,
                           mean_rel_diff_pct=float(np.mean([p["rel_diff_pct"] for p in pr if "rel_diff_pct" in p]))
                           if any("rel_diff_pct" in p for p in pr) else None)
                if okb:
                    # per-image estimate difference (TPI, mean over the 2 directions)
                    row["mean_abs_estimate_diff_tpi"] = float(np.mean(
                        [(abs(ra[i]["v"] - rb[i]["v"]) + abs(ra[i]["h"] - rb[i]["h"])) / 2 for i in okb]))
                    row["n_identical_estimates"] = sum(1 for i in okb if ra[i]["v"] == rb[i]["v"] and ra[i]["h"] == rb[i]["h"])
                if st == "legacy":
                    both = [i for i in gt if i in okb]
                    row["n_gt_images_both"] = len(both)
                    for tag, rows in (("a", ra), ("b", rb)):
                        if both:
                            mb = EV.metrics({i: rows[i] for i in both}, {i: gt[i] for i in both}, "best")
                            ma = EV.metrics({i: rows[i] for i in both}, {i: gt[i] for i in both}, "axis")
                            row[f"mae_best_{tag}"] = mb.get("mae")
                            row[f"mape_best_{tag}"] = mb.get("mape")
                            row[f"within10_best_{tag}"] = mb.get("within10")
                            row[f"mae_axis_{tag}"] = ma.get("mae")
                    row["gt_images"] = ",".join(both)
                for tag, rows in (("a", ra), ("b", rb)):
                    okv = [(rows[i]["v"], rows[i]["h"]) for i in imgs if i in rows and rows[i]["ok"]]
                    row[f"n_distinct_answers_{tag}"] = len(set(okv))
                sa = [p["seconds_a"] for p in pr if p["seconds_a"] == p["seconds_a"]]
                sb = [p["seconds_b"] for p in pr if p["seconds_b"] == p["seconds_b"]]
                row["median_s_per_call_a"] = float(np.median(sa)) if sa else None
                row["median_s_per_call_b"] = float(np.median(sb)) if sb else None
                summ.append(row)
    return paired, summ


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = C.legacy_gt()
    recs = load_records()
    est = EV.estimates(recs)
    s = dict(gt_images=sorted(gt), n_gt=len(gt),
             n_records={m: sum(1 for r in recs if r["short"] == m) for m in (BF16, W4, MLX)})
    # Mac-schema files hold ONLY the GPU models (molmo2_8b_bf16, molmo2_8b_bf16_w4mlx), so their rows can
    # be appended to <repo root>/vlm/results/*.csv without duplicating the Mac's own molmo2_8b rows.
    GPU = (BF16, W4)
    A = stageA(est, gt, GPU)
    EV.write_csv(os.path.join(OUT, "stageA_legacy_metrics.csv"), A)
    s["stageA"] = A
    # side-by-side incl. the read-only Mac MLX-4bit snapshot (MLX P2/P3/P4/P4q ran on only a few images)
    A3 = stageA(est, gt, (BF16, W4, MLX))
    EV.write_csv(os.path.join(OUT, "stageA_with_mlx_snapshot.csv"), A3)
    s["stageA_with_mlx_snapshot"] = A3
    bl_ax, bl_be = EV.loo_mean_baseline(gt)
    s["loo_mean_baseline"] = dict(axis=bl_ax, best=bl_be)
    EV.write_csv(os.path.join(OUT, "legacy_predictions.csv"), legacy_predictions(est, gt, GPU))
    EV.write_csv(os.path.join(OUT, "legacy_predictions_with_mlx_snapshot.csv"), legacy_predictions(est, gt))
    longp, wrows, allm, summ = rect150(est)
    EV.write_csv(os.path.join(OUT, "rect150_P1_predictions_long.csv"), longp)
    EV.write_csv(os.path.join(OUT, "rect150_P1_predictions_wide.csv"), wrows)
    EV.write_csv(os.path.join(OUT, "rect150_all_methods_long.csv"), allm)
    EV.write_csv(os.path.join(OUT, "rect150_summary.csv"), summ)
    # Mac rect150_P1_summary.csv layout (P1 rows only)
    mac_cols = ["model", "n_run", "n_parsed", "median_v", "median_h", "p05", "p95", "mean_tpi_small_crops",
                "n_small", "mean_tpi_large_crops", "n_large", "sec_per_img"]
    EV.write_csv(os.path.join(OUT, "rect150_P1_summary.csv"),
                 [dict({k: r[k] for k in mac_cols}, usd_total=0.0, usd_per_img=0.0, n_distinct_answers=r["n_distinct_answers"])
                  for r in summ if r["method"] == "P1"])
    s["rect150"] = summ
    st = stability(est, gt)
    EV.write_csv(os.path.join(OUT, "stability_resize.csv"), st)
    s["stability_resize"] = st
    stm = stability_mac_format(est, gt)
    EV.write_csv(os.path.join(OUT, "stability_P1_legacy.csv"), stm)
    s["stability"] = stm
    pdg = pointing_diag(recs)
    EV.write_csv(os.path.join(OUT, "molmo2_pointing_diagnostics.csv"), pdg)
    pts = {}
    for m in (BF16, W4, MLX):
        for sname in ("legacy", "rect150"):
            P = [x for x in pdg if x["model"] == m and x["set"] == sname and x["repeat"] == 0]
            if not P:
                continue
            cvs = [x["spacing_cv"] for x in P if x["spacing_cv"] is not None]
            full = [x for x in P if not x["view"].startswith("qc")]
            pts[f"{m}|{sname}"] = dict(
                n_outputs=len(P), n_truncated=sum(1 for x in P if x["truncated"]),
                n_finish_length=sum(1 for x in P if x["finish_reason"] == "length"),
                frac_uniform_grid_cv_lt_0p05=float(np.mean(np.array(cvs) < 0.05)) if cvs else None,
                frac_cv_lt_0p10=float(np.mean(np.array(cvs) < 0.10)) if cvs else None,
                median_spacing_cv=float(np.median(cvs)) if cvs else None,
                median_n_points_full=float(np.median([x["n_points"] for x in full])) if full else None,
                median_span_full_per1000=float(np.median([x["span_per1000"] for x in full if x["span_per1000"] is not None])) if full else None,
                median_offset_from_line_per1000=float(np.median([x["mean_offset_from_line_per1000"] for x in P if x["mean_offset_from_line_per1000"] is not None])) if P else None)
    s["pointing"] = pts
    bs, bsp = bootstrap(est, gt)
    EV.write_csv(os.path.join(OUT, "bootstrap_ci_legacy.csv"), bs)
    EV.write_csv(os.path.join(OUT, "bootstrap_paired_diff_legacy.csv"), bsp)
    s["bootstrap"], s["bootstrap_paired"] = bs, bsp
    paired, qs = quantisation(recs, est, gt)
    EV.write_csv(os.path.join(OUT, "quant_paired_records.csv"), paired)
    EV.write_csv(os.path.join(OUT, "quant_summary.csv"), qs)
    s["quantisation"] = qs
    # P1 answer distribution (degeneracy)
    for m in (BF16, W4, MLX):
        for sname in ("legacy", "rect150"):
            r = rows_for(est, m, sname, "P1")
            if r:
                from collections import Counter
                c = Counter((e["v"], e["h"]) for e in r.values() if e["ok"])
                s[f"P1_answer_hist|{m}|{sname}"] = {f"{k[0]}x{k[1]}": v for k, v in c.most_common(8)}
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print_tables(s)


def f(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def print_tables(s):
    print(f"\n## Stage A: legacy GT images (n = {s['n_gt']})\n")
    print("| Model | Prompt | parsed | MAE best | %err best | <=10% best | MAE axis | %err axis | <=10% axis | warp MAE | weft MAE | bias best | MAE best plain | weave | rho best | distinct | s/img |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s.get("stageA_with_mlx_snapshot", s["stageA"]):
        print(f"| {r['label']} | {r['method']} | {r['n_ok']}/{r['n_gt_run']} | {f(r['mae_best'])} | {f(r['mape_best'])} | "
              f"{r['within10_best'] if r['within10_best'] is not None else '-'}/{r['n_values'] or 0} | {f(r['mae_axis'])} | "
              f"{f(r['mape_axis'])} | {r['within10_axis'] if r['within10_axis'] is not None else '-'}/{r['n_values'] or 0} | "
              f"{f(r['mae_warp'])} | {f(r['mae_weft'])} | {f(r['bias_best'])} | {f(r['mae_best_plain'])} ({r['n_ok_plain']}/{r['n_plain']}) | "
              f"{r['weave_correct']}/{r['weave_n']} | {f(r['spearman_best'], 2)} | {r['n_distinct_answers_gt']} | {f(r['sec_per_img'])} |")
    b = s["loo_mean_baseline"]
    print(f"| LOO mean predictor | - | 13/13 | {f(b['best']['mae'])} | {f(b['best']['mape'])} | {b['best']['within10']}/26 | "
          f"{f(b['axis']['mae'])} | {f(b['axis']['mape'])} | {b['axis']['within10']}/26 | {f(b['axis']['mae_warp'])} | {f(b['axis']['mae_weft'])} | - | - | - | - | - | - |")
    for key in ("rect150", "stability_resize", "quantisation"):
        print(f"\n## {key}\n")
        for r in s.get(key, []):
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("\n## pointing\n")
    for k, v in s.get("pointing", {}).items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    for k, v in s.items():
        if k.startswith("P1_answer_hist"):
            print(k, v)


if __name__ == "__main__":
    main()

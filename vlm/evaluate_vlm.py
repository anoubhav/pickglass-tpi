"""Evaluate the VLM baselines (fixed roster, MODEL_ROSTER.md) on the legacy hand counts and
export the 150-image predictions.

Reads raw/or-{model}__{set}__main.jsonl (OpenRouter) and raw/molmo2_8b__{set}.jsonl (local MLX);
pilot / smoke-test files and non-roster local models are ignored.

Estimators (per model, per image):
    P1   whole crop, direct JSON                      -> (vertical, horizontal)
    P2   whole crop, step-by-step then JSON
    P3   central quarter-inch patch, JSON             -> 4 x (vertical, horizontal)
    P3s  sum of the counts on the 4 diagonal quarter-inch patches (GPT-6.1-sol, Gemini Flash only)
    P4   Molmo2 pointing on a red guide line through the whole crop (n points)
    P4q  Molmo2 pointing on a guide line through the central quarter-inch patch (4 x n points)
    P1x4 P1 on the crop downscaled x4 (resolution ablation, set legacy_x4)

Metrics on the 13 usable legacy images (GT from the legacy hand-count spreadsheet, parser reused
from pipeline/run_pipeline.py via gt_legacy.py):
    axis-matched    vertical_yarns <-> warp (ends), horizontal_yarns <-> weft (picks)
    best-assignment sorted pair vs sorted pair (photo orientation is not recorded)
    MAE (TPI) and mean % error over the 2 x n_ok values, within-10% count, bias (mean signed
    error), per-axis MAE (axis-matched), parse failures, wall-clock s/image and US$/image.
    An image counts as parsed only if both counts were parsed; MAE etc. are over parsed images.
Reference: leave-one-out mean predictor (predicts the mean of the other 12 GT images).

    python evaluate_vlm.py            # writes results/*.csv, results/summary.json, prints tables
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
from gt_legacy import legacy_gt          # noqa: E402
from parse_output import parse_points    # noqa: E402

RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "results")
LEGACY_NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
MODELS = ["gpt-6.1-sol", "claude-fable-5.1", "gemini-3.1-pro", "gemini-3.8-flash",
          "qwen3.8-max-prime", "qwen3.8-27b", "gemma-4-26b-a4b", "molmo2_8b"]
LABEL = {"gpt-6.1-sol": "GPT-6.1-sol", "claude-fable-5.1": "Claude Fable 5.1",
         "gemini-3.1-pro": "Gemini 3.1 Pro (preview)", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "qwen3.8-max-prime": "Qwen3.8-Max-Prime", "qwen3.8-27b": "Qwen3.8-27B (open)",
         "gemma-4-26b-a4b": "Gemma-4-26B-A4B (open)", "molmo2_8b": "Molmo2-8B 4-bit (local)"}
METHODS = ["P1", "P2", "P3", "P3s", "P4", "P4q"]


# ----------------------------------------------------------------------------
def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(RAW, "or-*__*__*.jsonl"))):
        tag = os.path.basename(f).rsplit("__", 1)[1][:-len(".jsonl")]
        if tag.startswith("pilot"):
            continue                      # protocol pilots (logged, not reported)
        for line in open(f):
            r = json.loads(line)
            r["short"] = r["model"].split(":", 1)[1] + ("" if tag == "main" else f"@{tag}")
            recs.append(r)
    for f in sorted(glob.glob(os.path.join(RAW, "molmo2_8b__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            r["short"], r["repeat"], r["cost_usd"] = "molmo2_8b", 0, 0.0
            if r["prompt_id"].startswith("P4") and not r.get("error"):
                r["parsed"] = parse_points(r["raw"])      # re-parse with the current (truncation-aware) parser
            recs.append(r)
    # keep the last successful record per key (resumed runs append; errors are retried)
    best = {}
    for r in recs:
        k = (r["short"], r["set"], r.get("repeat", 0), str(r["image_id"]), r["view"], r["prompt_id"])
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return list(best.values())


def estimates(recs):
    """-> {(model, set, repeat, method, image): dict(v, h, weave, ok, sec, cost, recs)}"""
    by = defaultdict(dict)
    for r in recs:
        by[(r["short"], r["set"], r.get("repeat", 0), str(r["image_id"]))][(r["view"], r["prompt_id"])] = r
    out = {}

    def sec(*rs):
        s = [x.get("seconds") for x in rs]
        return float(sum(s)) if all(isinstance(x, (int, float)) and not math.isnan(x) for x in s) else float("nan")

    def cost(*rs):
        return float(sum(float(x.get("cost_usd") or 0) for x in rs))

    for (m, st, rep, img), d in by.items():
        def put(method, v, h, w, rs, ok=None):
            ok = (v is not None and h is not None) if ok is None else ok
            out[(m, st, rep, method, img)] = dict(v=v, h=h, weave=w, ok=ok, sec=sec(*rs), cost=cost(*rs),
                                                  recs=rs)
        if st in ("legacy", "rect150"):
            for pid in ("P1", "P2"):
                r = d.get(("crop", pid))
                if r:
                    p = r["parsed"]
                    put(pid, p.get("vertical"), p.get("horizontal"), p.get("weave"), [r])
            r = d.get(("qc", "P3"))
            if r:
                p = r["parsed"]
                v, h = p.get("vertical"), p.get("horizontal")
                put("P3", None if v is None else 4 * v, None if h is None else 4 * h, p.get("weave"), [r])
            rs = [d.get((f"qd{k}", "P3")) for k in range(4)]
            if all(rs):      # P3s: sum over the 4 diagonal quarter-inch patches (covers the full inch)
                vs = [x["parsed"].get("vertical") for x in rs]
                hs = [x["parsed"].get("horizontal") for x in rs]
                put("P3s", None if None in vs else sum(vs), None if None in hs else sum(hs), rs[0]["parsed"].get("weave"), rs)
            for meth, (vh, vv), k in (("P4", ("guide_h", "guide_v"), 1), ("P4q", ("qc_guide_h", "qc_guide_v"), 4)):
                rh = next((x for (vw, _), x in d.items() if vw == vh), None)
                rv = next((x for (vw, _), x in d.items() if vw == vv), None)
                if rh and rv:
                    nh, nv = rh["parsed"].get("n_points", 0), rv["parsed"].get("n_points", 0)
                    okh = rh["parsed"].get("method", "none") != "none"
                    okv = rv["parsed"].get("method", "none") != "none"
                    put(meth, k * nh if okh else None, k * nv if okv else None, None, [rh, rv])
        elif st.startswith("legacy_x"):
            k = st.split("_x")[1]
            r = d.get((f"crop_x{k}", "P1"))
            if r:
                p = r["parsed"]
                put(f"P1x{k}", p.get("vertical"), p.get("horizontal"), p.get("weave"), [r])
    return out


# ----------------------------------------------------------------------------
def errors(v, h, g, mode):
    """Signed errors (2 values) and GT values for one image."""
    if mode == "axis":
        pred, gt = [v, h], [g["warp"], g["weft"]]
    else:
        pred, gt = sorted([v, h]), sorted([g["warp"], g["weft"]])
    return [p - t for p, t in zip(pred, gt)], gt


def metrics(rows, gt, mode):
    """rows: {image: est}.  Over the GT images present in gt."""
    e_all, t_all, warp_e, weft_e = [], [], [], []
    n_ok = n_fail = n_missing = 0
    for img, g in gt.items():
        est = rows.get(img)
        if est is None:
            n_missing += 1          # not run (e.g. a partial local run), not a parse failure
            continue
        if not est["ok"]:
            n_fail += 1
            continue
        n_ok += 1
        e, t = errors(est["v"], est["h"], g, mode)
        e_all += e
        t_all += t
        if mode == "axis":
            warp_e.append(e[0])
            weft_e.append(e[1])
    if not e_all:
        return dict(n_ok=0, n_fail=n_fail, n_missing=n_missing)
    e, t = np.array(e_all, float), np.array(t_all, float)
    pe = np.abs(e) / t * 100
    res = dict(n_ok=n_ok, n_fail=n_fail, n_missing=n_missing, mae=float(np.abs(e).mean()), mape=float(pe.mean()),
               within10=int((pe <= 10).sum()), n_values=int(len(e)), bias=float(e.mean()),
               medae=float(np.median(np.abs(e))))
    if mode == "axis":
        res.update(mae_warp=float(np.abs(warp_e).mean()), mae_weft=float(np.abs(weft_e).mean()))
    return res


def loo_mean_baseline(gt):
    keys = list(gt)
    rows = {}
    for k in keys:
        others = [gt[o] for o in keys if o != k]
        rows[k] = dict(v=np.mean([o["warp"] for o in others]), h=np.mean([o["weft"] for o in others]), ok=True)
    rows_sorted = {}
    for k in keys:
        others = [gt[o] for o in keys if o != k]
        lo = np.mean([min(o["warp"], o["weft"]) for o in others])
        hi = np.mean([max(o["warp"], o["weft"]) for o in others])
        rows_sorted[k] = dict(v=lo, h=hi, ok=True)
    return metrics(rows, gt, "axis"), metrics(rows_sorted, gt, "best")


def weave_acc(rows, gt):
    n = c = 0
    for img, g in gt.items():
        est = rows.get(img)
        if est is None or not est.get("weave") or not g.get("weave_name"):
            continue
        n += 1
        c += int(est["weave"] == g["weave_name"])
    return c, n


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    gt = legacy_gt()
    recs = load_records()
    est = estimates(recs)
    for m in sorted({r["short"] for r in recs}):
        if m not in MODELS:                # extra tagged runs, e.g. gpt-6.1-sol@reasoning-high
            MODELS.append(m)
            base, tag = m.split("@") if "@" in m else (m, "")
            LABEL[m] = f"{LABEL.get(base, base)} [{tag}]"
    summary = dict(gt_images=sorted(gt), n_gt=len(gt))

    # ---------------- Stage A table ----------------
    rows_out = []
    for m in MODELS:
        for meth in METHODS + ["P1x4", "P1x8", "P1x16"]:
            st = ("legacy_x" + meth[3:]) if meth.startswith("P1x") else "legacy"
            rows = {img: e for (mm, s, rep, me, img), e in est.items()
                    if mm == m and s == st and rep == 0 and me == meth}
            if not rows:
                continue
            ax, be = metrics(rows, gt, "axis"), metrics(rows, gt, "best")
            gt_plain = {k: g for k, g in gt.items() if g["weave_name"] == "plain"}   # drops formula-derived twill warps
            bp, ap = metrics(rows, gt_plain, "best"), metrics(rows, gt_plain, "axis")
            gt_rows = [rows[i] for i in gt if i in rows]
            all20 = [rows[i] for i in LEGACY_NAMES if i in rows]
            secs = [r["sec"] for r in gt_rows if not math.isnan(r["sec"])]
            wc, wn = weave_acc(rows, gt)
            rows_out.append(dict(
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
                sec_per_img=float(np.mean(secs)) if secs else None,
                usd_per_img=float(np.mean([r["cost"] for r in gt_rows])) if gt_rows else None))
    write_csv(os.path.join(OUT, "stageA_legacy_metrics.csv"), rows_out)
    summary["stageA"] = rows_out
    bl_ax, bl_be = loo_mean_baseline(gt)
    summary["loo_mean_baseline"] = dict(axis=bl_ax, best=bl_be)

    # per-image predictions (legacy, all models/methods)
    per_img = []
    for (m, st, rep, meth, img), e in sorted(est.items()):
        if not st.startswith("legacy"):
            continue
        g = gt.get(img)
        per_img.append(dict(model=m, set=st, repeat=rep, method=meth, image=img,
                            vertical=e["v"], horizontal=e["h"], weave=e["weave"], ok=e["ok"],
                            gt_warp=g["warp"] if g else None, gt_weft=g["weft"] if g else None,
                            gt_weave=g["weave_name"] if g else None, seconds=e["sec"], usd=e["cost"]))
    write_csv(os.path.join(OUT, "legacy_predictions.csv"), per_img)

    # ---------------- stability (P1, repeats) ----------------
    stab = []
    for m in MODELS:
        reps = sorted({rep for (mm, s, rep, me, img) in est if mm == m and s == "legacy" and me == "P1"})
        if len(reps) < 3:
            continue
        per = defaultdict(dict)
        for (mm, s, rep, me, img), e in est.items():
            if mm == m and s == "legacy" and me == "P1":
                per[img][rep] = e
        sds, ident, maxdiff, n_img = [], 0, [], 0
        for img, d in per.items():
            if not all(r in d and d[r]["ok"] for r in reps[:3]):
                continue
            n_img += 1
            vs = np.array([d[r]["v"] for r in reps[:3]], float)
            hs = np.array([d[r]["h"] for r in reps[:3]], float)
            sds += [vs.std(ddof=1), hs.std(ddof=1)]
            maxdiff += [vs.max() - vs.min(), hs.max() - hs.min()]
            ident += int(np.ptp(vs) == 0 and np.ptp(hs) == 0)
        maes = []
        for rep in reps[:3]:
            rows = {img: e for (mm, s, rp, me, img), e in est.items()
                    if mm == m and s == "legacy" and me == "P1" and rp == rep}
            maes.append(metrics(rows, gt, "best").get("mae"))
        stab.append(dict(model=m, repeats=len(reps[:3]), n_images_all_parsed=n_img,
                         mean_sd_tpi=float(np.mean(sds)) if sds else None,
                         mean_range_tpi=float(np.mean(maxdiff)) if maxdiff else None,
                         max_range_tpi=float(np.max(maxdiff)) if maxdiff else None,
                         identical_all3=ident, mae_best_per_repeat=maes))
    write_csv(os.path.join(OUT, "stability_P1_legacy.csv"), stab)
    summary["stability"] = stab

    # ---------------- resolution ablation ----------------
    abl = []
    for m in MODELS:
        a = {img: e for (mm, s_, rep, me, img), e in est.items() if mm == m and s_ == "legacy" and rep == 0 and me == "P1"}
        for k in ("4", "8", "16"):
            b = {img: e for (mm, s_, rep, me, img), e in est.items()
                 if mm == m and s_ == f"legacy_x{k}" and rep == 0 and me == f"P1x{k}"}
            if not a or not b:
                continue
            both = [i for i in gt if i in a and i in b and a[i]["ok"] and b[i]["ok"]]
            ma = metrics({i: a[i] for i in both}, {i: gt[i] for i in both}, "best")
            mb = metrics({i: b[i] for i in both}, {i: gt[i] for i in both}, "best")
            pair20 = [i for i in LEGACY_NAMES if i in a and i in b and a[i]["ok"] and b[i]["ok"]]
            d20 = [abs(a[i][c] - b[i][c]) for i in pair20 for c in ("v", "h")]
            abl.append(dict(model=m, factor=int(k), n_gt_paired=len(both), mae_best_full=ma.get("mae"),
                            mae_best_down=mb.get("mae"), mape_best_full=ma.get("mape"), mape_best_down=mb.get("mape"),
                            within10_full=ma.get("within10"), within10_down=mb.get("within10"),
                            bias_full=ma.get("bias"), bias_down=mb.get("bias"),
                            n_pairs_all20=len(pair20), mean_abs_change_tpi=float(np.mean(d20)) if d20 else None,
                            mean_pred_full=float(np.mean([np.mean([a[i]["v"], a[i]["h"]]) for i in pair20])) if pair20 else None,
                            mean_pred_down=float(np.mean([np.mean([b[i]["v"], b[i]["h"]]) for i in pair20])) if pair20 else None,
                            parse_fail_down=sum(1 for i in LEGACY_NAMES if i in b and not b[i]["ok"]),
                            n_run_down=len(b)))
    write_csv(os.path.join(OUT, "ablation_resolution.csv"), abl)
    summary["ablation_resolution"] = abl

    # ---------------- Molmo2 pointing diagnostics ----------------
    pd_rows = []
    for r in recs:
        if r["short"] != "molmo2_8b" or not r["prompt_id"].startswith("P4") or r["set"] != "legacy":
            continue
        pts = r["parsed"].get("points", [])
        axis = 0 if r["view"].endswith("_h") else 1          # along the guide line
        cross = 1 - axis
        xs = np.sort(np.array([p[axis] for p in pts], float)) if pts else np.array([])
        gaps = np.diff(xs) if len(xs) > 2 else np.array([])
        cv = float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else None
        off = float(np.mean([abs(p[cross] - 500) for p in pts])) if pts else None   # distance from the line (0..1000)
        pd_rows.append(dict(image=str(r["image_id"]), view=r["view"], n_points=len(pts), spacing_cv=cv,
                            truncated=r["parsed"].get("truncated"), max_tokens=r.get("max_tokens"),
                            mean_offset_from_line_per1000=off, seconds=r.get("seconds")))
    write_csv(os.path.join(OUT, "molmo2_pointing_diagnostics.csv"), pd_rows)
    if pd_rows:
        cvs = [x["spacing_cv"] for x in pd_rows if x["spacing_cv"] is not None]
        summary["molmo2_pointing"] = dict(
            n_outputs=len(pd_rows), n_with_3plus_points=len(cvs),
            frac_uniform_grid_cv_lt_0p05=float(np.mean(np.array(cvs) < 0.05)) if cvs else None,
            median_spacing_cv=float(np.median(cvs)) if cvs else None,
            median_n_points_full=float(np.median([x["n_points"] for x in pd_rows if not x["view"].startswith("qc")])),
            median_n_points_quarter=float(np.median([x["n_points"] for x in pd_rows if x["view"].startswith("qc")])))

    # ---------------- Stage B: 150 rectified crops ----------------
    manifest = json.load(open(os.path.join(os.path.dirname(HERE), "annotation_tool", "data", "manifest.json")))
    side = {str(im["id"]): im["rect_size"][0] for im in manifest["images"]}
    longp, wide = [], defaultdict(dict)
    for (m, st, rep, meth, img), e in sorted(est.items(), key=lambda x: (x[0][0], int(x[0][4]) if x[0][1] == "rect150" else 0)):
        if st != "rect150" or meth != "P1" or rep != 0:
            continue
        r = e["recs"][0]
        longp.append(dict(model=m, image_id=int(img), native_crop_px=side.get(img), vertical=e["v"],
                          horizontal=e["h"], weave=e["weave"], parse_ok=e["ok"], seconds=e["sec"], usd=e["cost"],
                          finish_reason=r.get("finish_reason"), provider=r.get("provider"),
                          model_repo=r.get("repo"), revision=r.get("revision"), raw_file_key=r.get("image_path")))
        wide[int(img)][f"{m}__v"] = e["v"]
        wide[int(img)][f"{m}__h"] = e["h"]
        wide[int(img)][f"{m}__weave"] = e["weave"]
    write_csv(os.path.join(OUT, "rect150_P1_predictions_long.csv"), longp)
    wrows = [dict(image_id=i, native_crop_px=side.get(str(i)), **wide[i]) for i in sorted(wide)]
    write_csv(os.path.join(OUT, "rect150_P1_predictions_wide.csv"), wrows)
    b_sum = []
    for m in MODELS:
        L = [r for r in longp if r["model"] == m]
        if not L:
            continue
        ok = [r for r in L if r["parse_ok"]]
        small = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] < 1000]
        big = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] >= 1000]
        vals = np.array([[r["vertical"], r["horizontal"]] for r in ok], float) if ok else np.zeros((0, 2))
        secs = [r["seconds"] for r in L if r["seconds"] is not None and not math.isnan(r["seconds"])]
        b_sum.append(dict(model=m, n_run=len(L), n_parsed=len(ok),
                          median_v=float(np.median(vals[:, 0])) if len(vals) else None,
                          median_h=float(np.median(vals[:, 1])) if len(vals) else None,
                          p05=float(np.percentile(vals, 5)) if len(vals) else None,
                          p95=float(np.percentile(vals, 95)) if len(vals) else None,
                          mean_tpi_small_crops=float(np.mean(small)) if small else None, n_small=len(small),
                          mean_tpi_large_crops=float(np.mean(big)) if big else None, n_large=len(big),
                          sec_per_img=float(np.mean(secs)) if secs else None,
                          usd_total=float(sum(r["usd"] for r in L)), usd_per_img=float(np.mean([r["usd"] for r in L]))))
    write_csv(os.path.join(OUT, "rect150_P1_summary.csv"), b_sum)
    summary["stageB"] = b_sum
    # inter-model agreement on the 150 (label-free): Spearman of the mean count
    agree = []
    ms = [m for m in MODELS if any(r["model"] == m for r in longp)]
    for i, m1 in enumerate(ms):
        for m2 in ms[i + 1:]:
            common = [k for k in wide if wide[k].get(f"{m1}__v") is not None and wide[k].get(f"{m2}__v") is not None
                      and wide[k].get(f"{m1}__h") is not None and wide[k].get(f"{m2}__h") is not None]
            if len(common) < 10:
                continue
            a = [wide[k][f"{m1}__v"] + wide[k][f"{m1}__h"] for k in common]
            b = [wide[k][f"{m2}__v"] + wide[k][f"{m2}__h"] for k in common]
            agree.append(dict(m1=m1, m2=m2, n=len(common), spearman=spearman(a, b),
                              mean_abs_diff_sum=float(np.mean(np.abs(np.array(a) - np.array(b))))))
    write_csv(os.path.join(OUT, "rect150_P1_intermodel_agreement.csv"), agree)
    summary["stageB_agreement"] = agree

    # ---------------- costs ----------------
    import csv as _csv
    cost_rows = list(_csv.DictReader(open(os.path.join(HERE, "logs", "openrouter_costs.csv"))))
    tot = defaultdict(float)
    for r in cost_rows:
        if r["tag"].startswith("pilot"):
            key = "pilot"
        elif r["tag"] != "main":
            key = f"{r['set']}/{r['tag']}"
        else:
            key = r["set"] + ("" if r["repeat"] in ("0", "") else "/repeat") + (f"/{r['prompt_id']}" if r["set"] == "legacy" else "")
        tot[(r["model"], key)] += float(r["cost_usd"] or 0)
    summary["costs"] = {f"{m}|{k}": v for (m, k), v in sorted(tot.items())}
    summary["cost_total_usd"] = float(sum(float(r["cost_usd"] or 0) for r in cost_rows))
    summary["n_api_calls"] = len(cost_rows)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print_tables(summary)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def write_csv(path, rows):
    import csv
    if not rows:
        open(path, "w").close()
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def print_tables(s):
    print(f"\n## Stage A: legacy GT images (n = {s['n_gt']})\n")
    print("| Model | Prompt | parsed | MAE best | %err best | ≤10% best | MAE axis | %err axis | ≤10% axis | "
          "warp MAE | weft MAE | bias best | MAE best, plain only | weave | s/img | $/img |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["stageA"]:
        if r["method"].startswith("P1x") or "@" in r["model"]:
            continue
        nrun = r["n_gt"] - (r.get("not_run_gt") or 0)
        parsed = f"{r['n_ok']}/{r['n_gt']}" if nrun == r["n_gt"] else f"{r['n_ok']}/{nrun} (run on {nrun} of {r['n_gt']})"
        print(f"| {r['label']} | {r['method']} | {parsed} | {f(r['mae_best'])} | {f(r['mape_best'])} | "
              f"{r['within10_best'] if r['within10_best'] is not None else '–'}/{r['n_values'] or 0} | {f(r['mae_axis'])} | "
              f"{f(r['mape_axis'])} | {r['within10_axis'] if r['within10_axis'] is not None else '–'}/{r['n_values'] or 0} | "
              f"{f(r['mae_warp'])} | {f(r['mae_weft'])} | {f(r['bias_best'])} | {f(r['mae_best_plain'])} ({r['n_ok_plain']}/{r['n_plain']}) | "
              f"{r['weave_correct']}/{r['weave_n']} | "
              f"{f(r['sec_per_img'])} | {f(r['usd_per_img'], 4)} |")
    b = s["loo_mean_baseline"]
    print(f"| LOO mean predictor | – | 13/13 | {f(b['best']['mae'])} | {f(b['best']['mape'])} | "
          f"{b['best']['within10']}/26 | {f(b['axis']['mae'])} | {f(b['axis']['mape'])} | {b['axis']['within10']}/26 | "
          f"{f(b['axis']['mae_warp'])} | {f(b['axis']['mae_weft'])} | – | – | – | – | – |")
    if s["stability"]:
        print("\n## Stability (P1, 20 legacy images x 3 repeats)\n")
        for r in s["stability"]:
            print(r)
    if s["ablation_resolution"]:
        print("\n## Resolution ablation (P1 crop vs crop downscaled x4)\n")
        for r in s["ablation_resolution"]:
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    if s.get("molmo2_pointing"):
        print("\n## Molmo2 pointing diagnostics\n")
        print(s["molmo2_pointing"])
    if s["stageB"]:
        print("\n## Stage B: 150 rectified crops (P1)\n")
        for r in s["stageB"]:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nAPI calls logged: {s['n_api_calls']}, total cost ${s['cost_total_usd']:.2f}")


if __name__ == "__main__":
    main()

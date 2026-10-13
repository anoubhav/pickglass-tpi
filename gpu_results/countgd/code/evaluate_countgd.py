"""Evaluate the CountGD baseline and export results in the VLM harness' CSV schema.

Reads outputs/raw/countgd__{legacy,rect150}__shard*.jsonl (run_countgd.py), writes
outputs/results/:
  legacy_predictions.csv            same columns as vlm/results/legacy_predictions.csv
  stageA_legacy_metrics.csv         same columns as vlm/results/stageA_legacy_metrics.csv
  rect150_predictions_long.csv      same columns as vlm/results/rect150_P1_predictions_long.csv
                                    (model = 'countgd' for the primary configuration,
                                     'countgd@<method>' for every other one, the VLM
                                     evaluator's convention for tagged extra runs)
  rect150_predictions_wide.csv      image_id, native_crop_px, <model>__v/__h/__weave
  rect150_summary.csv               same columns as vlm/results/rect150_P1_summary.csv
  countgd_degeneracy.csv            zero-count rate, count / periodicity ratio, box shapes
  countgd_whole_crop_cells.csv      whole-crop cell counts vs GT ends x picks (legacy GT)
  countgd_threshold_sensitivity.csv MAE vs score threshold (diagnostic, not used to pick)
  countgd_bootstrap_ci.csv          95% bootstrap CI (13 images resampled) of MAE (best) and of
                                    the paired MAE difference to the periodicity-only row
  countgd_runinfo.json, summary.json

Methods
  PER                      label-free periodicity alone (the exemplar generator), S/p
  CG-{A,B}-{yarn,thread}-{tx,ex3}
                           strip counts, pre-registered rule: TPI = number of detections
                           (score > 0.23) owned in the 1-inch strip
  CG-W-{tx,ex3}-lat        whole-crop cell detections -> sqrt-free lattice TPI
  <strip method>@band      POST HOC (added after the 2-image smoke test showed that CountGD
                           detects interlacing cells, ~2-3 rows of them per strip, rather than
                           yarns): count only detections whose centre lies within +-p_perp/2 of
                           the strip centre line.  A band one perpendicular spacing wide
                           contains exactly one cell per crossing yarn, so count = TPI with no
                           fitted factor.
PRIMARY (fixed before the full runs): CG-B-yarn-ex3 — CountGD's official adaptive-crop rule,
text + 3 automatic exemplars, pre-registered count rule.

Metrics on the 13 usable legacy GT images use the VLM evaluator's own `metrics`/`errors`
functions (axis-matched: vertical<->warp, horizontal<->weft; best-assignment: sorted pairs).
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
import cg_lib as L          # noqa: E402

ROOT = L.ROOT
RAW = os.path.join(ROOT, "outputs", "raw")
OUT = os.path.join(ROOT, "outputs", "results")
PRIMARY = "CG-B-yarn-ex3"
THR = 0.23
EV = L.eval_fns()
metrics, loo_mean_baseline, write_csv = EV["metrics"], EV["loo_mean_baseline"], EV["write_csv"]

DESC = {
    "PER": "periodicity only (exemplar generator, no CountGD)",
    "A": "whole strip", "B": "tiled strip (official crop rule)",
    "tx": "text only", "ex3": "text + 3 auto exemplars",
}


def label(m):
    if m == "PER":
        return "Periodicity only (exemplar generator)"
    base, _, rule = m.partition("@")
    p = base.split("-")
    if p[1] == "W":
        s = f"CountGD whole-crop cells, {DESC[p[2]]}, lattice TPI"
    else:
        s = f"CountGD {DESC[p[1]]}, '{p[2]}', {DESC[p[3]]}"
    return s + (" [post hoc: centre-band rule]" if rule == "band" else "")


def load(set_name):
    recs = {}
    for f in sorted(glob.glob(os.path.join(RAW, f"countgd__{set_name}__shard*of*.jsonl"))):
        if "__dev" in f:
            continue
        for line in open(f):
            r = json.loads(line)
            r["_file"] = os.path.basename(f)
            k = (r["image_id"], r["method"], r["part"])
            if k not in recs or (recs[k].get("error") and not r.get("error")):
                recs[k] = r
    return recs


def band_count(r, thr=THR):
    """Detections owned, above threshold and centred within +-p_perp/2 of the strip centre."""
    S = r["side_px"]
    c = S / 2.0
    p_perp = r["p_y"] if r["part"] == "h" else r["p_x"]
    n = 0
    for d in r["detections"]:
        if d[4] > thr and d[5]:
            cc = (d[1] + d[3]) / 2 if r["part"] == "h" else (d[0] + d[2]) / 2
            n += abs(cc - c) <= p_perp / 2
    return n


def count_at(r, thr):
    return sum(1 for d in r["detections"] if d[4] > thr and d[5])


def estimates(recs, thr=THR):
    """-> {(method, image): dict(v, h, ok, sec, rec_h, rec_v)}"""
    by = defaultdict(dict)
    for (img, m, part), r in recs.items():
        by[img][(m, part)] = r
    out = {}
    for img, d in by.items():
        r = d.get(("PER", "both"))
        if r:
            out[("PER", img)] = dict(v=r["count_v"], h=r["count_h"], ok=True, sec=r["seconds"], recs=[r])
        methods = {m for (m, part) in d if m != "PER"}
        for m in methods:
            if m.startswith("CG-W"):
                r = d.get((m, "whole"))
                if not r or thr != THR:
                    continue
                # a CountGD exception is a FAILURE of the method (ok=False), not a missing run
                ok = not r.get("error") and r.get("lat_ends") is not None and r.get("lat_picks") is not None
                out[(m + "-lat", img)] = dict(v=r.get("lat_ends"), h=r.get("lat_picks"), ok=ok,
                                              sec=r["seconds"], recs=[r])
                continue
            rh, rv = d.get((m, "h")), d.get((m, "v"))
            if not rh or not rv:
                continue
            per_s = d[("PER", "both")]["seconds"] if ("PER", "both") in d else 0.0
            sec = rh["seconds"] + rv["seconds"] - per_s          # periodicity counted once
            if rh.get("error") or rv.get("error"):
                # CountGD raised (mode A on very thin strips: fewer encoder tokens than the 900
                # queries -> 'selected index k out of range').  Counted as a method failure
                # (parse_fail_gt / parse_ok=False), exactly like an unparseable VLM answer.
                for mm in (m, m + "@band"):
                    out[(mm, img)] = dict(v=None, h=None, ok=False, sec=sec, recs=[rh, rv])
                continue
            out[(m, img)] = dict(v=float(count_at(rh, thr)), h=float(count_at(rv, thr)), ok=True,
                                 sec=sec, recs=[rh, rv])
            out[(m + "@band", img)] = dict(v=float(band_count(rh, thr)), h=float(band_count(rv, thr)),
                                           ok=True, sec=sec, recs=[rh, rv])
    return out


def method_order(ms):
    def key(m):
        base, _, rule = m.partition("@")
        return (0 if m == PRIMARY else 1 if base == PRIMARY else 2 if m == "PER" else 3,
                "W" in base, base, rule)
    return sorted(ms, key=key)


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = L.legacy_gt()
    leg = load("legacy")
    r150 = load("rect150")
    est_l = estimates(leg)
    est_r = estimates(r150)
    meths = method_order({m for (m, _) in est_l})
    summary = dict(primary=PRIMARY, n_gt=len(gt), gt_images=sorted(gt), threshold=THR)

    # ------------------------------------------------------------ legacy, per image
    per_img = []
    for m in meths:
        for img in L.LEGACY_NAMES:
            e = est_l.get((m, img))
            if e is None:
                continue
            g = gt.get(img)
            per_img.append(dict(model="countgd", set="legacy", repeat=0, method=m, image=img,
                                vertical=e["v"], horizontal=e["h"], weave=None, ok=e["ok"],
                                gt_warp=g["warp"] if g else None, gt_weft=g["weft"] if g else None,
                                gt_weave=g["weave_name"] if g else None, seconds=e["sec"], usd=0.0))
    write_csv(os.path.join(OUT, "legacy_predictions.csv"), per_img)

    # ------------------------------------------------------------ Stage A metrics
    rows_out = []
    gt_plain = {k: g for k, g in gt.items() if g["weave_name"] == "plain"}
    for m in meths:
        rows = {img: e for (mm, img), e in est_l.items() if mm == m}
        ax, be = metrics(rows, gt, "axis"), metrics(rows, gt, "best")
        bp, ap = metrics(rows, gt_plain, "best"), metrics(rows, gt_plain, "axis")
        gt_rows = [rows[i] for i in gt if i in rows]
        all20 = [rows[i] for i in L.LEGACY_NAMES if i in rows]
        secs = [r["sec"] for r in gt_rows if r["sec"] is not None and not math.isnan(r["sec"])]
        rows_out.append(dict(
            model="countgd", label=label(m), method=m, n_gt=len(gt),
            n_ok=ax.get("n_ok", 0), parse_fail_gt=ax.get("n_fail"), not_run_gt=ax.get("n_missing", 0),
            parse_fail_all20=sum(1 for r in all20 if not r["ok"]), n_run_all20=len(all20),
            mae_axis=ax.get("mae"), mape_axis=ax.get("mape"), within10_axis=ax.get("within10"),
            bias_axis=ax.get("bias"), mae_warp=ax.get("mae_warp"), mae_weft=ax.get("mae_weft"),
            mae_best=be.get("mae"), mape_best=be.get("mape"), within10_best=be.get("within10"),
            bias_best=be.get("bias"), medae_best=be.get("medae"), n_values=be.get("n_values"),
            weave_correct=0, weave_n=0,
            n_plain=len(gt_plain), n_ok_plain=bp.get("n_ok", 0), mae_best_plain=bp.get("mae"),
            mape_best_plain=bp.get("mape"), within10_best_plain=bp.get("within10"),
            mae_axis_plain=ap.get("mae"), within10_axis_plain=ap.get("within10"),
            sec_per_img=float(np.mean(secs)) if secs else None, usd_per_img=0.0))
    write_csv(os.path.join(OUT, "stageA_legacy_metrics.csv"), rows_out)
    summary["stageA"] = rows_out
    bl_ax, bl_be = loo_mean_baseline(gt)
    summary["loo_mean_baseline"] = dict(axis=bl_ax, best=bl_be)

    # ------------------------------------------------------------ bootstrap CIs (image-level resampling)
    rng = np.random.default_rng(0)
    keys = sorted(gt)
    B = 10000
    idx = rng.integers(0, len(keys), size=(B, len(keys)))

    def per_image_abs(m):
        out = {}
        for img in keys:
            e = est_l.get((m, img))
            if e is None or not e["ok"]:
                continue
            p, t = sorted([e["v"], e["h"]]), sorted([gt[img]["warp"], gt[img]["weft"]])
            out[img] = np.mean([abs(a - b) for a, b in zip(p, t)])
        return out

    ci_rows = []
    base = per_image_abs("PER")
    for m in [PRIMARY, PRIMARY + "@band", "PER", "CG-W-ex3-lat", "CG-W-tx-lat"]:
        a = per_image_abs(m)
        common = [k for k in keys if k in a]
        arr = np.array([a.get(k, np.nan) for k in keys])
        boots = np.nanmean(arr[idx], axis=1)
        row = dict(method=m, n_images=len(common), mae_best=float(np.nanmean(arr)),
                   ci95_lo=float(np.nanpercentile(boots, 2.5)), ci95_hi=float(np.nanpercentile(boots, 97.5)))
        if m != "PER":
            both = [k for k in keys if k in a and k in base]
            d = np.array([a[k] - base[k] if k in both else np.nan for k in keys])
            bd = np.nanmean(d[idx], axis=1)
            row.update(mae_diff_vs_PER=float(np.nanmean(d)), diff_ci95_lo=float(np.nanpercentile(bd, 2.5)),
                       diff_ci95_hi=float(np.nanpercentile(bd, 97.5)), n_paired=len(both))
        ci_rows.append(row)
    write_csv(os.path.join(OUT, "countgd_bootstrap_ci.csv"), ci_rows)
    summary["bootstrap_ci"] = ci_rows

    # ------------------------------------------------------------ threshold sensitivity (diagnostic)
    sens = []
    for thr in (0.15, 0.20, 0.23, 0.30, 0.40, 0.50):
        e_t = estimates(leg, thr)
        for m in [PRIMARY, PRIMARY + "@band", "CG-B-thread-ex3", "CG-B-thread-ex3@band",
                  "CG-A-yarn-ex3", "CG-A-yarn-ex3@band"]:
            rows = {img: e for (mm, img), e in e_t.items() if mm == m}
            be = metrics(rows, gt, "best")
            ax = metrics(rows, gt, "axis")
            sens.append(dict(method=m, threshold=thr, mae_best=be.get("mae"), mape_best=be.get("mape"),
                             within10_best=be.get("within10"), bias_best=be.get("bias"),
                             mae_axis=ax.get("mae"), n_values=be.get("n_values")))
    write_csv(os.path.join(OUT, "countgd_threshold_sensitivity.csv"), sens)

    # ------------------------------------------------------------ whole-crop cells vs GT
    wrows = []
    for img, g in gt.items():
        for m in ("CG-W-tx", "CG-W-ex3"):
            r = leg.get((img, m, "whole"))
            if not r or r.get("error"):
                continue
            exp = g["warp"] * g["weft"]
            wrows.append(dict(image=img, method=m, gt_weave=g["weave_name"], gt_warp=g["warp"], gt_weft=g["weft"],
                              n_cells=r["count"], gt_ends_x_picks=exp, ratio_to_gt=r["count"] / exp,
                              lat_ends=r.get("lat_ends"), lat_picks=r.get("lat_picks"),
                              sqrt_count_diagnostic=r.get("sqrt_count"),
                              per_v=r["side_px"] / r["p_x"], per_h=r["side_px"] / r["p_y"],
                              n_tiles=r["n_tiles"], seconds=r["seconds"]))
    write_csv(os.path.join(OUT, "countgd_whole_crop_cells.csv"), wrows)
    summary["whole_crop_cells"] = {
        m: dict(median_ratio=float(np.median([w["ratio_to_gt"] for w in wrows if w["method"] == m])),
                median_ratio_plain=float(np.median([w["ratio_to_gt"] for w in wrows
                                                    if w["method"] == m and w["gt_weave"] == "plain"])),
                n=sum(w["method"] == m for w in wrows))
        for m in ("CG-W-tx", "CG-W-ex3") if any(w["method"] == m for w in wrows)}

    # ------------------------------------------------------------ degeneracy diagnostics
    deg = []
    for set_name, recs, est in (("legacy", leg, est_l), ("rect150", r150, est_r)):
        for m in method_order({mm for (mm, _) in est}):
            if m == "PER":
                continue
            E = [e for (mm, _), e in est.items() if mm == m]
            if not E:
                continue
            vals = np.array([[e["v"], e["h"]] for e in E if e["ok"]], float)
            pers = np.array([[est[("PER", img)]["v"], est[("PER", img)]["h"]]
                             for (mm, img), e in est.items() if mm == m and e["ok"]], float)
            ratio = vals / pers if len(vals) else np.zeros((0, 2))
            boxes = []
            if not m.startswith("CG-W"):
                for e in E:
                    if not e["ok"]:
                        continue
                    for r in e["recs"]:
                        for d in r["detections"]:
                            if d[4] > THR and d[5]:
                                w, h = d[2] - d[0], d[3] - d[1]
                                pa = r["p_x"] if r["part"] == "h" else r["p_y"]
                                along, across = (w, h) if r["part"] == "h" else (h, w)
                                boxes.append((along / pa, across / max(r["strip_thick"], 1)))
            boxes = np.array(boxes) if boxes else np.zeros((0, 2))
            deg.append(dict(set=set_name, method=m, n_images=len(E), n_ok=int(len(vals)),
                            frac_zero_counts=float(np.mean(vals == 0)) if len(vals) else None,
                            median_count_over_periodicity=float(np.median(ratio)) if len(ratio) else None,
                            iqr_count_over_periodicity=float(np.subtract(*np.percentile(ratio, [75, 25])))
                            if len(ratio) else None,
                            median_box_along_in_spacings=float(np.median(boxes[:, 0])) if len(boxes) else None,
                            median_box_across_in_strip_thickness=float(np.median(boxes[:, 1])) if len(boxes) else None,
                            sec_per_img=float(np.mean([e["sec"] for e in E]))))
    write_csv(os.path.join(OUT, "countgd_degeneracy.csv"), deg)
    summary["degeneracy"] = deg

    # ------------------------------------------------------------ 150 crops
    manifest = json.load(open(os.path.join(L.DATA, "manifest.json")))
    side = {str(im["id"]): im["rect_size"][0] for im in manifest["images"]}
    longp, wide = [], defaultdict(dict)
    for m in method_order({mm for (mm, _) in est_r}):
        tag = "countgd" if m == PRIMARY else f"countgd@{m}"
        for img in sorted({i for (mm, i) in est_r if mm == m}, key=int):
            e = est_r[(m, img)]
            r = e["recs"][0]
            longp.append(dict(model=tag, image_id=int(img), native_crop_px=side.get(img), vertical=e["v"],
                              horizontal=e["h"], weave=None, parse_ok=e["ok"], seconds=e["sec"], usd=0.0,
                              finish_reason="error" if any(x.get("error") for x in e["recs"]) else "ok",
                              provider=f"local {r['host']} GPU{r['gpu']} ({r['gpu_name']})",
                              model_repo=r["repo"], revision=r["revision"],
                              raw_file_key=f"raw/{r['_file']}#{img}"))
            wide[int(img)][f"{tag}__v"] = e["v"]
            wide[int(img)][f"{tag}__h"] = e["h"]
            wide[int(img)][f"{tag}__weave"] = None
    write_csv(os.path.join(OUT, "rect150_predictions_long.csv"), longp)
    write_csv(os.path.join(OUT, "rect150_predictions_wide.csv"),
              [dict(image_id=i, native_crop_px=side.get(str(i)), **wide[i]) for i in sorted(wide)])
    b_sum = []
    for tag in dict.fromkeys(r["model"] for r in longp):
        Lr = [r for r in longp if r["model"] == tag]
        ok = [r for r in Lr if r["parse_ok"]]
        vals = np.array([[r["vertical"], r["horizontal"]] for r in ok], float) if ok else np.zeros((0, 2))
        small = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] < 1000]
        big = [np.mean([r["vertical"], r["horizontal"]]) for r in ok if r["native_crop_px"] >= 1000]
        secs = [r["seconds"] for r in Lr if r["seconds"] is not None]
        b_sum.append(dict(model=tag, n_run=len(Lr), n_parsed=len(ok),
                          median_v=float(np.median(vals[:, 0])) if len(vals) else None,
                          median_h=float(np.median(vals[:, 1])) if len(vals) else None,
                          p05=float(np.percentile(vals, 5)) if len(vals) else None,
                          p95=float(np.percentile(vals, 95)) if len(vals) else None,
                          mean_tpi_small_crops=float(np.mean(small)) if small else None, n_small=len(small),
                          mean_tpi_large_crops=float(np.mean(big)) if big else None, n_large=len(big),
                          sec_per_img=float(np.mean(secs)) if secs else None, usd_total=0.0, usd_per_img=0.0))
    write_csv(os.path.join(OUT, "rect150_summary.csv"), b_sum)
    summary["stageB"] = b_sum

    # ------------------------------------------------------------ run info
    info = defaultdict(lambda: dict(n=0, sec=0.0))
    gpus, names = set(), set()
    for recs in (leg, r150):
        for r in recs.values():
            gpus.add(str(r["gpu"]))
            names.add(r["gpu_name"])
    per_img_sec = {}
    for set_name, recs in (("legacy", leg), ("rect150", r150)):
        tot = defaultdict(float)
        for r in recs.values():
            tot[r["image_id"]] += r["seconds"]
        per_img_sec[set_name] = dict(n_images=len(tot), mean_total_sec_all_methods=float(np.mean(list(tot.values())))
                                     if tot else None)
    runinfo = dict(model="CountGD (Swin-B GroundingDINO, FSC-147 checkpoint)", repo="nikigoli/CountGD",
                   revision="6b989ad11db9413e44863ea5c57735cae6cacff8",
                   code="github.com/niki-amini-naieni/CountGD @ b6f362b3f5cd20db4a171faa410dfed8f2f466d8 (models_inference)",
                   gpus_physical_index=sorted(gpus), gpu_names=sorted(names), box_threshold=THR,
                   primary=PRIMARY, per_image_seconds=per_img_sec,
                   n_records=dict(legacy=len(leg), rect150=len(r150)),
                   n_errors=dict(legacy=sum(1 for r in leg.values() if r.get("error")),
                                 rect150=sum(1 for r in r150.values() if r.get("error"))))
    json.dump(runinfo, open(os.path.join(OUT, "countgd_runinfo.json"), "w"), indent=1)
    summary["runinfo"] = runinfo
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print_tables(summary)


def f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def print_tables(s):
    print(f"\n## CountGD, legacy GT images (n = {s['n_gt']}), primary = {s['primary']}\n")
    print("| Method | parsed | MAE best | %err best | ≤10% best | MAE axis | %err axis | ≤10% axis | bias best | "
          "MAE best plain | s/img |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["stageA"]:
        print(f"| {r['method']} | {r['n_ok']}/{r['n_gt']} | {f(r['mae_best'])} | {f(r['mape_best'])} | "
              f"{r['within10_best']}/{r['n_values']} | {f(r['mae_axis'])} | {f(r['mape_axis'])} | "
              f"{r['within10_axis']}/{r['n_values']} | {f(r['bias_best'])} | {f(r['mae_best_plain'])} "
              f"({r['n_ok_plain']}/{r['n_plain']}) | {f(r['sec_per_img'], 2)} |")
    b = s["loo_mean_baseline"]
    print(f"| LOO mean predictor | 13/13 | {f(b['best']['mae'])} | {f(b['best']['mape'])} | {b['best']['within10']}/26 | "
          f"{f(b['axis']['mae'])} | {f(b['axis']['mape'])} | {b['axis']['within10']}/26 | – | – | – |")
    print("\nWhole-crop cells:", s["whole_crop_cells"])
    print("\nDegeneracy:")
    for r in s["degeneracy"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nRect150:")
    for r in s["stageB"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nRun info:", s["runinfo"])


if __name__ == "__main__":
    main()

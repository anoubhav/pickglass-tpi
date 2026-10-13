"""Extra diagnostics (Mac): (1) does the tuned model LOCALISE yarns or emit a pitch-regular lattice?
phase test on synthetic val: median |offset| of matched points (in GT spacings) for the actual
predictions vs the same predictions shifted by half a GT spacing (anti-phase).  A model that localises
yarns has actual << shifted; a lattice with random phase has actual ~ shifted ~ 0.25.
(2) spacing regularity (gap CV) of point lists on synthetic val and legacy; (3) Spearman rho with the
hand counts on legacy-13 per readout; (4) synthetic-val learning curve on the first 24 val crops."""
import json, os, glob
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); FTR = os.path.dirname(HERE)
lab = {str(r["seed"]): r for r in map(json.loads, open(os.path.join(FTR, "data_val_labels.jsonl")))}

def recs(pattern):
    out = {}
    for p in glob.glob(os.path.join(FTR, pattern)):
        for l in open(p):
            r = json.loads(l); out[(r["model"], r["set"], r["image_id"], r["view"])] = r
    return list(out.values())

def coords(r):
    ax = r["view"][-1]
    return np.array(sorted(p[0] if ax == "h" else p[1] for p in r["parsed"].get("points", [])), float)

def match_offsets(pr, gt):
    if len(gt) < 2 or not len(pr): return []
    sp = np.median(np.diff(gt)); tol = 0.5 * sp
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(pr) for j, g in enumerate(gt) if abs(p - g) <= tol)
    up, ug, off = set(), set(), []
    for d, i, j in pairs:
        if i not in up and j not in ug: up.add(i); ug.add(j); off.append(d / sp)
    return off

rows = []
for r in recs("raw/*__synval.jsonl") + recs("raw_diag/diag_step*__synval.jsonl"):
    if not r["prompt_id"].startswith("P4"): continue
    v = "full" if r["view"] in ("guide_h", "guide_v") else r["view"].split("_guide_")[0]
    gt = np.array(lab[r["image_id"]]["views"][v][r["view"][-1]]) * 1000
    pr = coords(r)
    if len(gt) < 3 or len(pr) < 3: continue
    sp = np.median(np.diff(gt))
    a = match_offsets(pr, gt); s = match_offsets(pr + 0.5 * sp, gt)
    rows.append(dict(model=r["model"], scale="full" if v == "full" else "quarter", image=r["image_id"],
                     med_off=np.median(a) if a else np.nan, med_off_antiphase=np.median(s) if s else np.nan,
                     gap_cv=np.std(np.diff(pr)) / np.mean(np.diff(pr)), gt_gap_cv=np.std(np.diff(gt)) / np.mean(np.diff(gt)),
                     exact_regular=bool(np.ptp(np.diff(pr)) <= 1)))
D = pd.DataFrame(rows)
P = D.groupby(["model", "scale"]).agg(n=("image", "size"), med_offset=("med_off", "median"),
        med_offset_antiphase=("med_off_antiphase", "median"), gap_cv=("gap_cv", "median"),
        gt_gap_cv=("gt_gap_cv", "median"), frac_regular_within_1unit=("exact_regular", "mean")).reset_index()
# paired Wilcoxon actual vs anti-phase per model/scale
P["p_wilcoxon_actual_vs_antiphase"] = [stats.wilcoxon(g.med_off - g.med_off_antiphase).pvalue
                                       for _, g in D.dropna().groupby(["model", "scale"])]
print(P.round(3).to_string(index=False))
P.round(4).to_csv(os.path.join(FTR, "results", "synval_phase_test.csv"), index=False)

# legacy gap CV
L = []
for r in recs("raw/*__legacy.jsonl"):
    if r["prompt_id"].startswith("P4"):
        pr = coords(r)
        if len(pr) > 2:
            L.append(dict(model=r["model"], scale="full" if r["view"] in ("guide_h", "guide_v") else "quarter",
                          gap_cv=np.std(np.diff(pr)) / np.mean(np.diff(pr)), regular=bool(np.ptp(np.diff(pr)) <= 1)))
L = pd.DataFrame(L).groupby(["model", "scale"]).agg(n=("gap_cv", "size"), gap_cv_median=("gap_cv", "median"),
                                                     frac_regular_within_1unit=("regular", "mean")).reset_index()
print(L.round(3).to_string(index=False)); L.round(4).to_csv(os.path.join(FTR, "results", "legacy_gap_cv.csv"), index=False)

# Spearman on legacy-13 (axis-matched, 26 counts)
pc = pd.read_csv(os.path.join(FTR, "tables", "legacy13_ft_per_count.csv")); pc = pc[pc["assign"] == "axis"]
S = pc.groupby("key").apply(lambda g: pd.Series(dict(n=len(g), spearman=stats.spearmanr(g.pred, g.true)[0]))).reset_index()
print(S.round(3).to_string(index=False)); S.round(4).to_csv(os.path.join(FTR, "results", "legacy13_spearman.csv"), index=False)

# learning curve on the first 24 val crops (P4, P4q)
from collections import defaultdict
first24 = sorted(lab, key=int)[:24]
lc = []
for r in recs("raw/molmo2_4b_synthft__synval.jsonl") + recs("raw_diag/diag_step*__synval.jsonl"):
    pass
def mape(model, pattern):
    by = defaultdict(dict)
    for r in recs(pattern):
        if r["model"] == model and r["image_id"] in first24: by[r["image_id"]][r["view"]] = r
    out = {}
    for meth, views, mult in (("P4", ("guide_h", "guide_v"), 1), ("P4q", ("qc_guide_h", "qc_guide_v"), 4)):
        e = []
        for i, V in by.items():
            L_ = lab[i]
            for vw, gtn in zip(views, (L_["n_h_full"], L_["n_v_full"])):
                if vw in V: e.append(100 * abs(mult * V[vw]["parsed"]["n_points"] - gtn) / gtn)
        out[meth] = (np.mean(e), len(e))
    return out
for step, model, pat in ((0, "molmo2_4b_base", "raw/molmo2_4b_base__synval.jsonl"), (300, "diag_step300", "raw_diag/diag_step300__synval.jsonl"),
                         (500, "diag_step500", "raw_diag/diag_step500__synval.jsonl"), (1000, "molmo2_4b_synthft", "raw/molmo2_4b_synthft__synval.jsonl")):
    m = mape(model, pat); lc.append(dict(step=step, P4=m["P4"][0], P4q=m["P4q"][0], n_counts=m["P4"][1]))
LC = pd.DataFrame(lc); print(LC.round(2).to_string(index=False)); LC.round(3).to_csv(os.path.join(FTR, "results", "synval24_learning_curve.csv"), index=False)

"""iqa_generic.py -- do GENERIC no-reference IQA scores predict counting error? (label-safe parts)

WVAQ is an image-quality workshop, so a reviewer will ask whether an off-the-shelf no-reference IQA
model (BRISQUE, NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR; pyiqa 0.1.16) would do the job of the counter's own,
task-specific quality signals (its confidence, the resolvability ratio eta and the frozen cascade gate).
This script uses only data that already exist; it never reads a human count of the 76 scenes
(tools/iqa150.py does that on label day).

Per-image IQA scores (written by tools/iqa_scores.py, pyiqa, CPU):
  tables/iqa_legacy13_scores.csv   247 degraded legacy crops (13 x clean + 18 levels; the byte-identical
                                   PNGs of methods/degrade_vlmcrops.py, md5-checked against
                                   tables/degradation_legacy13_per_image.csv) + 130 motion/noise/shadow
                                   crops (methods/degrade_vlmcrops_extra.py)
  tables/iqa_genai_scores.csv      genai_sr restorations: 768-px JPEG viewing copies of every output (the
                                   full-resolution PNGs are on gpu-server), the clean references through the
                                   same viewing transform, the unrestored LR inputs, and the full-resolution
                                   PNGs that are on the Mac (1024^2 protocol, to validate the viewing copies)
  tables/iqa150_scores.csv         the 150 rectified crops (label-free)

Protocols: 'work' = the counter's working image (INTER_AREA to max side 1024, never up-sampled; PRIMARY:
what a no-reference gate in front of the counter sees), 'fix1024' = every image resized to 1024^2
(INTER_CUBIC when smaller; = genai_sr/server/iq_metrics.py for images >= 1024 px).  The two differ only
for the down-scaled legacy crops and the 85 quarter-resolution real crops.

Quality orientation: q = -BRISQUE, -NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR (higher = better), so a useful
predictor has rho < 0 against error and AUROC > 0.5 for flagging failures.

Status: BRISQUE and NIQE (protocol 'work' primary) are the pre-specified generic metrics; MUSIQ and CLIP-IQA had
already been seen in a pilot, so they, TOPIQ-NR, the within-fabric and per-family breakdowns ("when") and the
whole generative part are EXPLORATORY (column 'status').  The pre-specified 76-scene test is tools/iqa150.py.

A. legacy-13 degradation set (hand counts): Spearman(q, per-crop mean APE of the two counts, axis
   assignment as in the paper) with a crop-clustered bootstrap CI (13 fabrics = 13 clusters, B = 5000, seed
   20261011), AUROC of -q for "a count off by > 5 %" (and > 10 %), paired differences to the confidence of
   ours on the same resamples, and failures caught when each score escalates as many crops as the frozen gate.
   Targets: each method's error and "any method within 5 %" (a crop fails only if no listed method has both
   counts within 5 %; error = the best method's), so ours' confidence is not judged only by ours' own error.
   Scopes: all 19 conditions (247 crops; ours, log-Gabor, CNN, any of the three), clean + the 8 VLM conditions
   (117 crops; + GPT-6.1-sol P1/P3, any of ours/log-Gabor/CNN/GPT P1; the cascade's own evaluation set) and,
   ours / log-Gabor only, 29 conditions (+ motion, noise, shadow; 377 crops).
   A_by_family: the same per degradation family (13 clean crops + the family's levels).
   A_within_fabric: stratified AUROC over pairs of conditions of the SAME crop (and the mean within-crop
   Spearman), so a signal cannot score by recognising intrinsically hard fabrics; families with < 5 fabrics
   that have both outcomes are not reported.
B. generative enhancement (genai_sr): per set x condition x restorer, mean IQA and the paired change
   against bicubic and against the clean reference (scene-clustered bootstrap, B = 5000), next to the published
   count verdicts (genai_sr/results/genai_hallucination_table.csv), and the per-image rank correlation / AUROC
   of IQA and of ours' confidence with count error over the generative outputs, scene-clustered (R-LEG: 13
   fabrics, 2019 hand counts; R-HR: 53 crops from 40 PickGlass scenes, reference = each counter's own count on
   the clean crop), pooled and per set, for ours / CNN / log-Gabor / any of the three, plus GPT-6.1-sol P1 on
   the 25 R-LEG x8 SeedVR2 and Nano Banana Pro outputs it was run on.
C. label-free context on the 150 real crops: IQA vs ours' confidence and vs the frozen gate.

Outputs (--out, default tables/): iqa_legacy13.csv, iqa_genai.csv, iqa_summary.json, supp_iqa.tex (the supplement
table; NOT \\input anywhere by this script).  The raw pyiqa outputs behind the three score tables are kept durably
in handover/results/round2_tmp_copy/generic-iqa/out/ (tools/iqa_scores.py --finalize rebuilds the tables).

    .tools/venv/bin/python -B tools/iqa_generic.py   # numpy, scipy, pandas only; ~1 min
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TAB = os.path.join(PAPER, "tables")
SCORES = TAB          # where the per-image IQA tables are read (--scores-dir); outputs go to --out (default TAB)
GSR = os.path.join(ROOT, "genai_sr")

B = 5000
SEED = 20261011
METRICS = {"brisque": ("BRISQUE", -1), "niqe": ("NIQE", -1), "musiq": ("MUSIQ", 1), "clipiqa": ("CLIP-IQA", 1),
           "topiq_nr": ("TOPIQ-NR", 1)}
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
VLM_CONDS = {("gblur", 3.0), ("gblur", 6.0), ("down", 4.0), ("down", 8.0), ("jpeg", 20.0),
             ("glare", 0.05), ("glare", 0.3), ("corners", 0.05)}
GENERATIVE = ["realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
RESTORERS = ["bicubic", "swinir"] + GENERATIVE
ANY3 = "any method (ours, log-Gabor, CNN)"
ANY4 = "any method (ours, log-Gabor, CNN, GPT P1)"
ANY2 = "any method (ours, log-Gabor)"
# Pre-specification (handover_check.json, research critic, Oct 11): the pilot had already scored MUSIQ and CLIP-IQA
# on the legacy fabrics, so only BRISQUE and NIQE (protocol 'work' primary) and the 76-scene analysis
# (tools/iqa150.py) may be called pre-specified.  Every other metric, target, scope and breakdown is exploratory.
PRESPEC = ("brisque", "niqe")
FAMILIES = {"gblur": "defocus blur", "down": "down-sampling", "glare": "glare", "corners": "corner error",
            "jpeg": "JPEG", "motion": "motion blur", "noise": "noise", "shadow": "shadow"}
REST_NAME = {"bicubic": "Bicubic", "swinir": "SwinIR", "realesrgan": "Real-ESRGAN", "seedvr2": "SeedVR2",
             "invsr": "InvSR", "flux2klein": "FLUX.2 [klein]", "nbpro": "Nano Banana Pro", "none": "LR (none)",
             "clean": "clean reference"}


# ============================================================================ statistics
def rankdata(x):
    return stats.rankdata(x)


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auroc(score, pos):
    """P(score of a positive > score of a negative), ties 1/2.  score = 'badness' (= -q)."""
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_weights(groups, rng, b=B):
    """Cluster bootstrap as a (b, n) matrix of multiplicities: each resample draws len(clusters) clusters
    with replacement; an item's weight = how often its cluster was drawn."""
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    pick = rng.integers(0, len(ug), (b, len(ug)))
    cnt = np.zeros((b, len(ug)))
    np.add.at(cnt, (np.repeat(np.arange(b), len(ug)), pick.ravel()), 1)
    return cnt[:, inv], len(ug)


def wranks(x, W):
    """Mid-ranks of x in every weighted resample (row of W) = the ranks scipy would give the expanded
    resample (each item repeated W times)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    start = np.r_[True, xs[1:] != xs[:-1]]
    gid = np.cumsum(start) - 1
    Wg = np.add.reduceat(W[:, order], np.where(start)[0], axis=1)        # weight per tie group
    rank_g = np.cumsum(Wg, axis=1) - Wg + (Wg + 1) / 2
    R = np.empty_like(W)
    R[:, order] = rank_g[:, gid]
    return R


def boot_spearman(x, y, W):
    rx, ry = wranks(x, W), wranks(y, W)
    sw = W.sum(1, keepdims=True)
    mx, my = (W * rx).sum(1, keepdims=True) / sw, (W * ry).sum(1, keepdims=True) / sw
    cov = (W * (rx - mx) * (ry - my)).sum(1)
    vx, vy = (W * (rx - mx) ** 2).sum(1), (W * (ry - my) ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def boot_auroc(score, pos, W):
    pos = np.asarray(pos, bool)
    R = wranks(score, W)
    n1, n0 = (W * pos).sum(1), (W * ~pos).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W * R * pos).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_all(qs, err, fail, W):
    """{name: (bootstrap rho array, bootstrap AUROC array)} on the same resamples (paired)."""
    return {name: (boot_spearman(q, err, W), boot_auroc(-q, fail, W)) for name, q in qs.items()}


def ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def quality(df, metric, protocol):
    col = f"{metric}_{protocol}" if protocol else metric
    return METRICS[metric][1] * df[col].astype(float)


def any_target(tgt, names):
    """'Any method within 5 %': a crop fails only if NO listed method has both counts within 5 % (resp. 10 %);
    its error is the best method's mean APE.  A method without a valid count on a crop is skipped there."""
    X = pd.concat([tgt[n] for n in names])
    g = X.groupby(level=list(X.index.names))
    return pd.DataFrame({"ape": g.ape.min(), "fail5": g.fail5.all().astype(bool), "fail10": g.fail10.all().astype(bool)})


def status_of(kind, prot):
    if kind in ("conf", "eta", "gate"):
        return "task signal (reference)"
    if kind in PRESPEC:
        return "pre-specified" if prot == "work" else "pre-specified metric, secondary protocol"
    return "exploratory"


# ============================================================================ A. legacy-13
def rho_of(h, v, sharp, n_work):
    """= tools/cascade_tradeoff.py rho_of (resolvability ratio eta)."""
    h, v, sharp, n_work = (np.asarray(x, float) for x in (h, v, sharp, n_work))
    f = np.fmax(h, v) / n_work
    gain = (4 * np.sin(np.pi * np.clip(f, 1e-6, 0.5)) ** 2) ** 2
    r = sharp / gain
    return np.where(np.isfinite(r), r, 0.0)


def load_legacy():
    sc = pd.read_csv(os.path.join(SCORES, "iqa_legacy13_scores.csv"))
    sc["level"] = sc.level.astype(float).round(4)
    key = ["image", "kind", "level"]
    # wide per crop: <metric>_<protocol>; 'work' = fix1024 for crops >= 1024 px (identical input)
    wide = None
    for prot in ("fix1024", "work"):
        s = sc[sc.protocol == prot].set_index(key)[list(METRICS) + ["md5", "src_h", "src_w", "in_h", "in_w", "set"]]
        s = s.rename(columns={m: f"{m}_{prot}" for m in METRICS} | {"in_h": f"in_h_{prot}", "in_w": f"in_w_{prot}"})
        if wide is None:
            wide = s
        else:
            wide = wide.join(s[[c for c in s.columns if c.endswith(prot)]], how="left")
    for m in METRICS:
        wide[f"{m}_work"] = wide[f"{m}_work"].fillna(wide[f"{m}_fix1024"])
    wide["work_is_fix1024"] = (np.fmax(wide.src_h, wide.src_w) >= 1024).astype(int)
    wide = wide.reset_index()

    # errors (paper's per-image table: byte-identical inputs, md5 checked)
    d = pd.read_csv(os.path.join(TAB, "degradation_legacy13_per_image.csv"))
    d = d[d.run == "main"].copy()
    d["level"] = d.level.astype(float).round(4)
    ours = d[d.method == "ours"].set_index(key)
    chk = wide[wide.set == "legacy13"].set_index(key)
    bad = (chk.md5 != ours.md5.reindex(chk.index)).sum()
    if bad:
        raise SystemExit(f"{bad} legacy crops differ from the md5 in degradation_legacy13_per_image.csv")
    tgt = {}
    for m, name in [("ours", "ours"), ("log-Gabor", "log-Gabor"), ("CNN", "CNN"), ("gpt-6.1-sol P1", "GPT-6.1-sol P1"),
                    ("gpt-6.1-sol P3", "GPT-6.1-sol P3")]:
        g = d[(d.method == m) & d.ok.astype(bool)].set_index(key)
        tgt[name] = pd.DataFrame({"ape": 0.5 * (g.ape_warp + g.ape_weft),
                                  "fail5": np.fmax(g.ape_warp, g.ape_weft) > 5,
                                  "fail10": np.fmax(g.ape_warp, g.ape_weft) > 10})
    # the 130 extra crops (motion / noise / shadow): ours + log-Gabor from the extra results file
    x = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_degrade_extra.csv"))
    x["level"] = x.level.astype(float).round(4)
    chk_x = wide[wide.set == "legacy13_extra"].set_index(key)
    xo = x[x.estimator == "ours"].set_index(key)
    bad = (chk_x.md5 != xo.md5.reindex(chk_x.index)).sum()
    if bad:
        raise SystemExit(f"{bad} extra crops differ from the md5 in legacy13_vlmcrops_degrade_extra.csv")
    for est, name in [("ours", "ours"), ("gabor", "log-Gabor")]:
        g = x[x.estimator == est].set_index(key)
        aw, af = 100 * (g.h - g.gt_warp).abs() / g.gt_warp, 100 * (g.v - g.gt_weft).abs() / g.gt_weft
        ok = np.isfinite(g.h) & np.isfinite(g.v)
        e = pd.DataFrame({"ape": 0.5 * (aw + af), "fail5": np.fmax(aw, af) > 5, "fail10": np.fmax(aw, af) > 10})[ok]
        tgt[name] = pd.concat([tgt[name], e])
    tgt[ANY3] = any_target(tgt, ["ours", "log-Gabor", "CNN"])
    tgt[ANY4] = any_target(tgt, ["ours", "log-Gabor", "CNN", "GPT-6.1-sol P1"])
    tgt[ANY2] = any_target(tgt, ["ours", "log-Gabor"])
    # task-specific signals: ours' confidence, eta, the frozen gate (cascade_thresholds.json, fits.vlm9_a10)
    m = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_degrade.csv"))
    m = pd.concat([m, x])
    m["level"] = m.level.astype(float).round(4)
    mo = m[m.estimator == "ours"].set_index(key)
    sig = pd.DataFrame({"conf": mo.conf.fillna(0.0),
                        "eta": rho_of(mo.h, mo.v, mo.sharpness, np.minimum(mo.S_deg, 1024))}, index=mo.index)
    th = json.load(open(os.path.join(TAB, "cascade_thresholds.json")))["fits"]["vlm9_a10"]["gate"]
    sig["gate_escalate"] = (sig.conf < th["t_c"]) | (sig.eta < th["r_min"])
    # check against the cascade's own inputs (117 crops)
    L = pd.read_csv(os.path.join(TAB, "cascade_legacy13_inputs.csv"))
    L["level"] = L.level.astype(float).round(4)
    L = L.set_index(key)
    assert np.allclose(sig.eta.reindex(L.index), L.rho, rtol=1e-3, atol=1e-5), "eta differs from cascade inputs"
    assert (sig.gate_escalate.reindex(L.index) == ((L.conf < th["t_c"]) | (L.rho < th["r_min"]))).all()
    assert np.allclose(sig.conf.reindex(L.index), L.conf, atol=1e-4)
    wide = wide.set_index(key).join(sig, how="left").reset_index()
    wide["vlm_cond"] = [k == "clean" or (k, lv) in VLM_CONDS for k, lv in zip(wide.kind, wide.level)]
    return wide, tgt, th


def predictors():
    P = [("ours confidence", "conf", None), ("resolvability eta", "eta", None),
         ("cascade gate (frozen; 1 = kept offline)", "gate", None)]
    for prot in ("work", "fix1024"):
        for m in METRICS:
            P.append((f"{METRICS[m][0]} [{prot}]", m, prot))
    return P


def pred_q(J, kind, prot):
    if kind == "conf":
        return J.conf.to_numpy(float)
    if kind == "eta":
        return J.eta.to_numpy(float)
    if kind == "gate":
        return (~J.gate_escalate.astype(bool)).to_numpy(float)
    return quality(J, kind, prot).to_numpy(float)


def part_a(wide, tgt):
    rng = np.random.default_rng(SEED)
    key = ["image", "kind", "level"]
    scopes = {"all 19 conditions (247 crops)": lambda w: w[w.set == "legacy13"],
              "clean + 8 VLM conditions (117 crops)": lambda w: w[(w.set == "legacy13") & w.vlm_cond],
              "29 conditions incl. motion/noise/shadow (377 crops)": lambda w: w}
    targets = {"all 19 conditions (247 crops)": ["ours", "log-Gabor", "CNN", ANY3],
               "clean + 8 VLM conditions (117 crops)": ["ours", "log-Gabor", "CNN", "GPT-6.1-sol P1", "GPT-6.1-sol P3",
                                                        ANY4],
               "29 conditions incl. motion/noise/shadow (377 crops)": ["ours", "log-Gabor", ANY2]}
    rows = []
    for sname, sf in scopes.items():
        Wd = sf(wide)
        for tname in targets[sname]:
            J = Wd.set_index(key).join(tgt[tname], how="inner").reset_index()
            J = J.sort_values(key).reset_index(drop=True)
            Wb, ncl = cluster_weights(J.image.to_numpy(), rng)
            err, f5, f10 = J.ape.to_numpy(float), J.fail5.to_numpy(bool), J.fail10.to_numpy(bool)
            qs = {name: pred_q(J, kind, prot) for name, kind, prot in predictors()}
            k_gate = int(J.gate_escalate.sum())
            boot = boot_all(qs, err, f5, Wb)
            base = boot["ours confidence"]
            for name, kind, prot in predictors():
                q = qs[name]
                r = dict(scope=sname, target_method=tname, predictor=name, protocol=prot or "",
                         status=status_of(kind, prot), n=len(J),
                         n_crops=ncl, n_fail5=int(f5.sum()), n_fail10=int(f10.sum()),
                         spearman=spearman(q, err), auroc_fail5=auroc(-q, f5), auroc_fail10=auroc(-q, f10))
                r["spearman_lo"], r["spearman_hi"] = ci(boot[name][0])
                r["auroc_fail5_lo"], r["auroc_fail5_hi"] = ci(boot[name][1])
                if name != "ours confidence":
                    dr = boot[name][0] - base[0]
                    da = boot[name][1] - base[1]
                    r["d_spearman_vs_conf"] = r["spearman"] - spearman(qs["ours confidence"], err)
                    r["d_spearman_vs_conf_lo"], r["d_spearman_vs_conf_hi"] = ci(dr)
                    r["d_auroc5_vs_conf"] = r["auroc_fail5"] - auroc(-qs["ours confidence"], f5)
                    r["d_auroc5_vs_conf_lo"], r["d_auroc5_vs_conf_hi"] = ci(da)
                # matched escalation budget: escalate the k_gate lowest-quality crops (ties: stable by crop key)
                if kind == "gate":
                    esc = J.gate_escalate.to_numpy(bool)
                else:
                    esc = np.zeros(len(J), bool)
                    esc[np.argsort(q, kind="stable")[:k_gate]] = True
                r.update(k_escalated=k_gate, fail5_caught=int((esc & f5).sum()), fail10_caught=int((esc & f10).sum()),
                         kept_fail10_rate=float((~esc & f10).sum() / max((~esc).sum(), 1)))
                rows.append(r)
    A = pd.DataFrame(rows)
    A.insert(0, "part", "A_predict_error")
    # per-condition means (mechanism): ours / GPT P1 error, conf, IQA [work]
    W = wide.set_index(key)
    C = []
    for (kind, level), g in wide.groupby(["kind", "level"]):
        idx = g.set_index(key).index
        row = dict(part="A_by_condition", set=g.set.iloc[0], kind=kind, level=level, n=len(g),
                   ours_mape=tgt["ours"].ape.reindex(idx).mean(),
                   ours_fail5=int(tgt["ours"].fail5.reindex(idx).fillna(False).astype(bool).sum()),
                   gpt_p1_mape=tgt["GPT-6.1-sol P1"].ape.reindex(idx).mean(),
                   gabor_mape=tgt["log-Gabor"].ape.reindex(idx).mean(),
                   conf_mean=W.conf.reindex(idx).mean(), eta_median=W.eta.reindex(idx).median(),
                   gate_escalated=int(W.gate_escalate.reindex(idx).sum()))
        for m in METRICS:
            row[f"{m}_work_mean"] = W[f"{m}_work"].reindex(idx).mean()
        C.append(row)
    C = pd.DataFrame(C)
    return A, C


def part_a_family(wide, tgt):
    """EXPLORATORY 'when': within each degradation family (13 clean crops + the family's levels), how well does
    each signal rank the error and flag a > 5 % count?  Crop-clustered bootstrap, paired differences to ours'
    confidence on the same resamples.  GPT-6.1-sol P1 exists only for clean + the 8 VLM conditions."""
    rng = np.random.default_rng(SEED + 7)
    key = ["image", "kind", "level"]
    preds = [p for p in predictors() if p[2] in (None, "work")]
    rows = []
    for fam, fname in FAMILIES.items():
        extra = fam in ("motion", "noise", "shadow")
        Wd = wide[((wide.set == "legacy13") & (wide.kind == "clean")) | (wide.kind == fam)]
        for tname in ["ours", "GPT-6.1-sol P1", ANY2 if extra else ANY3]:
            if extra and tname.startswith("GPT"):
                continue
            J = Wd.set_index(key).join(tgt[tname], how="inner").reset_index()
            J = J.sort_values(key).reset_index(drop=True)
            f5 = J.fail5.to_numpy(bool)
            if len(J) < 20 or f5.sum() < 3 or (~f5).sum() < 3:
                continue
            Wb, ncl = cluster_weights(J.image.to_numpy(), rng)
            err = J.ape.to_numpy(float)
            qs = {name: pred_q(J, kind, prot) for name, kind, prot in preds}
            boot = boot_all(qs, err, f5, Wb)
            base = boot["ours confidence"]
            for name, kind, prot in preds:
                q = qs[name]
                r = dict(part="A_by_family", family=fname, kind=fam, target_method=tname, predictor=name,
                         protocol=prot or "", status=status_of(kind, prot) if kind not in PRESPEC else "exploratory",
                         n=len(J), n_crops=ncl, n_fail5=int(f5.sum()),
                         levels=",".join(f"{lv:g}" for lv in sorted(J.level[J.kind == fam].unique())),
                         spearman=spearman(q, err), auroc_fail5=auroc(-q, f5))
                r["spearman_lo"], r["spearman_hi"] = ci(boot[name][0])
                r["auroc_fail5_lo"], r["auroc_fail5_hi"] = ci(boot[name][1])
                if name != "ours confidence":
                    r["d_auroc5_vs_conf"] = r["auroc_fail5"] - auroc(-qs["ours confidence"], f5)
                    r["d_auroc5_vs_conf_lo"], r["d_auroc5_vs_conf_hi"] = ci(boot[name][1] - base[1])
                rows.append(r)
    return pd.DataFrame(rows)


def within_crop_stats(q, err, fail, groups, W_rows):
    """Within-fabric association (EXPLORATORY 'when'): only comparisons between conditions of the SAME crop, so a
    signal cannot score by recognising intrinsically hard fabrics.  Returns the stratified AUROC (pairs of a failing
    and a non-failing condition of one crop; ties 1/2), the mean within-crop Spearman, and their bootstrap
    replicates for crop-weight matrix W_rows (b x n_crops)."""
    ug = np.unique(groups)
    conc, pairs, rhos = np.zeros(len(ug)), np.zeros(len(ug)), np.full(len(ug), np.nan)
    for k, g in enumerate(ug):
        m = groups == g
        qq, ee, ff = q[m], err[m], fail[m]
        if ff.any() and (~ff).any():
            d = qq[ff][:, None] - qq[~ff][None, :]          # failing condition should have LOWER quality
            conc[k] = (d < 0).sum() + 0.5 * (d == 0).sum()
            pairs[k] = d.size
        rhos[k] = spearman(qq, ee)
    ok = np.isfinite(rhos)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_b = (W_rows @ conc) / (W_rows @ pairs)
        r_b = (W_rows[:, ok] @ rhos[ok]) / W_rows[:, ok].sum(1)
    a = conc.sum() / pairs.sum() if pairs.sum() else np.nan
    return a, float(np.nanmean(rhos)), a_b, r_b, int((pairs > 0).sum())


def part_a_within(wide, tgt):
    """EXPLORATORY: within-fabric versions of part A (all conditions per scope, and per degradation family)."""
    rng = np.random.default_rng(SEED + 11)
    key = ["image", "kind", "level"]
    preds = [p for p in predictors() if p[2] in (None, "work")]
    s19, s9, s29 = ("all 19 conditions (247 crops)", "clean + 8 VLM conditions (117 crops)",
                    "29 conditions incl. motion/noise/shadow (377 crops)")
    jobs = [(s19, wide[wide.set == "legacy13"], ["ours", ANY3]),
            (s9, wide[(wide.set == "legacy13") & wide.vlm_cond], ["ours", "GPT-6.1-sol P1", ANY4]),
            (s29, wide, ["ours", ANY2])]
    for fam, fname in FAMILIES.items():
        extra = fam in ("motion", "noise", "shadow")
        Wd = wide[((wide.set == "legacy13") & (wide.kind == "clean")) | (wide.kind == fam)]
        jobs.append((f"family: {fname}", Wd, ["ours", ANY2] if extra else ["ours", "GPT-6.1-sol P1", ANY3]))
    W_rows = rng.multinomial(13, np.ones(13) / 13, size=B).astype(float)     # crop bootstrap (13 crops)
    rows = []
    for scope, Wd, tnames in jobs:
        for tname in tnames:
            J = Wd.set_index(key).join(tgt[tname], how="inner").reset_index().sort_values(key).reset_index(drop=True)
            groups = J.image.to_numpy()
            if len(np.unique(groups)) != 13:
                continue
            err, f5 = J.ape.to_numpy(float), J.fail5.to_numpy(bool)
            res = {name: within_crop_stats(pred_q(J, kind, prot), err, f5, groups, W_rows) for name, kind, prot in preds}
            base = res["ours confidence"]
            if scope.startswith("family") and base[4] < 5:     # < 5 fabrics with both outcomes: not estimable
                continue
            for name, kind, prot in preds:
                a, r, a_b, r_b, n_mixed = res[name]
                row = dict(part="A_within_fabric", scope=scope, target_method=tname, predictor=name, protocol=prot or "",
                           status="task signal (reference)" if kind in ("conf", "eta", "gate") else "exploratory",
                           n=len(J), n_crops=13, n_fail5=int(f5.sum()), n_crops_mixed=n_mixed,
                           auroc_fail5_within=a, mean_spearman_within=r)
                row["auroc_fail5_within_lo"], row["auroc_fail5_within_hi"] = ci(a_b)
                row["mean_spearman_within_lo"], row["mean_spearman_within_hi"] = ci(r_b)
                if name != "ours confidence":
                    row["d_auroc5_within_vs_conf"] = a - base[0]
                    row["d_auroc5_within_vs_conf_lo"], row["d_auroc5_within_vs_conf_hi"] = ci(a_b - base[2])
                rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ B. genai_sr
def genai_clusters(keys, sets):
    """Bootstrap clusters for genai_sr images: each R-LEG crop is its own scene; the 53 R-HR crops come from
    40 PickGlass scenes (annotation_tool manifest scene_id), so R-HR crops of one scene form one cluster."""
    gm = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv")).drop_duplicates("key").set_index("key")
    man = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    scene = {int(e["id"]): e.get("scene_id") for e in man["images"]}
    return np.array([f"LEG:{k}" if st == "R-LEG" else f"scene:{scene[int(gm.loc[k, 'ident'])]}"
                     for k, st in zip(keys, sets)])


def load_genai():
    sc = pd.read_csv(os.path.join(SCORES, "iqa_genai_scores.csv"))
    # count errors per image (primary reference on R-LEG = hand counts; R-HR: each estimator vs its own
    # clean prediction, the pre-registered secondary / self-consistency reference used in the paper's table)
    pc = pd.read_csv(os.path.join(GSR, "results", "per_count_long.csv"))
    clean = pc[pc.restorer == "clean"].set_index(["key", "estimator", "axis"]).pred
    pc["ref_self"] = clean.reindex(pd.MultiIndex.from_frame(pc[["key", "estimator", "axis"]])).to_numpy()
    pc["ref_use"] = np.where(pc.set == "R-LEG", pc.ref, pc.ref_self)
    pc["ape_use"] = 100 * (pc.pred - pc.ref_use).abs() / pc.ref_use
    e = pc.groupby(["key", "set", "cond", "restorer", "estimator"]).agg(ape=("ape_use", "mean"),
                                                                       ape_max=("ape_use", "max")).reset_index()
    e["fail5"], e["fail10"] = e.ape_max > 5, e.ape_max > 10
    E = e.pivot_table(index=["key", "set", "cond", "restorer"], columns="estimator", values=["ape", "fail5", "fail10"])
    E.columns = [f"{a}_{b}" for a, b in E.columns]
    # server MUSIQ / CLIP-IQA at 1024^2 from the full-resolution PNGs (genai_sr/server/iq_metrics.py)
    srv = pd.concat([pd.read_csv(os.path.join(GSR, "results", f"iq_{r}.csv")) for r in ["clean"] + RESTORERS])
    srv = srv.rename(columns={"musiq": "musiq_srv", "clipiqa": "clipiqa_srv"})
    return sc, E.reset_index(), srv


def paired_mean_ci(diff, rng, b=B, groups=None):
    """Mean of paired differences with a bootstrap CI; resampling scenes (groups) when given."""
    diff = np.asarray(diff, float)
    ok = np.isfinite(diff)
    groups = np.arange(diff.size) if groups is None else np.asarray(groups)
    diff, groups = diff[ok], groups[ok]
    if diff.size < 3:
        return float(np.mean(diff)) if diff.size else np.nan, np.nan, np.nan
    W, _ = cluster_weights(groups, rng, b)
    bs = (W @ diff) / W.sum(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def part_b(sc, E, srv):
    rng = np.random.default_rng(SEED + 1)
    view = sc[sc.protocol.isin(["view768", "view768_made"])].copy()      # restored outputs + clean refs
    V = view.pivot_table(index=["key", "set", "cond", "restorer"], values=list(METRICS)).reset_index()
    # clean reference per key (cond 'clean')
    Vc = V[V.restorer == "clean"].set_index("key")
    Sc_ = srv[srv.restorer == "clean"].set_index("key")
    tab = pd.read_csv(os.path.join(GSR, "results", "genai_hallucination_table.csv"))
    rows = []
    sources = [(m, "view768", V, Vc) for m in METRICS] + \
              [("musiq", "server1024", srv.rename(columns={"musiq_srv": "musiq"}), Sc_.rename(columns={"musiq_srv": "musiq"})),
               ("clipiqa", "server1024", srv.rename(columns={"clipiqa_srv": "clipiqa"}),
                Sc_.rename(columns={"clipiqa_srv": "clipiqa"}))]
    for metric, src, T, Tc in sources:
        sign = METRICS[metric][1]
        T = T[["key", "set", "cond", "restorer", metric]].dropna()
        for (st, cond), g in T[T.restorer != "clean"].groupby(["set", "cond"]):
            gb = g[g.restorer == "bicubic"].set_index("key")[metric]
            for rest, gr in g.groupby("restorer"):
                q = sign * gr.set_index("key")[metric]
                qc = sign * Tc[metric].reindex(q.index)
                qb = sign * gb.reindex(q.index)
                grp = genai_clusters(q.index, [st] * len(q))
                r = dict(part="B_cell", metric=METRICS[metric][0], source=src, set=st, cond=cond, restorer=rest,
                         generative=rest in GENERATIVE, n_img=len(q), q_mean=q.mean(), q_clean_mean=qc.mean(),
                         share_above_clean=float((q > qc).mean()))
                r["dq_vs_bicubic"], r["dq_vs_bicubic_lo"], r["dq_vs_bicubic_hi"] = (
                    paired_mean_ci((q - qb).to_numpy(), rng, groups=grp) if rest != "bicubic" else (np.nan,) * 3)
                r["dq_vs_clean"], r["dq_vs_clean_lo"], r["dq_vs_clean_hi"] = paired_mean_ci((q - qc).to_numpy(), rng,
                                                                                            groups=grp)
                t = tab[(tab.set == st) & (tab.cond == cond) & (tab.restorer == rest)]
                for est in ("ours", "cnn", "gabor"):
                    for c in ("MAPE", "dMAPE_vs_none", "dMAPE_vs_none_lo", "dMAPE_vs_none_hi", "verdict_vs_none",
                              "dMAPE_vs_bicubic", "verdict_vs_bicubic"):
                        r[f"{est}_{c}"] = t[f"{est}_{c}"].iloc[0] if len(t) and f"{est}_{c}" in t else np.nan
                rows.append(r)
    cells = pd.DataFrame(rows)

    # per-metric summary over the 27 generative cells (2 sets x 3 conditions; Nano Banana Pro on R-LEG only)
    summ = []
    for (metric, src), g in cells[cells.generative].groupby(["metric", "source"]):
        up_b = (g.dq_vs_bicubic_lo > 0)
        dn_b = (g.dq_vs_bicubic_hi < 0)
        up_c = (g.dq_vs_clean_lo > 0)
        ge_c = (g.dq_vs_clean > 0)
        cnn_w, ours_w = g.cnn_verdict_vs_none == "worse", g.ours_verdict_vs_none == "worse"
        summ.append(dict(part="B_summary", metric=metric, source=src, n_cells=len(g),
                         cells_iqa_up_vs_bicubic=int(up_b.sum()), cells_iqa_down_vs_bicubic=int(dn_b.sum()),
                         cells_iqa_mean_above_clean=int(ge_c.sum()), cells_iqa_sig_above_clean=int(up_c.sum()),
                         share_outputs_above_clean=float((g.share_above_clean * g.n_img).sum() / g.n_img.sum()),
                         cells_cnn_worse=int(cnn_w.sum()), cells_cnn_worse_and_iqa_up=int((cnn_w & up_b).sum()),
                         cells_ours_worse=int(ours_w.sum()), cells_ours_worse_and_iqa_up=int((ours_w & up_b).sum()),
                         cells_count_worse_any=int((cnn_w | ours_w).sum()),
                         cells_count_worse_any_and_iqa_up=int(((cnn_w | ours_w) & up_b).sum())))
    summ = pd.DataFrame(summ)

    # per-image: does IQA rank the count error of the generative outputs?  (EXPLORATORY; scene-clustered)
    # targets: each counter; 'any' = none of the three counters has both counts within 5 %; GPT-6.1-sol P1 only
    # exists on the R-LEG x8 SeedVR2 / Nano Banana Pro outputs (genai_sr/results/gpt_p1_predictions.csv).
    # References: R-LEG = the 2019 hand counts; R-HR = each counter's own count on the clean crop (self-consistency).
    J = V.merge(E, on=["key", "set", "cond", "restorer"], how="inner")
    J = J[J.restorer.isin(GENERATIVE)].reset_index(drop=True)
    J["cluster"] = genai_clusters(J.key, J.set)
    cf = pd.read_csv(os.path.join(GSR, "results", "per_count_long.csv"))
    cf = cf[cf.estimator == "ours"].groupby(["key", "cond", "restorer"]).ours_conf.first()
    J["conf"] = cf.reindex(pd.MultiIndex.from_frame(J[["key", "cond", "restorer"]])).to_numpy(float)
    for c in ("ape", "fail5"):
        for est in ("ours", "cnn", "gabor"):
            J[f"{c}_{est}"] = J[f"{c}_{est}"].astype(float)
    J["ape_any"] = J[["ape_ours", "ape_cnn", "ape_gabor"]].min(axis=1)
    J["fail5_any"] = (J[["fail5_ours", "fail5_cnn", "fail5_gabor"]].fillna(1) > 0).all(axis=1).astype(float)  # missing = no success
    gd = pd.read_csv(os.path.join(GSR, "results", "gpt_p1_predictions.csv"))
    gd = gd[gd.source.isin([f"{r} (this job)" for r in GENERATIVE])].copy()
    gd["restorer"] = gd.source.str.replace(" (this job)", "", regex=False)
    gd["key"] = "leg_" + gd.image.astype(str)
    aw = 100 * (gd.vertical.astype(float) - gd.warp).abs() / gd.warp
    af = 100 * (gd.horizontal.astype(float) - gd.weft).abs() / gd.weft
    gd["ape_gpt"], gd["fail5_gpt"] = 0.5 * (aw + af), (np.fmax(aw, af) > 5).astype(float)
    gd["cond"] = "D8"
    J = J.merge(gd[["key", "cond", "restorer", "ape_gpt", "fail5_gpt"]], on=["key", "cond", "restorer"], how="left")
    subsets = {"pooled (R-LEG + R-HR)": J, "R-LEG (hand counts)": J[J.set == "R-LEG"],
               "R-HR (self-consistency reference)": J[J.set == "R-HR"],
               "R-LEG x8, SeedVR2 + Nano Banana Pro": J[np.isfinite(J.ape_gpt)]}
    corr = []
    for sub, K in subsets.items():
        K = K.reset_index(drop=True)
        Wb, ncl = cluster_weights(K.cluster.to_numpy(), rng)
        ests = ("gpt",) if sub.startswith("R-LEG x8") else ("ours", "cnn", "gabor", "any")
        for est in ests:
            err, f5 = K[f"ape_{est}"].to_numpy(float), K[f"fail5_{est}"].to_numpy(float) > 0
            qs = {METRICS[m][0]: quality(K, m, None).to_numpy(float) for m in METRICS}
            qs["ours confidence"] = K.conf.to_numpy(float)
            boot = boot_all(qs, err, f5, Wb)
            for name, q in qs.items():
                row = dict(part="B_perimage", subset=sub, metric=name,
                           source="counter" if name == "ours confidence" else "view768",
                           status=("task signal (reference)" if name == "ours confidence" else "exploratory"),
                           target_method={"any": "any of ours/CNN/log-Gabor", "gpt": "GPT-6.1-sol P1"}.get(est, est),
                           n=len(K), n_refs=ncl, n_fail5=int(f5.sum()), spearman=spearman(q, err),
                           auroc_fail5=auroc(-q, f5))
                row["spearman_lo"], row["spearman_hi"] = ci(boot[name][0])
                row["auroc_fail5_lo"], row["auroc_fail5_hi"] = ci(boot[name][1])
                if name != "ours confidence":
                    row["d_auroc5_vs_conf"] = row["auroc_fail5"] - auroc(-qs["ours confidence"], f5)
                    row["d_auroc5_vs_conf_lo"], row["d_auroc5_vs_conf_hi"] = ci(boot[name][1] - boot["ours confidence"][1])
                    row["d_spearman_vs_conf"] = row["spearman"] - spearman(qs["ours confidence"], err)
                    row["d_spearman_vs_conf_lo"], row["d_spearman_vs_conf_hi"] = ci(boot[name][0] -
                                                                                    boot["ours confidence"][0])
                corr.append(row)
    corr = pd.DataFrame(corr)

    # viewing-copy proxy check: view768 vs 1024^2 full resolution (server MUSIQ/CLIP-IQA; local full-res PNGs)
    chk = []
    full = sc[sc.protocol == "fix1024"].pivot_table(index=["key", "cond", "restorer"], values=list(METRICS))
    vv = V.set_index(["key", "cond", "restorer"])
    for m in METRICS:
        j = vv[[m]].join(full[[m]], rsuffix="_full", how="inner").dropna()
        chk.append(dict(part="B_proxy_check", metric=METRICS[m][0], against="local full-res PNG, 1024^2", n=len(j),
                        spearman=spearman(j[m].to_numpy(), j[f"{m}_full"].to_numpy()),
                        mean_view=j[m].mean(), mean_full=j[f"{m}_full"].mean()))
    for m in ("musiq", "clipiqa"):
        s = srv.set_index(["key", "cond", "restorer"])[f"{m}_srv"]
        j = vv[[m]].join(s, how="inner").dropna()
        chk.append(dict(part="B_proxy_check", metric=METRICS[m][0], against="server full-res, 1024^2", n=len(j),
                        spearman=spearman(j[m].to_numpy(), j[f"{m}_srv"].to_numpy()),
                        mean_view=j[m].mean(), mean_full=j[f"{m}_srv"].mean()))
        # local 1024^2 recomputation of the server value on the files present on the Mac (reproducibility)
        jj = full[[m]].join(s, how="inner").dropna()
        chk.append(dict(part="B_proxy_check", metric=METRICS[m][0], against="server value vs local recompute (same input)",
                        n=len(jj), max_abs_diff=float((jj[m] - jj[f"{m}_srv"]).abs().max())))
    chk = pd.DataFrame(chk)

    # LR inputs as stored (native size) -- context only (different size from the restored outputs)
    nat = sc[sc.protocol == "native"].groupby(["set", "cond"])[list(METRICS)].mean().reset_index()
    nat.insert(0, "part", "B_lr_native_mean")
    return cells, summ, corr, chk, nat


# ============================================================================ C. real 150 (label-free)
def part_c():
    sc = pd.read_csv(os.path.join(SCORES, "iqa150_scores.csv"))
    out = []
    rep = sc[sc.is_scene_rep.astype(bool)]
    for unit, g in (("76 scene representatives", rep), ("150 files", sc)):
        for m in METRICS:
            q = quality(g, m, "work").to_numpy(float)
            out.append(dict(part="C_labelfree", unit=unit, metric=METRICS[m][0], protocol="work", n=len(g),
                            spearman_vs_conf=spearman(q, g.ours_conf.to_numpy(float)),
                            auroc_gate_escalation=auroc(-q, g.escalate_gate.astype(bool).to_numpy()),
                            spearman_vs_side_px=spearman(q, g.S.to_numpy(float)),
                            q_mean_full_res=float(np.mean(q[g.S.to_numpy() >= 1024])),
                            q_mean_quarter_res=float(np.mean(q[g.S.to_numpy() < 1024]))))
    return pd.DataFrame(out)


# ============================================================================ outputs
def fmt_ci(v, lo, hi, d=2, star=False):
    """'0.74 [0.54, 0.90]' with the CI in \\tiny; star = paired 95 % CI against ours' confidence excludes 0."""
    if v is None or not np.isfinite(v):
        return "--"
    f = lambda x: f"{x:.{d}f}".replace("-", "$-$")            # noqa: E731
    return f"{f(v)}{'$^{*}$' if star else ''}\\,{{\\tiny[{f(lo)},\\,{f(hi)}]}}"


def excl0(lo, hi):
    return lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)


def supp_table(A, summ, Wf, corr):
    """Supplement table: rows = quality signals; columns = which error, on which images."""
    def get(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        return df[m].iloc[0] if m.any() else None
    s19, s9 = "all 19 conditions (247 crops)", "clean + 8 VLM conditions (117 crops)"
    leg = "R-LEG (hand counts)"
    rows = [("\\ours confidence", "ours confidence", "ours confidence", None),
            ("Resolvability $\\eta$", "resolvability eta", None, None),
            ("Cascade gate (frozen)", "cascade gate (frozen; 1 = kept offline)", None, None)]
    rows += [(METRICS[m][0] + ("$^{\\dagger}$" if m in PRESPEC else ""), f"{METRICS[m][0]} [work]", METRICS[m][0], m)
             for m in METRICS]
    L = ["% Generated by paper/tools/iqa_generic.py -- do not edit by hand.",
         "\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3.2pt}",
         "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         " & \\multicolumn{5}{c}{Non-generative degradations of the 13 hand-counted legacy crops} & "
         "\\multicolumn{2}{c}{Generative restorations} \\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(l){7-8}",
         " & \\multicolumn{2}{c}{\\ours error, 247 crops} & GPT-6.1-sol P1 & No method & \\ours, within & "
         "\\ours, R-LEG & IQA up \\\\",
         "\\cmidrule(lr){2-3}",
         "Quality signal & $\\rho$ & AUROC & AUROC, 117 & within 5\\%, 117 & fabric, 247 & AUROC, 191 & vs bicubic \\\\",
         "\\midrule"]
    for label, pred, gname, m in rows:
        a = get(A, part="A_predict_error", scope=s19, target_method="ours", predictor=pred)
        g = get(A, part="A_predict_error", scope=s9, target_method="GPT-6.1-sol P1", predictor=pred)
        y = get(A, part="A_predict_error", scope=s9, target_method=ANY4, predictor=pred)
        w = get(Wf, scope=s19, target_method="ours", predictor=pred)
        b = get(corr, subset=leg, target_method="ours", metric=gname) if gname else None
        st = lambda r, c: r is not None and excl0(r.get(c + "_lo"), r.get(c + "_hi"))   # noqa: E731
        cells = [fmt_ci(a.spearman, a.spearman_lo, a.spearman_hi, star=st(a, "d_spearman_vs_conf")),
                 fmt_ci(a.auroc_fail5, a.auroc_fail5_lo, a.auroc_fail5_hi, star=st(a, "d_auroc5_vs_conf")),
                 fmt_ci(g.auroc_fail5, g.auroc_fail5_lo, g.auroc_fail5_hi, star=st(g, "d_auroc5_vs_conf")),
                 fmt_ci(y.auroc_fail5, y.auroc_fail5_lo, y.auroc_fail5_hi, star=st(y, "d_auroc5_vs_conf")),
                 fmt_ci(w.auroc_fail5_within, w.auroc_fail5_within_lo, w.auroc_fail5_within_hi,
                        star=st(w, "d_auroc5_within_vs_conf")),
                 fmt_ci(b.auroc_fail5, b.auroc_fail5_lo, b.auroc_fail5_hi, star=st(b, "d_auroc5_vs_conf"))
                 if b is not None else "--"]
        if m is None:
            cells.append("--")
        else:
            sm = summ[(summ.metric == METRICS[m][0]) & (summ.source == "view768")].iloc[0]
            cells.append(f"{int(sm.cells_iqa_up_vs_bicubic)}/{int(sm.n_cells)}")
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if pred.startswith("cascade gate"):
            L.append("\\midrule")
    nf = lambda r: int(r.n_fail5)                                                        # noqa: E731
    a0 = get(A, part="A_predict_error", scope=s19, target_method="ours", predictor="ours confidence")
    g0 = get(A, part="A_predict_error", scope=s9, target_method="GPT-6.1-sol P1", predictor="ours confidence")
    y0 = get(A, part="A_predict_error", scope=s9, target_method=ANY4, predictor="ours confidence")
    b0 = get(corr, subset=leg, target_method="ours", metric="ours confidence")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{\\textbf{Which quality signal predicts counting error, and when.} "
          "AUROC for flagging a photo with a count off by more than 5\\% (higher = better; 0.5 = chance) and "
          "Spearman $\\rho$ between quality and the mean absolute \\% error (negative = useful), with 95\\% CIs "
          "from a bootstrap that resamples the 13 fabrics. Columns 1--5: blur, down-sampling, JPEG, glare and "
          f"corner-error versions of the 13 legacy crops, scored against the 2019 hand counts ({nf(a0)} of {int(a0.n)}, "
          f"{nf(g0)} of {int(g0.n)} and {nf(y0)} of {int(y0.n)} failures). ``No method within 5\\%'': none of \\ours, log-Gabor, the CNN and "
          "GPT-6.1-sol P1 has both counts within 5\\%. ``Within fabric'': only conditions of the same crop are "
          "compared, so a signal cannot score by recognising intrinsically hard fabrics. Column 6: outputs of five "
          "generative restorers (Real-ESRGAN, SeedVR2, InvSR, FLUX.2 [klein], Nano Banana Pro) on $\\times$4, "
          f"$\\times$8 and $\\times$4+JPEG legacy inputs, against the hand counts ({nf(b0)} of {int(b0.n)} failures). "
          "Column 7: restorer $\\times$ condition cells (of 27, legacy and full-resolution PickGlass inputs) in which "
          "the score of the generative outputs is above bicubic's (paired 95\\% CI above zero). Generic IQA "
          "(pyiqa) sees the counter's working image (at most 1024\\,px, never up-sampled; generated outputs: "
          "768\\,px viewing copies). $^{\\dagger}$Metric pre-specified; the other generic-IQA metrics and the "
          "within-fabric and generative columns are exploratory. "
          "$^{*}$Paired 95\\% CI of the difference to \\ours confidence excludes zero.}",
          "\\label{tab:supp_iqa}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    global SCORES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=TAB, help="directory of the iqa_*_scores.csv tables (default tables/)")
    ap.add_argument("--out", default=TAB)
    a = ap.parse_args()
    SCORES, OUT = a.scores_dir, a.out
    os.makedirs(OUT, exist_ok=True)
    wide, tgt, th = load_legacy()
    A, C = part_a(wide, tgt)
    F = part_a_family(wide, tgt)
    Wf = part_a_within(wide, tgt)
    sc, E, srv = load_genai()
    cells, summ, corr, chk, nat = part_b(sc, E, srv)
    Cl = part_c() if os.path.exists(os.path.join(SCORES, "iqa150_scores.csv")) else pd.DataFrame()
    A_out = pd.concat([A, F, Wf, C], ignore_index=True)
    A_out.round(4).to_csv(os.path.join(OUT, "iqa_legacy13.csv"), index=False)
    pd.concat([summ, cells, corr, chk, nat], ignore_index=True).round(4).to_csv(os.path.join(OUT, "iqa_genai.csv"),
                                                                              index=False)
    open(os.path.join(OUT, "supp_iqa.tex"), "w").write(supp_table(A, summ, Wf, corr))

    def pick(scope, tm, pred):
        r = A[(A.scope == scope) & (A.target_method == tm) & (A.predictor == pred)].iloc[0]
        keys = ["n", "n_crops", "n_fail5", "spearman", "spearman_lo", "spearman_hi", "auroc_fail5", "auroc_fail5_lo",
                "auroc_fail5_hi", "d_spearman_vs_conf", "d_spearman_vs_conf_lo", "d_spearman_vs_conf_hi",
                "d_auroc5_vs_conf", "d_auroc5_vs_conf_lo", "d_auroc5_vs_conf_hi", "k_escalated", "fail5_caught",
                "fail10_caught"]
        return {k: (round(float(r[k]), 4) if k in r and pd.notna(r[k]) else None) for k in keys}
    import hashlib
    S = {"generated_by": "paper/tools/iqa_generic.py",
         "inputs_sha256": {f: hashlib.sha256(open(os.path.join(SCORES, f), "rb").read()).hexdigest()
                           for f in ("iqa_legacy13_scores.csv", "iqa_genai_scores.csv", "iqa150_scores.csv")
                           if os.path.exists(os.path.join(SCORES, f))},
         "status": "pre-specified: BRISQUE and NIQE ('work' protocol primary); everything else exploratory; "
                   "task signals (confidence, eta, gate) are references",
         "bootstrap": dict(B=B, seed=SEED, cluster="fabric (legacy, 13) / scene (genai: 13 R-LEG fabrics + 40 "
                           "PickGlass scenes behind the 53 R-HR crops)"),
         "gate_thresholds": th, "legacy13": {}, "genai_summary": {}, "genai_perimage": {},
         "within_fabric": Wf.round(4).to_dict("records"), "by_family": F.round(4).to_dict("records"),
         "proxy_check": chk.round(4).to_dict("records"),
         "real150_labelfree": Cl.round(4).to_dict("records") if len(Cl) else None}
    preds = [p[0] for p in predictors()]
    for scope, tms in (("all 19 conditions (247 crops)", ["ours", "log-Gabor", "CNN", ANY3]),
                       ("clean + 8 VLM conditions (117 crops)", ["ours", "log-Gabor", "GPT-6.1-sol P1", "GPT-6.1-sol P3",
                                                                 ANY4]),
                       ("29 conditions incl. motion/noise/shadow (377 crops)", ["ours", "log-Gabor", ANY2])):
        S["legacy13"][scope] = {tm: {p: pick(scope, tm, p) for p in preds} for tm in tms}
    S["genai_summary"] = summ.round(4).to_dict("records")
    S["genai_perimage"] = corr.round(4).to_dict("records")
    json.dump(S, open(os.path.join(OUT, "iqa_summary.json"), "w"), indent=1, default=float)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for scope in ("all 19 conditions (247 crops)", "clean + 8 VLM conditions (117 crops)",
                  "29 conditions incl. motion/noise/shadow (377 crops)"):
        print(f"\n== {scope}")
        x = A[A.scope == scope][["target_method", "predictor", "n", "n_fail5", "spearman", "spearman_lo", "spearman_hi",
                                 "auroc_fail5", "auroc_fail5_lo", "auroc_fail5_hi", "d_spearman_vs_conf_lo",
                                 "d_spearman_vs_conf_hi", "k_escalated", "fail5_caught", "fail10_caught"]]
        print(x.round(2).to_string(index=False))
    print("\n== by family (exploratory 'when'; clean + family levels; work protocol)")
    print(F[["family", "target_method", "predictor", "n", "n_fail5", "spearman", "spearman_lo", "spearman_hi",
             "auroc_fail5", "auroc_fail5_lo", "auroc_fail5_hi", "d_auroc5_vs_conf_lo", "d_auroc5_vs_conf_hi"]]
          .round(2).to_string(index=False))
    print("\n== within fabric (exploratory; stratified AUROC, mean within-crop Spearman; work protocol)")
    print(Wf[["scope", "target_method", "predictor", "n", "n_fail5", "n_crops_mixed", "auroc_fail5_within",
              "auroc_fail5_within_lo", "auroc_fail5_within_hi", "d_auroc5_within_vs_conf_lo",
              "d_auroc5_within_vs_conf_hi", "mean_spearman_within"]].round(2).to_string(index=False))
    print("\n== by condition")
    print(C.drop(columns=["part"]).round(2).to_string(index=False))
    print("\n== genai summary (27 generative cells)")
    print(summ.to_string(index=False))
    print("\n== genai per-image (generative outputs)")
    print(corr.round(2).to_string(index=False))
    print("\n== proxy check")
    print(chk.round(3).to_string(index=False))
    if len(Cl):
        print("\n== real 150 label-free")
        print(Cl.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

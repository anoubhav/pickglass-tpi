"""combined_v1_eval.py -- evaluation of the PRE-REGISTERED combined counter combined-v1 (written AFTER the freeze:
methods/results/PREREG_combined_v1.json).  Reads only frozen predictions / frozen-counter
outputs; never changes a threshold.

  --synth    the pre-declared held-out synthetic test (PREREG 'heldout_test'): fresh seeds 970000+ / 975000+
             (methods/results/combined_v1_synth_heldout.csv), 19 conditions; P5 / MAPE / P10 of combined-v1 vs ours vs
             log-Gabor; paired cluster bootstrap over crops (10 000, seed 20261011); decision rule as declared.
             Also the same numbers on the fit set S-cal (in-sample, for reference only).
             -> methods/results/combined_v1_eval_synth.json, combined_v1_eval_synth_by_condition.csv
  --legacy   POST HOC (legacy-13 was used during development): clean canonical crops in the legacy13_unified.csv
             row format with paired tests from paper/tools/legacy_unified.py (bootstrap over images, 10 000 resamples,
             seed 20261008 + 1, Wilcoxon, McNemar); every legacy degradation condition (19 + 10) and the
             PickGlass-Stress renders with paired image-bootstrap CIs vs ours and vs log-Gabor.
             -> methods/results/combined_v1_legacy13_unified_rows.csv, _per_count.csv, _paired.csv,
                combined_v1_legacy13_by_condition.csv, combined_v1_stress_legacy13.csv

    .tools/venv/bin/python methods/combined_v1_eval.py --synth --legacy
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
PAPER = os.path.join(ROOT, "paper")
PREREG = os.path.join(RES, "PREREG_combined_v1.json")
RULE_PY = os.path.join(RES, "combined_frozen_v1.py")
HELDOUT = os.path.join(RES, "combined_v1_synth_heldout.csv")
SCAL = {v: os.path.join(RES, f"cascade_synth_cal_{v}.csv") for v in ("legacygrid", "default")}
B = 10000
SEED = 20261011
APE_MISS = 100.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prereg_sha256s():
    """Release: SHA-256 of PREREG on disk, plus the frozen original's (tools/freeze_checksums.json) when this is
    the unchanged path-redacted release copy."""
    now = sha256(PREREG)
    out = {now}
    fc = os.path.join(ROOT, "tools", "freeze_checksums.json")
    if os.path.exists(fc):
        rel = os.path.relpath(PREREG, ROOT).replace(os.sep, "/")
        for r in json.load(open(fc)):
            if r["path"] == rel and r["sha256_released"] == now:
                out.add(r["sha256_original"])
    return out


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_frozen():
    pr = json.load(open(PREREG))
    assert sha256(RULE_PY) == pr["rule"]["file"]["sha256"], "combined_frozen_v1.py changed after the pre-registration"
    c = pr["frozen_dependencies"]["counter"]
    assert sha256(os.path.join(ROOT, c["path"])) == c["sha256"], "frozen counter changed"
    for k, v in pr["predictions"].items():
        assert sha256(os.path.join(ROOT, v["path"])) == v["sha256"], f"prediction file changed: {k}"
    return pr


def ape(pred, true):
    a = 100 * np.abs(np.asarray(pred, float) / np.asarray(true, float) - 1)
    return np.where(np.isfinite(a), a, APE_MISS)


# =================================================================================== synthetic
def synth_frame(path, set_name, R):
    s = pd.read_csv(path)
    s["level"] = s.level.astype(float).round(4)
    s["set"] = set_name
    var = s["variant"] if "variant" in s else set_name
    s["crop"] = (pd.Series(var, index=s.index).astype(str) + ":" + s.name.astype(str)).to_numpy()
    s["cond"] = s.kind + ":" + s.level.map(lambda x: f"{x:g}")
    t = np.c_[s.h_count, s.v_count]
    A = {"ours": np.c_[ape(s.ours_h, t[:, 0]), ape(s.ours_v, t[:, 1])],
         "log-Gabor": np.c_[ape(s.gabor_h, t[:, 0]), ape(s.gabor_v, t[:, 1])]}
    for var_, key in (("primary", "combined-v1"), ("secondary", "combined-v1-mape")):
        o = R.combine_arrays(s.ours_h, s.ours_v, s.gabor_h, s.gabor_v, s.sharpness, s.N_work, variant=var_)
        A[key] = np.c_[ape(o["h"], t[:, 0]), ape(o["v"], t[:, 1])]
        if var_ == "primary":
            src = np.c_[o["src_h"], o["src_v"]]
    return s, A, src


def cluster_boot(s, A, rng):
    """Per crop sums of hits5 / APE / hits10 and counts -> bootstrap draws of P5, MAPE, P10 for every method."""
    crops = s.crop.to_numpy()
    u, inv = np.unique(crops, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(float) * 2
    S = {}
    for k, a in A.items():
        S[k] = {"h5": np.bincount(inv, (a <= 5).sum(1), len(u)), "ape": np.bincount(inv, a.sum(1), len(u)),
                "h10": np.bincount(inv, (a <= 10).sum(1), len(u))}
    idx = rng.integers(0, len(u), size=(B, len(u)))
    N = n[idx].sum(1)
    D = {k: {"P5": 100 * v["h5"][idx].sum(1) / N, "MAPE": v["ape"][idx].sum(1) / N, "P10": 100 * v["h10"][idx].sum(1) / N}
         for k, v in S.items()}
    return D, len(u)


def summarise(s, A, label, rng):
    D, n_crops = cluster_boot(s, A, rng)
    out = {"set": label, "n_crops": n_crops, "n_rows": int(len(s)), "n_counts": int(2 * len(s)), "methods": {}, "paired": {}}
    for k, a in A.items():
        out["methods"][k] = {"P5": float(100 * (a <= 5).mean()), "MAPE": float(a.mean()), "P10": float(100 * (a <= 10).mean()),
                             **{f"{m}_lo": float(np.percentile(D[k][m], 2.5)) for m in ("P5", "MAPE", "P10")},
                             **{f"{m}_hi": float(np.percentile(D[k][m], 97.5)) for m in ("P5", "MAPE", "P10")}}
    for a_ in ("combined-v1", "combined-v1-mape"):
        for b_ in ("ours", "log-Gabor"):
            r = {}
            for m in ("P5", "MAPE", "P10"):
                d = D[a_][m] - D[b_][m]
                r[f"d{m}"] = out["methods"][a_][m] - out["methods"][b_][m]
                r[f"d{m}_lo"], r[f"d{m}_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            # per count: which of the two is within 5 % (McNemar-style discordant counts)
            wa, wb = A[a_] <= 5, A[b_] <= 5
            r["counts_only_A_within5"], r["counts_only_B_within5"] = int((wa & ~wb).sum()), int((~wa & wb).sum())
            out["paired"][f"{a_} vs {b_}"] = r
    return out


def eval_synth(args):
    pr = check_frozen()
    R = load_mod("combined_frozen_v1", RULE_PY)
    meta = json.load(open(HELDOUT.replace(".csv", "_meta.json")))
    assert meta.get("prereg_sha256") in prereg_sha256s(), "held-out set was not scored after this pre-registration"
    h, Ah, src = synth_frame(HELDOUT, "held-out", R)
    rng = np.random.default_rng(SEED)
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg_sha256": sha256(PREREG),
           "heldout_csv": {"path": os.path.relpath(HELDOUT, ROOT), "sha256": sha256(HELDOUT), "meta": meta},
           "ape_convention": f"non-finite prediction = miss, APE {APE_MISS:g} %",
           "bootstrap": f"paired cluster bootstrap over crops, {B} resamples, seed {SEED}",
           "prereg_errata": [
               "PREREG 'motivation' says ours is 'better clean-ish'; on clean legacy-13 ours has the higher MAPE (11.6 vs "
               "9.2 %) but one more count within 5 % (19 vs 18 of 26). Wording only; nothing was fitted on legacy.",
               "PREREG fit note says S-cal 'had never been used to choose anything about ours vs log-Gabor'; the cascade's "
               "'agree-ext' router (escalate to the VLM when |ours/log-Gabor - 1| > delta) was fitted on S-cal (Oct 7). "
               "It chose when to call the VLM, not between ours and log-Gabor; disclosed here, record left frozen."]}
    res["heldout_pooled"] = summarise(h, Ah, "held-out (both generator variants, 19 conditions)", rng)
    for var in ("legacygrid", "default"):
        m = (h.variant == var).to_numpy()
        res[f"heldout_{var}"] = summarise(h[m], {k: a[m] for k, a in Ah.items()}, f"held-out {var}", rng)
    m = (h.kind == "clean").to_numpy()
    res["heldout_clean"] = summarise(h[m], {k: a[m] for k, a in Ah.items()}, "held-out clean only", rng)
    m = (h.kind == "gblur").to_numpy()
    res["heldout_defocus"] = summarise(h[m], {k: a[m] for k, a in Ah.items()}, "held-out defocus (gblur 1-6u)", rng)
    srcc = pd.Series(src.ravel()).value_counts()
    res["heldout_primary_sources_pct"] = {k: float(100 * v / src.size) for k, v in srcc.items()}
    # decision rule (PREREG heldout_test)
    P = res["heldout_pooled"]
    dec = {}
    for b_ in ("ours", "log-Gabor"):
        r = P["paired"][f"combined-v1 vs {b_}"]
        dec[b_] = bool(r["dP5"] > 0 and r["dP5_lo"] > 0)
    res["decision"] = {"better_than_ours": dec["ours"], "better_than_log_gabor": dec["log-Gabor"],
                       "better_than_both": dec["ours"] and dec["log-Gabor"],
                       "rule": pr["heldout_test"]["decision_rule"]}
    # in-sample reference (fit set)
    fit = [synth_frame(SCAL[v], f"S-cal-{v}", R) for v in ("legacygrid", "default")]
    f = pd.concat([x[0] for x in fit], ignore_index=True)
    Af = {k: np.r_[fit[0][1][k], fit[1][1][k]] for k in fit[0][1]}
    res["fitset_in_sample"] = summarise(f, Af, "S-cal fit set (in-sample, reference only)", rng)
    # per condition (held-out, pooled over both variants)
    rows = []
    for scope, frame, AA in (("held-out", h, Ah), ("S-cal (in-sample)", f, Af)):
        for cond, g in frame.groupby("cond"):
            m = (frame.cond == cond).to_numpy()
            r = {"scope": scope, "cond": cond, "n_counts": int(2 * m.sum())}
            for k, a in AA.items():
                r[f"P5_{k}"] = 100 * (a[m] <= 5).mean()
                r[f"MAPE_{k}"] = a[m].mean()
            rows.append(r)
    C = pd.DataFrame(rows)
    C.round(2).to_csv(os.path.join(RES, "combined_v1_eval_synth_by_condition.csv"), index=False)
    json.dump(res, open(os.path.join(RES, "combined_v1_eval_synth.json"), "w"), indent=1)
    for k in ("heldout_pooled", "heldout_legacygrid", "heldout_default", "heldout_clean", "heldout_defocus", "fitset_in_sample"):
        x = res[k]
        print(f"\n== {x['set']}: {x['n_crops']} crops, {x['n_counts']} counts")
        for mth, v in x["methods"].items():
            print(f"  {mth:18s} P5 {v['P5']:5.1f} [{v['P5_lo']:.1f}, {v['P5_hi']:.1f}]  MAPE {v['MAPE']:6.2f} "
                  f"[{v['MAPE_lo']:.2f}, {v['MAPE_hi']:.2f}]  P10 {v['P10']:5.1f}")
        for pk, v in x["paired"].items():
            print(f"  {pk:32s} dP5 {v['dP5']:+5.2f} [{v['dP5_lo']:+.2f}, {v['dP5_hi']:+.2f}]  dMAPE {v['dMAPE']:+6.2f} "
                  f"[{v['dMAPE_lo']:+.2f}, {v['dMAPE_hi']:+.2f}]  only-A/only-B within 5%: "
                  f"{v['counts_only_A_within5']}/{v['counts_only_B_within5']}")
    print("\nprimary sources on held-out:", {k: round(v, 1) for k, v in res["heldout_primary_sources_pct"].items()})
    print("DECISION:", res["decision"])
    print(C[C.scope == "held-out"].drop(columns="scope").round(1).to_string(index=False))


# =================================================================================== legacy (POST HOC)
def legacy_gt(LU):
    """GT of Tab. 2 as it entered legacy13_unified_per_count.csv (no xlsx / openpyxl needed)."""
    pc = pd.read_csv(os.path.join(PAPER, "tables", "legacy13_unified_per_count.csv"))
    pc = pc[(pc.key == "est:ours") & (pc["assign"] == "axis")]
    rows = []
    for img, g in pc.groupby("image"):
        w, f = g[g.axis == "warp"].iloc[0], g[g.axis == "weft"].iloc[0]
        rows.append(dict(image=img, warp=w.true, weft=f.true, weave="1/1" if bool(w.plain) else "other",
                         warp_formula=bool(w.formula), weft_formula=bool(f.formula)))
    gt = pd.DataFrame(rows).set_index("image")
    assert len(gt) == 13
    return gt


def method(LU, key, label, h, v, note=""):
    return LU.P("classical", label, "--", h, v, key=key, usd=0.0, offline="yes", note=note)


def eval_legacy(args):
    check_frozen()
    LU = load_mod("legacy_unified", os.path.join(PAPER, "tools", "legacy_unified.py"))
    gt = legacy_gt(LU)
    # ---------------------------------------------------------------- clean canonical crops
    e = pd.read_csv(os.path.join(RES, "legacy13_vlmcrops_estimators.csv"))
    e = e[e.source == "vlmcrop"]
    c = pd.read_csv(os.path.join(RES, "legacy13_vlmcrops_estimators_combined_v1.csv"))
    M = {}
    for est, key, label in (("ours", "est:ours", "Ours (PG-Lattice: lattice + fusion)"),
                            ("gabor", "est:gabor", "Log-Gabor bank")):
        g = e[e.estimator == est].set_index("image")
        M[key] = method(LU, key, label, g.h, g.v)
    for est, key, label in (("combined_v1", "est:combined_v1", "Combined v1 (PG-Lattice + log-Gabor, pre-registered)"),
                            ("combined_v1_mape", "est:combined_v1_mape", "Combined v1, MAPE-objective variant (secondary)")):
        g = c[c.estimator == est].set_index("image")
        M[key] = method(LU, key, label, g.h, g.v, note="POST HOC on legacy-13; rule frozen 2026-10-11 on synthetic data")
    rows, counts = [], []
    for k, m in M.items():
        for assign in ("axis", "sorted"):
            r, d = LU.score(m, gt, assign)
            rows.append(dict(group="classical", method=m["method"], prompt="--", key=k, repeat=0, assign=assign,
                             crop_rule="canonical vlm/crops/legacy PNG", input_px=m["input_px"], **r, usd_per_img=0.0,
                             s_per_img=np.nan, latency_hw="", offline="yes", in_tex=False, note=m["note"]))
            counts.append(d.assign(key=k, method=m["method"], prompt="--", assign=assign))
    T = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(PAPER, "tables", "legacy13_unified.csv"))
    for k in ("est:ours", "est:gabor"):   # must reproduce Tab. 2 / Tab. S2 exactly
        a = ref[(ref.key == k) & (ref["assign"] == "axis")].iloc[0]
        b = T[(T.key == k) & (T["assign"] == "axis")].iloc[0]
        assert abs(a.MAPE - b.MAPE) < 1e-3 and abs(a.MAPE_lo - b.MAPE_lo) < 1e-3 and a.n_w5 == b.n_w5, k
    T = T[list(ref.columns)]
    T.round({cc: 3 for cc in T.columns if cc != "usd_per_img"}).to_csv(
        os.path.join(RES, "combined_v1_legacy13_unified_rows.csv"), index=False)
    pd.concat(counts).round(3).to_csv(os.path.join(RES, "combined_v1_legacy13_unified_per_count.csv"), index=False)
    comps = [("combined_v1 vs ours", "est:combined_v1", "est:ours"), ("combined_v1 vs log-Gabor", "est:combined_v1", "est:gabor"),
             ("combined_v1_mape vs ours", "est:combined_v1_mape", "est:ours"),
             ("combined_v1_mape vs log-Gabor", "est:combined_v1_mape", "est:gabor"),
             ("ours vs log-Gabor", "est:ours", "est:gabor")]
    PR = pd.DataFrame([LU.paired(M[a], M[b], gt, assign, lab) for lab, a, b in comps for assign in ("axis", "sorted")])
    PR["n_w5_A"] = [int((LU.per_count(M[a], gt, s).ape <= 5).sum()) for lab, a, b in comps for s in ("axis", "sorted")]
    PR["n_w5_B"] = [int((LU.per_count(M[b], gt, s).ape <= 5).sum()) for lab, a, b in comps for s in ("axis", "sorted")]
    PR.insert(0, "status", "POST HOC")
    PR.round(4).to_csv(os.path.join(RES, "combined_v1_legacy13_paired.csv"), index=False)
    print("== legacy-13 clean canonical crops (POST HOC), axis-matched:")
    print(T[T["assign"] == "axis"][["key", "n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10", "n_gross25"]]
          .round(2).to_string(index=False))
    print(PR[PR["assign"] == "axis"][["comparison", "MAPE_A", "MAPE_B", "dMAPE", "dMAPE_lo", "dMAPE_hi", "p_wilcoxon_abs_pct_err_26",
                                      "n_w5_A", "n_w5_B", "n_w10_A", "n_w10_B", "p_mcnemar_w10"]].round(3).to_string(index=False))
    # ---------------------------------------------------------------- degradation conditions (19 + 10)
    rows = []
    for src, comb in (("legacy13_vlmcrops_degrade.csv", "legacy13_vlmcrops_degrade_combined_v1.csv"),
                      ("legacy13_vlmcrops_degrade_extra.csv", "legacy13_vlmcrops_degrade_extra_combined_v1.csv")):
        d = pd.concat([pd.read_csv(os.path.join(RES, src)), pd.read_csv(os.path.join(RES, comb))], ignore_index=True)
        d["level"] = d.level.astype(float).round(4)
        for (kind, level), g in d.groupby(["kind", "level"]):
            MM = {}
            for est in ("ours", "gabor", "combined_v1", "combined_v1_mape"):
                x = g[g.estimator == est].set_index("image")
                MM[est] = method(LU, est, est, x.h, x.v)
            r = {"kind": kind, "level": level, "source": src}
            for est, m in MM.items():
                sc, dd = LU.score(m, gt, "axis")
                r.update({f"MAPE_{est}": sc["MAPE"], f"MAPE_lo_{est}": sc["MAPE_lo"], f"MAPE_hi_{est}": sc["MAPE_hi"],
                          f"n_w5_{est}": sc["n_w5"], f"n_w10_{est}": sc["n_w10"], f"n_counts_{est}": sc["n_counts"]})
            for a_ in ("combined_v1", "combined_v1_mape"):
                for b_ in ("ours", "gabor"):
                    p = LU.paired(MM[a_], MM[b_], gt, "axis", f"{a_} vs {b_}")
                    r[f"dMAPE_{a_}_vs_{b_}"], r[f"dMAPE_lo_{a_}_vs_{b_}"], r[f"dMAPE_hi_{a_}_vs_{b_}"] = (
                        p["dMAPE"], p["dMAPE_lo"], p["dMAPE_hi"])
                    r[f"p_wilcoxon_img_{a_}_vs_{b_}"] = p["p_wilcoxon_image_mean_pct_err_13"]
            x = g[g.estimator == "combined_v1"]
            src_all = np.r_[[s.split("|") for s in x.fusion]].ravel()
            r["pct_mean_primary"] = 100 * np.mean(src_all == "mean")
            x = g[g.estimator == "combined_v1_mape"]
            r["pct_fallback_secondary"] = 100 * np.mean(np.r_[[s.split("|") for s in x.fusion]].ravel() == "gabor-fallback")
            rows.append(r)
    DG = pd.DataFrame(rows)
    DG.insert(0, "status", "POST HOC")
    DG.round(3).to_csv(os.path.join(RES, "combined_v1_legacy13_by_condition.csv"), index=False)
    show = ["kind", "level", "MAPE_ours", "MAPE_gabor", "MAPE_combined_v1", "MAPE_combined_v1_mape", "n_w5_ours", "n_w5_gabor",
            "n_w5_combined_v1", "n_w5_combined_v1_mape", "dMAPE_combined_v1_vs_ours", "dMAPE_lo_combined_v1_vs_ours",
            "dMAPE_hi_combined_v1_vs_ours", "dMAPE_combined_v1_vs_gabor", "dMAPE_lo_combined_v1_vs_gabor",
            "dMAPE_hi_combined_v1_vs_gabor", "pct_mean_primary", "pct_fallback_secondary"]
    print("\n== legacy-13 degradations (POST HOC), axis-matched, 26 counts each:")
    print(DG[show].round(1).to_string(index=False))
    # ---------------------------------------------------------------- PickGlass-Stress (primary only)
    SS = load_mod("score_stress", os.path.join(ROOT, "camera_aug", "score_stress.py"))
    lab = pd.read_csv(os.path.join(ROOT, "camera_aug", "stress", "legacy13", "labels.csv"), dtype={"source_id": str})
    s = pd.read_csv(os.path.join(ROOT, "camera_aug", "results", "stress_legacy13_classical.csv"), dtype={"source_id": str})
    cs = pd.read_csv(os.path.join(RES, "stress_legacy13_combined_v1.csv"), dtype={"source_id": str})
    pred = pd.concat([s[["source_id", "condition", "rep", "estimator", "h", "v"]],
                      cs[["source_id", "condition", "rep", "estimator", "h", "v"]]], ignore_index=True)
    d = SS.per_count(pred, lab)
    m = SS.score(d)
    out = []
    for cond, g in d.groupby("condition"):
        t = {est: SS.img_table(g[g.estimator == est]) for est in ("ours", "gabor", "combined_v1")}
        r = {"condition": cond, "name": SS.NAMES.get(cond, cond)}
        for est in t:
            mm = m[(m.estimator == est) & (m.condition == cond)].iloc[0]
            r.update({f"MAPE_{est}": mm.MAPE, f"MAPE_lo_{est}": mm.MAPE_lo, f"MAPE_hi_{est}": mm.MAPE_hi,
                      f"w5_pct_{est}": float(100 * (g[g.estimator == est].ape <= 5).mean())})
        rng = np.random.default_rng([SEED, zlib.crc32(cond.encode())])
        for b_ in ("ours", "gabor"):
            j = t["combined_v1"].join(t[b_], rsuffix="_b", how="inner")
            diff = (j.ape - j.ape_b).to_numpy()
            w = j.n.to_numpy(float)
            ii = rng.integers(0, len(j), size=(B, len(j)))
            bd = (diff[ii] * w[ii]).sum(1) / w[ii].sum(1)
            r[f"dMAPE_vs_{b_}"] = float((diff * w).sum() / w.sum())
            r[f"dMAPE_lo_vs_{b_}"], r[f"dMAPE_hi_vs_{b_}"] = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))
            r[f"p_wilcoxon_img_vs_{b_}"] = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        out.append(r)
    ST = pd.DataFrame(out)
    ST["_o"] = ST.condition.map({c_: i for i, c_ in enumerate(SS.ORDER)})
    ST = ST.sort_values("_o").drop(columns="_o")
    ST.insert(0, "status", "POST HOC")
    ST.round(3).to_csv(os.path.join(RES, "combined_v1_stress_legacy13.csv"), index=False)
    print("\n== PickGlass-Stress legacy-13 (POST HOC; camera conditions averaged over 3 renders):")
    print(ST[["condition", "MAPE_ours", "MAPE_gabor", "MAPE_combined_v1", "w5_pct_ours", "w5_pct_gabor", "w5_pct_combined_v1",
              "dMAPE_vs_ours", "dMAPE_lo_vs_ours", "dMAPE_hi_vs_ours", "dMAPE_vs_gabor", "dMAPE_lo_vs_gabor",
              "dMAPE_hi_vs_gabor"]].round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    if a.synth:
        eval_synth(a)
    if a.legacy:
        eval_legacy(a)

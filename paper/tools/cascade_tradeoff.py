"""cascade_tradeoff.py -- quality-aware cascade: run the offline counter ("ours", free, ~3.5 s) by
default and escalate a photo to a VLM (GPT-6.1-sol P1, ~US$0.0115 and ~14 s per call) only when a
no-reference signal computed from ours' own output says ours is unreliable.

DESIGN (fixed before any legacy cascade number was computed; see the 'protocol' key of
tables/cascade_thresholds.json for the time stamps):

Routers (decide per photo from ours / log-Gabor outputs on that photo; no labels):
  conf        escalate iff conf < t_c        (ours' logistic confidence, fitted on synthetic data)
  blur        escalate iff rho < r_min       resolvability check:
                rho = sharpness / (4 sin^2(pi f))^2,  f = max(h, v) / N_work  [cycles / working px]
              'sharpness' (already computed by counting.preprocess) is var(Laplacian) / var(image);
              for a grating at frequency f the 4-neighbour Laplacian has gain 4 sin^2(pi f), so rho ~ 1
              when the image carries its energy at the claimed (finer) yarn frequency and << 1 when
              it cannot (the over-count under defocus, where the spectrum locks onto residual
              high-frequency artefacts).  Motivated by the defocus failure already reported in the
              paper (sec. 5.3); its threshold is fitted on synthetic data only.
  gate        escalate iff conf < t_c OR rho < r_min                         [PRIMARY ROUTER]
  agree-int   escalate iff the lattice estimate was not confirmed by a second internal estimator
              (profile / tiles, 6 % rule of _fuse) on at least one axis       [parameter-free]
  agree-ext   escalate iff max over axes |ours / log-Gabor - 1| > delta      (two offline counters)
Both-and-pick (the VLM is always called):
  both-median per axis the median of {ours, log-Gabor, VLM}
  both-vote   per axis ours if |ours / VLM - 1| <= 10 %, else whichever of ours / VLM is closer
              (log ratio) to log-Gabor
Oracle router: per photo the better of ours / VLM (uses the labels; upper bound).

Thresholds (t_c, r_min, delta) are chosen ONLY on the synthetic calibration set S-cal
(methods/cascade_synth.py, 200 fresh synthetic crops x the 9 conditions on which the VLM was run):
maximise coverage (share answered offline) subject to selective risk <= ALPHA = 10 %, risk =
share of accepted crops whose two counts are NOT both within 10 % of the synthetic truth.
Sensitivity (reported, not used to choose): ALPHA 5 / 20 %, fit on all 19 synthetic conditions, and
the a-priori rule conf >= 0.9.  Alternative fallbacks: GPT-6.1-sol P3, Gemini 3.8 Flash P3.

Evaluation: the 13 hand-counted legacy crops x the 9 conditions (clean + 8 degradations) on which
ours, log-Gabor and the VLMs read byte-identical pixels (tables/degradation_legacy13_per_image.csv
lineage, via degradation_table.load()).  Axis convention: vertical yarns <-> warp.  95 % CIs:
image-level (cluster) bootstrap, 10,000 resamples, the same resamples for every policy (paired);
Wilcoxon signed-rank on the 13 per-image mean APEs.  Cost: the actual OpenRouter cost of each
escalated call of the stored run (no new API calls).

Outputs
  tables/cascade_thresholds.json      synthetic-fitted thresholds + synthetic operating points
  tables/cascade_synth_by_condition.csv
  tables/cascade_legacy13.csv         one row per policy (MAPE, w10, worst-condition MAPE, $, offline, paired tests)
  tables/cascade_legacy13_by_condition.csv
  tables/cascade_sweep.csv            fixed-escalation-rate sweeps (risk-coverage / cost-accuracy)
  tables/cascade_150_escalation.csv   label-free escalation rate of the gate on the 150 real photos
  tables/cascade_signal_quality.csv   Spearman / AUC of each routing signal (legacy and synthetic)
  figs/cascade_tradeoff.pdf / .png

    <venv2>/bin/python tools/cascade_tradeoff.py --fit      # synthetic only -> thresholds json
    <venv2>/bin/python tools/cascade_tradeoff.py --eval     # apply once to legacy + figure
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figs")
RES = os.path.join(ROOT, "methods", "results")
sys.path.insert(0, HERE)

SYN_CSV = os.path.join(RES, "cascade_synth_cal_legacygrid.csv")
SYN_DEFAULT_CSV = os.path.join(RES, "cascade_synth_cal_default.csv")
THR_JSON = os.path.join(TAB, "cascade_thresholds.json")
VLM_CONDS = [("clean", 0.0), ("gblur", 3.0), ("gblur", 6.0), ("down", 4.0), ("down", 8.0), ("jpeg", 20.0),
             ("glare", 0.05), ("glare", 0.3), ("corners", 0.05)]
COND_LABEL = {("clean", 0.0): "clean", ("gblur", 3.0): "defocus 3u", ("gblur", 6.0): "defocus 6u",
              ("down", 4.0): "down x4", ("down", 8.0): "down x8", ("jpeg", 20.0): "JPEG q20",
              ("glare", 0.05): "glare 5%", ("glare", 0.3): "glare 30%", ("corners", 0.05): "corners 5%"}
ALPHA = 0.10
ALPHAS_SENS = [0.05, 0.20]
LAT_OK = {"fft+profile", "fft+tiles"}  # lattice confirmed by a second estimator on this axis
B = 10000
SEED = 20261008


# ----------------------------------------------------------------------------- signals
def rho_of(h, v, sharp, n_work):
    """Resolvability ratio: observed Laplacian/variance ratio over that of a grating at the finer
    claimed yarn frequency (0 when ours returned no count)."""
    h, v, sharp, n_work = (np.asarray(x, float) for x in (h, v, sharp, n_work))
    f = np.fmax(h, v) / n_work
    gain = (4 * np.sin(np.pi * np.clip(f, 1e-6, 0.5)) ** 2) ** 2
    r = sharp / gain
    return np.where(np.isfinite(r), r, 0.0)


def d_ext_of(h, v, gh, gv):
    d = np.fmax(np.abs(np.asarray(h) / np.asarray(gh) - 1), np.abs(np.asarray(v) / np.asarray(gv) - 1))
    return np.where(np.isfinite(d), d, np.inf)


def lat_confirmed(fusion):
    """1 if the lattice estimate was confirmed on both axes, else 0."""
    out = []
    for s in fusion:
        if not isinstance(s, str):
            out.append(0.0)
            continue
        out.append(float(all(p in LAT_OK for p in s.split("|"))))
    return np.array(out)


# ----------------------------------------------------------------------------- synthetic fit
def load_synth(path=None):
    s = pd.read_csv(path or SYN_CSV)
    s["level"] = s.level.astype(float).round(4)
    s["vlm_cond"] = [(k, lv) in set(VLM_CONDS) for k, lv in zip(s.kind, s.level)]
    s["ape_h"] = 100 * (s.ours_h / s.h_count - 1).abs()
    s["ape_v"] = 100 * (s.ours_v / s.v_count - 1).abs()
    s["err_img"] = 0.5 * (s.ape_h + s.ape_v)
    s["fail"] = ~((s.ape_h <= 10) & (s.ape_v <= 10))  # NaN counts -> fail
    s["conf"] = s.ours_conf.fillna(0.0)
    s["rho"] = rho_of(s.ours_h, s.ours_v, s.sharpness, s.N_work)
    s["d_ext"] = d_ext_of(s.ours_h, s.ours_v, s.gabor_h, s.gabor_v)
    s["lat_ok"] = lat_confirmed(s.fusion)
    return s


def fit_1d(score, fail, alpha):
    """accept iff score >= t; max coverage s.t. selective risk <= alpha -> (t, coverage, risk)."""
    score, fail = np.asarray(score, float), np.asarray(fail, bool)
    cands = np.unique(score)[::-1]  # descending
    best = (np.inf, 0.0, 0.0)
    for t in cands:
        acc = score >= t
        cov, risk = acc.mean(), fail[acc].mean()
        if risk <= alpha and cov > best[1]:
            best = (float(t), float(cov), float(risk))
    return best


def fit_gate(conf, rho, fail, alpha, nq=101):
    conf, rho, fail = np.asarray(conf, float), np.asarray(rho, float), np.asarray(fail, bool)
    tcs = np.unique(np.r_[0.0, np.quantile(conf, np.linspace(0, 0.995, nq))])
    rms = np.unique(np.r_[0.0, np.quantile(rho, np.linspace(0, 0.995, nq))])
    best = dict(t_c=np.inf, r_min=np.inf, coverage=0.0, risk=np.nan)
    for tc in tcs:
        a1 = conf >= tc
        for rm in rms:
            acc = a1 & (rho >= rm)
            cov = acc.mean()
            if cov == 0:
                continue
            risk = fail[acc].mean()
            if risk <= alpha and (cov > best["coverage"] + 1e-12 or
                                  (abs(cov - best["coverage"]) <= 1e-12 and risk < best["risk"])):
                best = dict(t_c=float(tc), r_min=float(rm), coverage=float(cov), risk=float(risk))
    return best


def fit(args):
    s = load_synth()
    sv = s[s.vlm_cond]
    out = {"protocol": {"fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "synthetic_csv": os.path.relpath(SYN_CSV, ROOT),
                        "n_images": int(s.name.nunique()), "n_rows_vlm_conds": int(len(sv)),
                        "n_rows_all_conds": int(len(s)), "alpha_primary": ALPHA,
                        "objective": "max coverage s.t. share of accepted synthetic crops without both "
                                     "counts within 10 % <= alpha",
                        "legacy_labels_used": False},
           "fits": {}}
    for scope, df in (("vlm9", sv), ("all19", s)):
        for a in [ALPHA] + ALPHAS_SENS:
            key = f"{scope}_a{int(round(100 * a)):02d}"
            g = fit_gate(df.conf, df.rho, df.fail, a)
            t_c, cov_c, risk_c = fit_1d(df.conf, df.fail, a)
            r_b, cov_b, risk_b = fit_1d(df.rho, df.fail, a)
            d_e, cov_e, risk_e = fit_1d(-df.d_ext.clip(upper=1e6), df.fail, a)
            out["fits"][key] = {
                "gate": g,
                "conf": dict(t_c=t_c, coverage=cov_c, risk=risk_c),
                "blur": dict(r_min=r_b, coverage=cov_b, risk=risk_b),
                "agree_ext": dict(delta=-d_e, coverage=cov_e, risk=risk_e)}
    # sensitivity: the generator's default resolution distribution (crops <= 1024 px), scored on the
    # GPU server (numpy 2 / OpenCV 5): thresholds re-fitted there and applied to legacy as a check
    if os.path.exists(SYN_DEFAULT_CSV):
        sd = load_synth(SYN_DEFAULT_CSV)
        sdv = sd[sd.vlm_cond]
        out["protocol"]["synthetic_default_csv"] = os.path.relpath(SYN_DEFAULT_CSV, ROOT)
        for a in [ALPHA]:
            key = f"default_vlm9_a{int(round(100 * a)):02d}"
            t_c, cov_c, risk_c = fit_1d(sdv.conf, sdv.fail, a)
            out["fits"][key] = {"gate": fit_gate(sdv.conf, sdv.rho, sdv.fail, a),
                                "conf": dict(t_c=t_c, coverage=cov_c, risk=risk_c)}
        out["fits"]["default_vlm9_ours_only"] = dict(coverage=1.0, risk=float(sdv.fail.mean()))
    # secondary blur threshold, declared before any legacy cascade number was computed (after seeing
    # that the default-generator synthetic set does not reproduce the defocus over-count, so a
    # synthetic fit may leave the blur check unused): label-free, from the 150 REAL clean crops --
    # r_min = the p-th percentile of rho, i.e. the blur check alone escalates p % of clean real photos
    d150 = pd.read_csv(os.path.join(RES, "all150_degrade.csv"))
    d150 = d150[(d150.estimator == "ours") & (d150.crop == "gt") & (d150.kind == "clean")]
    r150 = rho_of(d150.h, d150.v, d150.sharpness, np.minimum(d150.S_deg, 1024))
    out["fits"]["real150_clean_rho_pct"] = {f"p{p:02d}": float(np.percentile(r150, p)) for p in (2, 5, 10)}
    out["fits"]["real150_clean_rho_pct"]["n"] = int(len(r150))
    # parameter-free rules on synthetic, for reference
    for scope, df in (("vlm9", sv), ("all19", s)):
        acc = df.lat_ok >= 1
        out["fits"][f"{scope}_agree_int"] = dict(coverage=float(acc.mean()), risk=float(df.fail[acc].mean()))
        acc = df.conf >= 0.9
        out["fits"][f"{scope}_conf090"] = dict(coverage=float(acc.mean()), risk=float(df.fail[acc].mean()))
        out["fits"][f"{scope}_ours_only"] = dict(coverage=1.0, risk=float(df.fail.mean()))
    json.dump(out, open(THR_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))
    synth_by_condition(s, out)


def synth_by_condition(s, thr):
    g = thr["fits"]["vlm9_a10"]["gate"]
    rows = []
    for (k, lv), d in s.groupby(["kind", "level"]):
        esc_gate = (d.conf < g["t_c"]) | (d.rho < g["r_min"])
        esc_conf = d.conf < thr["fits"]["vlm9_a10"]["conf"]["t_c"]
        rho_s = stats.spearmanr(d.conf, d.err_img.fillna(1e3)).correlation
        rows.append(dict(kind=k, level=lv, vlm_cond=bool(d.vlm_cond.iloc[0]), n=len(d),
                         ours_MAPE=float(np.nanmean(np.r_[d.ape_h, d.ape_v])),
                         ours_both_w10=float(100 * (~d.fail).mean()), conf_mean=float(d.conf.mean()),
                         rho_median=float(d.rho.median()), esc_gate_pct=float(100 * esc_gate.mean()),
                         esc_conf_pct=float(100 * esc_conf.mean()),
                         fail_caught_gate_pct=float(100 * esc_gate[d.fail].mean()) if d.fail.any() else np.nan,
                         fail_caught_conf_pct=float(100 * esc_conf[d.fail].mean()) if d.fail.any() else np.nan,
                         risk_kept_gate_pct=float(100 * d.fail[~esc_gate].mean()) if (~esc_gate).any() else np.nan,
                         spearman_conf_err=float(rho_s)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(TAB, "cascade_synth_by_condition.csv"), index=False)


# ----------------------------------------------------------------------------- legacy data
def load_legacy():
    import degradation_table as DT
    d = DT.load()
    d["cond"] = list(zip(d.kind, d.level.astype(float).round(4)))
    d = d[d.cond.isin(VLM_CONDS)]

    def pick(method, run="main"):
        x = d[(d.method == method) & (d.run == run)].set_index(["image", "kind", "level"])
        return x

    ours, gab = pick("ours"), pick("log-Gabor")
    m = pd.read_csv(os.path.join(RES, "legacy13_vlmcrops_degrade.csv"))
    m = m[m.estimator == "ours"].copy()
    m["level"] = m.level.astype(float).round(4)
    m = m.set_index(["image", "kind", "level"])
    L = pd.DataFrame(index=ours.index)
    L["gt_w"], L["gt_f"] = ours.gt_warp, ours.gt_weft
    L["ours_w"], L["ours_f"], L["conf"] = ours.warp_pred, ours.weft_pred, ours.conf
    L["gab_w"], L["gab_f"] = gab.warp_pred, gab.weft_pred
    L["sharp"], L["S_deg"], L["fusion"] = m.sharpness, m.S_deg, m.fusion
    assert np.allclose(L.ours_w, m.h.reindex(L.index)) and np.allclose(L.conf, m.conf.reindex(L.index))
    L["N_work"] = np.minimum(L.S_deg, 1024)
    for name, meth, run in (("gpt", "gpt-6.1-sol P1", "main"), ("gpt_r1", "gpt-6.1-sol P1", "repeat1"),
                            ("gpt_r2", "gpt-6.1-sol P1", "repeat2"), ("gp3", "gpt-6.1-sol P3", "main"),
                            ("gem", "gemini-3.8-flash P3", "main")):
        x = pick(meth, run)
        L[f"{name}_w"], L[f"{name}_f"] = x.warp_pred, x.weft_pred
        L[f"{name}_ok"] = x.ok.reindex(L.index).fillna(False).astype(bool)
        L[f"{name}_cost"] = x.cost_usd.reindex(L.index)
    L = L.reset_index()
    L["level"] = L.level.astype(float).round(4)
    L["cond"] = [COND_LABEL[(k, lv)] for k, lv in zip(L.kind, L.level)]
    L["rho"] = rho_of(L.ours_w, L.ours_f, L.sharp, L.N_work)
    L["d_ext"] = d_ext_of(L.ours_w, L.ours_f, L.gab_w, L.gab_f)
    L["lat_ok"] = lat_confirmed(L.fusion)
    assert len(L) == 117 and L.gpt_ok.all(), (len(L), L.gpt_ok.sum())
    return L


# ----------------------------------------------------------------------------- policies
def _vote(o, g, lg):
    """per axis: ours if within 10 % of the VLM, else the one of (ours, VLM) closer to log-Gabor."""
    agree = np.abs(o / g - 1) <= 0.10
    closer_o = np.abs(np.log(o / lg)) <= np.abs(np.log(g / lg))
    return np.where(agree | closer_o, o, g)


def policies(L, thr):
    F = thr["fits"]
    P = {}
    ow, of = L.ours_w.to_numpy(float), L.ours_f.to_numpy(float)

    def cascade(esc, vlm="gpt"):
        ok = L[f"{vlm}_ok"].to_numpy(bool)
        use = esc & ok  # a VLM abstention / parse failure falls back to ours (the call is still paid)
        w = np.where(use, L[f"{vlm}_w"], ow)
        f = np.where(use, L[f"{vlm}_f"], of)
        cost = np.where(esc, L[f"{vlm}_cost"].fillna(0), 0.0)
        return dict(w=w, f=f, esc=esc, cost=cost)

    n = len(L)
    P["ours only"] = dict(w=ow, f=of, esc=np.zeros(n, bool), cost=np.zeros(n))
    P["log-Gabor only"] = dict(w=L.gab_w.to_numpy(float), f=L.gab_f.to_numpy(float), esc=np.zeros(n, bool),
                               cost=np.zeros(n))
    P["GPT only"] = cascade(np.ones(n, bool))
    conf, rho, dext = L.conf.to_numpy(), L.rho.to_numpy(), L.d_ext.to_numpy()
    for scope in ("vlm9", "all19"):
        for a in [ALPHA] + ALPHAS_SENS:
            key = f"{scope}_a{int(round(100 * a)):02d}"
            tag = "" if key == "vlm9_a10" else f" [{scope}, a={a:.2f}]"
            g = F[key]["gate"]
            P[f"cascade gate{tag}"] = cascade((conf < g["t_c"]) | (rho < g["r_min"]))
            P[f"cascade conf{tag}"] = cascade(conf < F[key]["conf"]["t_c"])
            if key == "vlm9_a10":
                # the blur check alone cannot reach alpha on synthetic (clean failures are not blur
                # failures), so 'blur' = the blur component of the fitted gate on its own
                P["cascade blur (gate's r_min only)"] = cascade(rho < g["r_min"])
                P["cascade agree-ext"] = cascade(dext > F[key]["agree_ext"]["delta"])
    if "default_vlm9_a10" in F:
        g = F["default_vlm9_a10"]["gate"]
        P["cascade gate [default-generator synth]"] = cascade((conf < g["t_c"]) | (rho < g["r_min"]))
        P["cascade conf [default-generator synth]"] = cascade(conf < F["default_vlm9_a10"]["conf"]["t_c"])
    for pk in ("p05", "p02", "p10"):
        rr = F["real150_clean_rho_pct"][pk]
        P[f"cascade conf + blur(real-150 {pk})"] = cascade((conf < F["vlm9_a10"]["conf"]["t_c"]) | (rho < rr))
    P["cascade agree-int"] = cascade(L.lat_ok.to_numpy() < 1)
    P["cascade conf<0.9 (a priori)"] = cascade(conf < 0.9)
    g = F["vlm9_a10"]["gate"]
    esc_gate = (conf < g["t_c"]) | (rho < g["r_min"])
    P["cascade gate -> GPT P3"] = cascade(esc_gate, "gp3")
    P["cascade gate -> Gemini P3"] = cascade(esc_gate, "gem")
    gw, gf = L.gpt_w.to_numpy(float), L.gpt_f.to_numpy(float)
    lw, lf = L.gab_w.to_numpy(float), L.gab_f.to_numpy(float)
    allc = L.gpt_cost.to_numpy(float)
    P["both, median of 3"] = dict(w=np.median(np.c_[ow, lw, gw], 1), f=np.median(np.c_[of, lf, gf], 1),
                                  esc=np.ones(n, bool), cost=allc)
    P["both, vote"] = dict(w=_vote(ow, gw, lw), f=_vote(of, gf, lf), esc=np.ones(n, bool), cost=allc)
    e_o = 0.5 * (np.abs(ow / L.gt_w - 1) + np.abs(of / L.gt_f - 1))
    e_g = 0.5 * (np.abs(gw / L.gt_w - 1) + np.abs(gf / L.gt_f - 1))
    P["oracle router"] = cascade((e_g < e_o).to_numpy())
    # clean-only GPT repeats (run-to-run variability of the fallback)
    for r in ("gpt_r1", "gpt_r2"):
        x = L[f"{r}_w"].notna().to_numpy()
        esc = esc_gate & x
        w = np.where(esc, L[f"{r}_w"], ow)
        f = np.where(esc, L[f"{r}_f"], of)
        P[f"_gate clean, GPT {r[-2:]}"] = dict(w=w, f=f, esc=esc, cost=np.where(esc, L[f"{r}_cost"], 0.0),
                                               mask=x)
        P[f"_GPT only clean, {r[-2:]}"] = dict(w=L[f"{r}_w"].to_numpy(float), f=L[f"{r}_f"].to_numpy(float),
                                              esc=x, cost=np.where(x, L[f"{r}_cost"], 0.0), mask=x)
    return P


# ----------------------------------------------------------------------------- metrics
def ape_matrix(L, p):
    """(n, 2) absolute % errors."""
    return 100 * np.c_[np.abs(p["w"] / L.gt_w - 1), np.abs(p["f"] / L.gt_f - 1)]


def metrics_from(ape, conds, imgs, sel=None):
    """pooled MAPE, w10 %, worst-condition MAPE (+ which), per-condition MAPE on the rows of `sel`."""
    if sel is not None:
        ape, conds = ape[sel], conds[sel]
    out = {"MAPE": float(ape.mean()), "w10": float(100 * (ape <= 10).mean())}
    per = {c: float(ape[conds == c].mean()) for c in np.unique(conds)}
    wc = max(per, key=per.get)
    out.update(worst=per[wc], worst_cond=wc, per=per)
    return out


def boot_index(images, rng, nb=None):
    u = np.array(sorted(set(images)))
    idx = {im: np.flatnonzero(images == im) for im in u}
    draws = rng.choice(u, (nb or B, len(u)), replace=True)
    return [np.concatenate([idx[i] for i in row]) for row in draws]


def evaluate(L, P, thr):
    rng = np.random.default_rng(SEED)
    images, conds = L.image.to_numpy(), L.cond.to_numpy()
    BI = boot_index(images, rng)
    cond_list = [COND_LABEL[c] for c in VLM_CONDS]
    A = {k: ape_matrix(L, p) for k, p in P.items() if not k.startswith("_")}
    # bootstrap distributions (paired: same resamples for every policy)
    bootM = {k: np.empty(B) for k in A}
    bootW = {k: np.empty(B) for k in A}
    bootT = {k: np.empty(B) for k in A}
    for b, ix in enumerate(BI):
        cb = conds[ix]
        for k, a in A.items():
            ab = a[ix]
            bootM[k][b] = ab.mean()
            bootT[k][b] = 100 * (ab <= 10).mean()
            bootW[k][b] = max(ab[cb == c].mean() for c in cond_list)
    img_mean = {k: pd.Series(a.mean(1)).groupby(images).mean() for k, a in A.items()}
    rows, crow = [], []
    n = len(L)
    for k, p in P.items():
        if k.startswith("_"):
            continue
        m = metrics_from(A[k], conds, images)
        r = dict(policy=k, n_photos=n, n_counts=2 * n, MAPE=m["MAPE"],
                 MAPE_lo=np.percentile(bootM[k], 2.5), MAPE_hi=np.percentile(bootM[k], 97.5),
                 w10_pct=m["w10"], w10_lo=np.percentile(bootT[k], 2.5), w10_hi=np.percentile(bootT[k], 97.5),
                 worst_MAPE=m["worst"], worst_lo=np.percentile(bootW[k], 2.5),
                 worst_hi=np.percentile(bootW[k], 97.5), worst_cond=m["worst_cond"],
                 clean_MAPE=m["per"]["clean"], escalated_pct=100 * p["esc"].mean(),
                 offline_pct=100 * (1 - p["esc"].mean()), usd_per_1k=1000 * p["cost"].sum() / n)
        for ref in ("ours only", "GPT only"):
            tag = "ours" if ref == "ours only" else "gpt"
            dm, dw = bootM[k] - bootM[ref], bootW[k] - bootW[ref]
            r[f"dMAPE_vs_{tag}"] = m["MAPE"] - float(A[ref].mean())
            r[f"dMAPE_vs_{tag}_lo"], r[f"dMAPE_vs_{tag}_hi"] = np.percentile(dm, [2.5, 97.5])
            r[f"dworst_vs_{tag}"] = m["worst"] - metrics_from(A[ref], conds, images)["worst"]
            r[f"dworst_vs_{tag}_lo"], r[f"dworst_vs_{tag}_hi"] = np.percentile(dw, [2.5, 97.5])
            dimg = (img_mean[k] - img_mean[ref]).to_numpy()
            nz = np.abs(dimg) > 1e-9
            r[f"wilcoxon_p_vs_{tag}"] = (float(stats.wilcoxon(dimg[nz]).pvalue) if nz.sum() >= 5 else np.nan)
            r[f"n_img_better_vs_{tag}"] = int((dimg < -1e-9).sum())
            r[f"n_img_worse_vs_{tag}"] = int((dimg > 1e-9).sum())
        rows.append(r)
        for c in cond_list:
            sel = conds == c
            crow.append(dict(policy=k, cond=c, n_photos=int(sel.sum()), MAPE=float(A[k][sel].mean()),
                             w10_pct=float(100 * (A[k][sel] <= 10).mean()),
                             escalated_pct=float(100 * p["esc"][sel].mean()),
                             usd_per_1k=float(1000 * p["cost"][sel].sum() / sel.sum())))
    # clean-only repeats of the GPT fallback
    for k, p in P.items():
        if not k.startswith("_"):
            continue
        sel = p["mask"]
        a = ape_matrix(L, p)[sel]
        rows.append(dict(policy=k[1:], n_photos=int(sel.sum()), n_counts=int(2 * sel.sum()), MAPE=float(a.mean()),
                         w10_pct=float(100 * (a <= 10).mean()), clean_MAPE=float(a.mean()),
                         escalated_pct=float(100 * p["esc"][sel].mean()),
                         offline_pct=float(100 * (1 - p["esc"][sel].mean())),
                         usd_per_1k=float(1000 * p["cost"][sel].sum() / sel.sum())))
    T = pd.DataFrame(rows)
    C = pd.DataFrame(crow)
    return T, C


def sweeps(L, thr, n_rand=2000):
    """Escalate the k photos ranked least reliable by each signal, k = 0..n (vectorised)."""
    rng = np.random.default_rng(SEED + 1)
    conds = L.cond.to_numpy()
    cond_list = [COND_LABEL[c] for c in VLM_CONDS]
    ow, of, gw, gf = (L[c].to_numpy(float) for c in ("ours_w", "ours_f", "gpt_w", "gpt_f"))
    a_o = 100 * np.c_[np.abs(ow / L.gt_w - 1), np.abs(of / L.gt_f - 1)]
    a_g = 100 * np.c_[np.abs(gw / L.gt_w - 1), np.abs(gf / L.gt_f - 1)]
    D = a_g.mean(1) - a_o.mean(1)                                   # change of the row's mean APE
    Dw = (a_g <= 10).sum(1) - (a_o <= 10).sum(1)                    # change of counts within 10 %
    cost = L.gpt_cost.to_numpy(float)
    onehot = np.stack([conds == c for c in cond_list], 1).astype(float)  # (n, 9)
    nc = onehot.sum(0)
    base_c = np.array([a_o[conds == c].mean() for c in cond_list])
    g = thr["fits"]["vlm9_a10"]["gate"]
    conf, rho = L.conf.to_numpy(), L.rho.to_numpy()
    gate_key = np.where(rho < g["r_min"], -2 + rho / max(g["r_min"], 1e-12), conf)  # blur-flagged first
    orders = {"conf": np.argsort(conf, kind="stable"), "gate": np.argsort(gate_key, kind="stable"),
              "agree-ext": np.argsort(-L.d_ext.to_numpy(), kind="stable"),
              "oracle": np.argsort(D, kind="stable")}  # largest error reduction first
    n = len(L)

    def curve(o):
        z = np.zeros(1)
        mape = a_o.mean() + np.r_[z, np.cumsum(D[o])] / n
        per = base_c[None] + np.r_[np.zeros((1, 9)), np.cumsum(D[o, None] * onehot[o], 0)] / nc[None]
        w10 = 100 * ((a_o <= 10).sum() + np.r_[z, np.cumsum(Dw[o])]) / (2 * n)
        usd = 1000 * np.r_[z, np.cumsum(cost[o])] / n
        return usd, mape, per.max(1), w10

    rows = []
    k = np.arange(n + 1)
    for name, o in orders.items():
        usd, mape, worst, w10 = curve(o)
        rows += [dict(order=name, k=int(i), escalated_pct=100 * i / n, usd_per_1k=usd[i], MAPE=mape[i],
                      worst_MAPE=worst[i], w10_pct=w10[i]) for i in k]
    acc = [np.zeros(n + 1) for _ in range(4)]
    for _ in range(n_rand):
        for j, x in enumerate(curve(rng.permutation(n))):
            acc[j] += x / n_rand
    rows += [dict(order="random (mean)", k=int(i), escalated_pct=100 * i / n, usd_per_1k=acc[0][i], MAPE=acc[1][i],
                  worst_MAPE=acc[2][i], w10_pct=acc[3][i]) for i in k]
    return pd.DataFrame(rows)


def signal_quality(L, s):
    rows = []
    for setname, df, err, fail in (
            ("legacy13 x 9 conditions", L, 50 * (np.abs(L.ours_w / L.gt_w - 1) + np.abs(L.ours_f / L.gt_f - 1)),
             ~((np.abs(L.ours_w / L.gt_w - 1) <= .1) & (np.abs(L.ours_f / L.gt_f - 1) <= .1))),
            ("synthetic S-cal x 9 conditions", s[s.vlm_cond], s[s.vlm_cond].err_img.fillna(1e3),
             s[s.vlm_cond].fail),
            ("synthetic S-cal x 19 conditions", s, s.err_img.fillna(1e3), s.fail)):
        err, fail = np.asarray(err, float), np.asarray(fail, bool)
        groups = np.asarray(df.image if "image" in df else df.name)
        BI = boot_index(groups, np.random.default_rng(SEED + 2), nb=2000)
        for sig, x in (("conf", df.conf), ("rho (resolvability)", np.log(np.clip(df.rho, 1e-6, None))),
                       ("-d_ext (ours vs log-Gabor)", -np.clip(df.d_ext, 0, 1e3)),
                       ("lattice confirmed (agree-int)", df.lat_ok)):
            x = np.asarray(x, float)
            sp = stats.spearmanr(x, err).correlation
            auc = _auc(x, ~fail)
            bs = np.array([stats.spearmanr(x[ix], err[ix]).correlation for ix in BI])
            ba = np.array([_auc(x[ix], ~fail[ix]) for ix in BI])
            rows.append(dict(set=setname, signal=sig, n=len(x), n_images=len(set(groups)), spearman_vs_err=sp,
                             spearman_lo=np.nanpercentile(bs, 2.5), spearman_hi=np.nanpercentile(bs, 97.5),
                             auc_success=auc, auc_lo=np.nanpercentile(ba, 2.5), auc_hi=np.nanpercentile(ba, 97.5),
                             fail_rate=float(fail.mean())))
    return pd.DataFrame(rows)


def failure_catch(L, thr):
    """Legacy: ours' failures (not both counts within 10 %) per condition and how many each router escalates."""
    F = thr["fits"]["vlm9_a10"]
    g = F["gate"]
    fail = ~((np.abs(L.ours_w / L.gt_w - 1) <= .1) & (np.abs(L.ours_f / L.gt_f - 1) <= .1)).to_numpy()
    gross = ((np.abs(L.ours_w / L.gt_w - 1) > .25) | (np.abs(L.ours_f / L.gt_f - 1) > .25)).to_numpy()
    R = {"conf": L.conf < F["conf"]["t_c"], "gate conf part": L.conf < g["t_c"], "gate blur part": L.rho < g["r_min"],
         "gate": (L.conf < g["t_c"]) | (L.rho < g["r_min"]), "agree-ext": L.d_ext > F["agree_ext"]["delta"]}
    rows = []
    for c in ["all"] + [COND_LABEL[k] for k in VLM_CONDS]:
        sel = np.ones(len(L), bool) if c == "all" else (L.cond == c).to_numpy()
        r = dict(cond=c, n_photos=int(sel.sum()), ours_fail=int(fail[sel].sum()), ours_gross=int(gross[sel].sum()),
                 conf_mean=float(L.conf[sel].mean()), conf_mean_fail=float(L.conf[sel & fail].mean()) if (sel & fail).any() else np.nan)
        for k, e in R.items():
            e = np.asarray(e, bool)
            r[f"{k}: escalated"] = int(e[sel].sum())
            r[f"{k}: fail caught"] = int((e & fail)[sel].sum())
            r[f"{k}: gross caught"] = int((e & gross)[sel].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def _auc(score, pos):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return np.nan
    r = stats.rankdata(score)
    n1 = pos.sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (~pos).sum()))


def escalation_150(thr):
    """Label-free: share of the 150 real photos (clean and degraded) the gate would send to the VLM."""
    p = os.path.join(RES, "all150_degrade.csv")
    d = pd.read_csv(p)
    d = d[(d.estimator == "ours") & (d.crop == "gt")].copy()
    d["N_work"] = np.minimum(d.S_deg, 1024)
    d["rho"] = rho_of(d.h, d.v, d.sharpness, d.N_work)
    g = thr["fits"]["vlm9_a10"]["gate"]
    tc = thr["fits"]["vlm9_a10"]["conf"]["t_c"]
    d["esc_gate"] = (d.conf.fillna(0) < g["t_c"]) | (d.rho < g["r_min"])
    d["esc_blur_only"] = d.rho < g["r_min"]
    d["esc_conf"] = d.conf.fillna(0) < tc
    out = d.groupby(["kind", "level"]).agg(n=("image", "size"), esc_gate_pct=("esc_gate", "mean"),
                                           esc_blur_pct=("esc_blur_only", "mean"),
                                           esc_conf_pct=("esc_conf", "mean"), conf_mean=("conf", "mean"),
                                           rho_median=("rho", "median")).reset_index()
    for c in ("esc_gate_pct", "esc_blur_pct", "esc_conf_pct"):
        out[c] *= 100
    return out


def decisions_150(thr):
    """Frozen, label-free cascade decisions on the 150 clean real photos (pre-registration of the held-out
    test): join with the human counts when they arrive; GPT-6.1-sol P1 outputs already exist (no API cost)."""
    d = pd.read_csv(os.path.join(RES, "all150_degrade.csv"))
    d = d[(d.estimator == "ours") & (d.crop == "gt") & (d.kind == "clean")].copy()
    d["rho"] = rho_of(d.h, d.v, d.sharpness, np.minimum(d.S_deg, 1024))
    g = thr["fits"]["vlm9_a10"]["gate"]
    d["escalate_gate"] = (d.conf.fillna(0) < g["t_c"]) | (d.rho < g["r_min"])
    v = pd.read_csv(os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_long.csv"))
    v = v[v.model == "gpt-6.1-sol"][["image_id", "vertical", "horizontal", "parse_ok", "usd"]]
    v = v.rename(columns={"image_id": "image", "vertical": "gpt_v_yarns", "horizontal": "gpt_h_yarns",
                          "parse_ok": "gpt_ok", "usd": "gpt_usd"})
    d["image"] = d.image.astype(int)
    out = d[["image", "S_deg", "h", "v", "conf", "sharpness", "rho", "escalate_gate"]].merge(v, on="image", how="left")
    out = out.rename(columns={"h": "ours_v_yarns", "v": "ours_h_yarns"})
    use = out.escalate_gate & out.gpt_ok.fillna(False).astype(bool)
    # axis convention as everywhere: h = vertical yarns crossing a horizontal line, v = horizontal yarns
    out["cascade_v_yarns"] = np.where(use, out.gpt_v_yarns, out.ours_v_yarns)
    out["cascade_h_yarns"] = np.where(use, out.gpt_h_yarns, out.ours_h_yarns)
    out["t_c"], out["r_min"] = g["t_c"], g["r_min"]
    out.round(5).to_csv(os.path.join(TAB, "cascade_150_decisions.csv"), index=False)
    return out


# ----------------------------------------------------------------------------- figure
CAT = {"ours": "#2a78d6", "gpt": "#eb6834", "cnn": "#1baf7a", "gemini": "#4a3aa7"}
INK, INK2, GRID, CTX = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"


def figure(s, thr, SW, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    # (a) synthetic risk-coverage
    ax = axes[0]
    sv = s[s.vlm_cond]
    fail = sv.fail.to_numpy()
    g = thr["fits"]["vlm9_a10"]["gate"]
    gate_key = np.where(sv.rho < g["r_min"], -2 + sv.rho / max(g["r_min"], 1e-12), sv.conf)
    for key, lab, col, ls in ((sv.conf.to_numpy(), "confidence", CAT["ours"], "-"),
                              (np.log(np.clip(sv.rho.to_numpy(), 1e-6, None)), "resolvability ρ", CTX, "-"),
                              (gate_key, "gate (ρ, then conf.)", CAT["cnn"], "-"),
                              (-np.clip(sv.d_ext.to_numpy(), 0, 1e3), "ours vs log-Gabor", CAT["gemini"], ":")):
        o = np.argsort(-key, kind="stable")
        cov = np.arange(1, len(o) + 1) / len(o)
        risk = np.cumsum(fail[o]) / np.arange(1, len(o) + 1)
        ax.plot(100 * cov, 100 * risk, color=col, ls=ls, lw=1.4, label=lab)
    ax.axhline(100 * ALPHA, color=INK2, lw=0.7, ls=(0, (1, 2)))
    ax.plot(100 * g["coverage"], 100 * g["risk"], "o", color=CAT["cnn"], ms=4.5, mec="white", mew=0.6, zorder=5)
    ax.annotate("chosen gate", (100 * g["coverage"], 100 * g["risk"]), xytext=(4, -10), textcoords="offset points",
                fontsize=5.6, color=INK2)
    ax.text(16, 100 * ALPHA + 1, f"target risk {int(100 * ALPHA)}%", fontsize=5.6, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(60, 100 * fail.mean() + 5))
    ax.set_xlabel("Answered offline (% of synthetic crops)")
    ax.set_ylabel("Not both counts within 10% (%)")
    ax.set_title("(a) Synthetic: selective risk of ours", loc="left")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)

    # (b), (c) legacy cost-accuracy
    pts = [("ours only", "ours only", CAT["ours"], "o"), ("GPT only", "GPT-6.1-sol only", CAT["gpt"], "D"),
           ("cascade gate", "cascade: gate", CAT["cnn"], "*"), ("cascade conf", "cascade: conf.", CAT["ours"], "^"),
           ("cascade agree-ext", "cascade: agreement", CAT["gemini"], "v"),
           ("both, vote", "both + vote", INK2, "s"), ("oracle router", "oracle router", INK2, "P")]
    for ax, ycol, title, ylab in ((axes[1], "MAPE", "(b) Legacy, 9 conditions: pooled", "MAPE (%)"),
                                  (axes[2], "worst_MAPE", "(c) Legacy: worst condition", "Worst-condition MAPE (%)")):
        for order, lab, col, ls in (("gate", "gate order", CAT["cnn"], "-"), ("conf", "conf. order", CAT["ours"], "-"),
                                    ("random (mean)", "random order", CTX, "--"), ("oracle", "oracle order", INK2, ":")):
            c = SW[SW.order == order]
            ax.plot(c.usd_per_1k, c[ycol], color=col, ls=ls, lw=1.1 if order != "gate" else 1.5,
                    label=lab if ax is axes[1] else None, zorder=2)
        tcol = "MAPE" if ycol == "MAPE" else "worst_MAPE"
        for key, lab, col, mk in pts:
            r = T[T.policy == key].iloc[0]
            star, hollow = mk == "*", key.startswith("both")
            ax.plot(r.usd_per_1k, r[tcol], ls="none", marker=mk, ms=9 if star else 4.8, color=col,
                    mfc="white" if hollow else col, mec=INK if star else (col if hollow else "white"),
                    mew=0.9 if hollow else 0.6, zorder=6 if star else (4 if hollow else 5),
                    label=lab if ax is axes[1] else None)
        ax.set_yscale("log")
        ax.set_xlabel("API cost (US$ per 1,000 photos)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ticks = [5, 10, 20, 50, 100, 200]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lo = min(SW.MAPE.min(), T.MAPE.min()) * 0.8
    axes[1].set_ylim(lo, SW.MAPE.max() * 1.3)
    axes[2].set_ylim(min(SW.worst_MAPE.min(), T.worst_MAPE.min()) * 0.8, SW.worst_MAPE.max() * 1.3)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=6, frameon=False, handlelength=1.9,
               columnspacing=0.9, fontsize=6.0)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.15, top=0.80, wspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"cascade_tradeoff.{ext}"), dpi=300)
    plt.close(fig)


def latex_table(T):
    """Compact main-paper table (tables/cascade_main.tex), generated from cascade_legacy13.csv."""
    rows = [("ours only", r"\ours only"), ("GPT only", "GPT-6.1-sol P1 only"),
            ("cascade conf", r"Cascade: confidence $<t_c$"),
            ("cascade gate", r"Cascade: confidence or resolvability (gate)"),
            ("cascade agree-ext", r"Cascade: \ours vs log-Gabor disagree"),
            ("both, median of 3", r"Always both: median of 3"),
            ("oracle router", r"Oracle router (labels)")]
    out = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
           r"Policy & MAPE (\%) & $\le$10\% & Worst cond. & Offline & US\$/1k \\", r"\midrule"]
    for key, lab in rows:
        r = T[T.policy == key].iloc[0]
        out.append(f"{lab} & {r.MAPE:.1f} [{r.MAPE_lo:.1f}, {r.MAPE_hi:.1f}] & {r.w10_pct:.0f} & "
                   f"{r.worst_MAPE:.1f} & {r.offline_pct:.0f}\\% & {r.usd_per_1k:.2f} \\\\")
        if key in ("GPT only", "cascade agree-ext"):
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "cascade_main.tex"), "w").write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- main
def run_eval():
    thr = json.load(open(THR_JSON))
    s = load_synth()
    L = load_legacy()
    P = policies(L, thr)
    T, C = evaluate(L, P, thr)
    SW = sweeps(L, thr)
    SQ = signal_quality(L, s)
    E = escalation_150(thr)
    FC = failure_catch(L, thr)
    D150 = decisions_150(thr)
    print(f"150 clean photos: gate escalates {D150.escalate_gate.sum()}/{len(D150)}; "
          f"US$/1k = {1000 * D150.gpt_usd[D150.escalate_gate].sum() / len(D150):.2f}")
    FC.round(4).to_csv(os.path.join(TAB, "cascade_failure_catch.csv"), index=False)
    T.round(4).to_csv(os.path.join(TAB, "cascade_legacy13.csv"), index=False)
    C.round(4).to_csv(os.path.join(TAB, "cascade_legacy13_by_condition.csv"), index=False)
    SW.round(4).to_csv(os.path.join(TAB, "cascade_sweep.csv"), index=False)
    SQ.round(4).to_csv(os.path.join(TAB, "cascade_signal_quality.csv"), index=False)
    E.round(4).to_csv(os.path.join(TAB, "cascade_150_escalation.csv"), index=False)
    keep = ["image", "kind", "level", "cond", "gt_w", "gt_f", "ours_w", "ours_f", "conf", "rho", "d_ext", "lat_ok",
            "gab_w", "gab_f", "gpt_w", "gpt_f", "gpt_cost", "sharp", "N_work"]
    L[keep].round(5).to_csv(os.path.join(TAB, "cascade_legacy13_inputs.csv"), index=False)
    figure(s, thr, SW, T)
    latex_table(T)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["policy", "MAPE", "MAPE_lo", "MAPE_hi", "w10_pct", "worst_MAPE", "worst_cond", "clean_MAPE",
            "escalated_pct", "usd_per_1k", "dMAPE_vs_ours", "dMAPE_vs_ours_lo", "dMAPE_vs_ours_hi",
            "wilcoxon_p_vs_ours", "dMAPE_vs_gpt", "dMAPE_vs_gpt_lo", "dMAPE_vs_gpt_hi", "wilcoxon_p_vs_gpt",
            "dworst_vs_gpt", "dworst_vs_gpt_lo", "dworst_vs_gpt_hi"]
    print(T[cols].round(2).to_string(index=False))
    print(C.pivot(index="policy", columns="cond", values="MAPE").round(1).to_string())
    print(C.pivot(index="policy", columns="cond", values="escalated_pct").round(0).to_string())
    print(SQ.round(3).to_string(index=False))
    print(E.round(1).to_string(index=False))
    print(FC.round(2).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit(a)
    if a.eval:
        run_eval()

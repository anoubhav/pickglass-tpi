"""combined_frozen_v1.py -- FROZEN combined offline counter "combined-v1": PG-Lattice ("ours") + log-Gabor bank.

Pre-registered on 2026-10-11/12 (US Eastern), BEFORE any human count of the 76 PickGlass scenes existed;
see methods/results/PREREG_combined_v1.json (rule, thresholds, objective, synthetic data, SHA-256 of this
file and of every input).  Thresholds were chosen on SYNTHETIC crops with exact labels only (S-cal, both
generator variants: methods/results/cascade_synth_cal_{legacygrid,default}.csv) by methods/combined_v1.py --fit.
The frozen counter (methods/results/counting_frozen_final.py) is only imported, never modified.

Inputs per photo (all label-free outputs of the frozen counter on the SAME pixels):
  ours_h, ours_v      estimate(..., "ours")   counts per crop side (h = vertical yarns crossing a horizontal line)
  gabor_h, gabor_v    estimate(..., "gabor")  log-Gabor bank, same pre-processing (lcn, glare mask)
  sharpness           ours' debug["sharpness"] = var(Laplacian) / var(image) of the working image
  n_work              working side = min(max(H, W), 1024) (counting_frozen_final.preprocess)

Rule (per photo, then per axis a in {h, v}); two thresholds R_MIN, D_AGREE:
  rho = sharpness / (4 sin^2(pi f))^2,  f = max(ours_h, ours_v) / n_work    (the resolvability ratio of
        paper/tools/cascade_tradeoff.py rho_of; rho = 0 when ours returned no count)
  1. defocus fallback:  rho < R_MIN                       -> log-Gabor on both axes
  2. agreement:         |ln(ours_a / gabor_a)| <= D_AGREE  -> geometric mean sqrt(ours_a * gabor_a)
  3. otherwise                                           -> ours_a
  A non-finite component is replaced by the other one; both non-finite -> NaN.
Deterministic, label-free at inference, no state.

Fitted values (combined_v1.py --fit, 2026-10-11T22:18 -0400, objective declared before fitting: maximise
P5 = share of synthetic counts within 5 % of the exact label, ties -> lower MAPE):
  PRIMARY   R_MIN = 0 (the defocus fallback never fires: on synthetic data every R_MIN > 0 LOWERED P5, because
            the generator's log-Gabor under-counts by 60-70 % at defocus >= 4u while ours stays usable),
            D_AGREE = 0.18 (average the two counters when they agree within a factor e^0.18 = 1.197).
            S-cal (400 crops x 19 conditions, 15 200 counts): P5 75.7 % vs ours 72.9 %, log-Gabor 61.7 %.
  SECONDARY R_MIN_SEC = 0.0011375, D_AGREE_SEC = 0.55 (objective: minimise MAPE, ties -> higher P5): keeps the
            defocus fallback (fires on 6.8 % of S-cal rows); MAPE 15.7 % vs ours 18.1 %, but P5 73.2 %.
            Sensitivity analysis only, not the pre-registered primary.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np

RULE_ID = "combined-v1"
RULE_ID_SEC = "combined-v1-mape"

# ---- frozen thresholds (filled from methods/results/combined_v1_fit.json; do not edit after PREREG) ----
R_MIN = 0.0         # defocus fallback: log-Gabor on both axes iff rho < R_MIN   (0 = never: the P5 fit switched it off)
D_AGREE = 0.18      # geometric mean on an axis iff |ln(ours/gabor)| <= D_AGREE  (ratio within [0.8353, 1.1972])
R_MIN_SEC = 0.0011375274142385549    # secondary (MAPE-objective) variant: fallback fires on 6.75% of S-cal rows
D_AGREE_SEC = 0.55

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.dirname(HERE)
FROZEN_COUNTER = os.path.join(HERE, "counting_frozen_final.py")
CONF_MODEL_JSON = os.path.join(METHODS, "conf_model.json")
N_WORK_MAX = 1024


def _thresholds(variant):
    r, d = (R_MIN, D_AGREE) if variant == "primary" else (R_MIN_SEC, D_AGREE_SEC)
    if r is None or d is None:
        raise RuntimeError("combined_frozen_v1: thresholds not filled in (run combined_v1.py --fit first)")
    return float(r), float(d)


def rho_of(ours_h, ours_v, sharpness, n_work):
    """Resolvability ratio (identical to paper/tools/cascade_tradeoff.py rho_of)."""
    h, v, s, n = (np.asarray(x, float) for x in (ours_h, ours_v, sharpness, n_work))
    f = np.fmax(h, v) / n
    gain = (4 * np.sin(np.pi * np.clip(f, 1e-6, 0.5)) ** 2) ** 2
    r = s / gain
    return np.where(np.isfinite(r), r, 0.0)


def combine_arrays(ours_h, ours_v, gabor_h, gabor_v, sharpness, n_work, r_min=None, d_agree=None,
                   variant="primary"):
    """Vectorised rule. Returns dict of arrays: h, v, rho, src_h, src_v (sources: 'gabor-fallback',
    'mean', 'ours', 'gabor-only', 'ours-only', 'none')."""
    if r_min is None or d_agree is None:
        r_min, d_agree = _thresholds(variant)
    oh, ov, gh, gv = (np.asarray(x, float) for x in (ours_h, ours_v, gabor_h, gabor_v))
    rho = rho_of(oh, ov, sharpness, n_work)
    fb = rho < r_min
    out = {"rho": rho, "fallback": fb}
    for ax, o, g in (("h", oh, gh), ("v", ov, gv)):
        fo, fg = np.isfinite(o) & (o > 0), np.isfinite(g) & (g > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(fo & fg, np.log(np.where(fo, o, 1.0) / np.where(fg, g, 1.0)), np.inf)
            gm = np.sqrt(np.where(fo, o, np.nan) * np.where(fg, g, np.nan))
        agree = np.abs(lr) <= d_agree
        val = np.where(fb & fg, g, np.where(agree, gm, np.where(fo, o, np.where(fg, g, np.nan))))
        src = np.where(fb & fg, "gabor-fallback",
                       np.where(agree, "mean", np.where(fo & fg, "ours",
                                                         np.where(fo, "ours-only", np.where(fg, "gabor-only", "none")))))
        out[ax], out["src_" + ax] = val, src
    return out


def combine(ours_h, ours_v, gabor_h, gabor_v, sharpness, n_work, variant="primary"):
    """Scalar convenience wrapper -> dict(h, v, rho, src_h, src_v, rule)."""
    r = combine_arrays([ours_h], [ours_v], [gabor_h], [gabor_v], [sharpness], [n_work], variant=variant)
    return {"h": float(r["h"][0]), "v": float(r["v"][0]), "rho": float(r["rho"][0]),
            "src_h": str(r["src_h"][0]), "src_v": str(r["src_v"][0]),
            "rule": RULE_ID if variant == "primary" else RULE_ID_SEC}


_C = None


def load_frozen_counter():
    """methods/results/counting_frozen_final.py imported read-only by path; its confidence model is the
    methods/conf_model.json every study used (injected, as methods/run_legacy_vlmcrops.py does)."""
    global _C
    if _C is None:
        spec = importlib.util.spec_from_file_location("counting_frozen_final", FROZEN_COUNTER)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with open(CONF_MODEL_JSON) as f:
            m.CONF_MODEL = json.load(f)
        _C = m
    return _C


def estimate(crop_bgr, variant="primary"):
    """End-to-end: frozen counter (ours + log-Gabor, preproc lcn, glare mask) on one rectified crop -> rule."""
    C = load_frozen_counter()
    res = C.estimate_many(crop_bgr, ("ours", "gabor"), preproc="lcn", glare="mask")
    o, g = res["ours"], res["gabor"]
    n_work = int(min(max(crop_bgr.shape[:2]), N_WORK_MAX))
    out = combine(o["h_count"], o["v_count"], g["h_count"], g["v_count"], o["debug"].get("sharpness", math.nan),
                  n_work, variant=variant)
    out.update(ours_h=o["h_count"], ours_v=o["v_count"], gabor_h=g["h_count"], gabor_v=g["v_count"],
               ours_conf=o["confidence"], sharpness=o["debug"].get("sharpness"), n_work=n_work)
    return out

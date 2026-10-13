import pandas as pd, numpy as np, sys
from synthan import load, cat
from consist import consistency, MAN

def decide_single(r, rule, p):
    """returns (fx, fy) from frozen-output features in row r (dict)."""
    fx = fy = 1.0
    if rule in ("sym", "sym+ridge", "sym+H", "all"):
        Dx, Dy, Cc = r.get("Dx"), r.get("Dy"), r.get("C")
        if np.isfinite(Dx) and Dx > p["td"]: fx = 2
        if np.isfinite(Dy) and Dy > p["td"]: fy = 2
        if p.get("tc") is not None and np.isfinite(Cc) and Cc > p["tc"] and fx == 1 and fy == 1: fx = fy = 2
    if rule in ("ridge", "sym+ridge", "all"):
        if fx == 1 and np.isfinite(r.get("xr2", np.nan)) and r["xr2"] >= p["rho"] and r.get("xd2", 0) >= 0.35: fx = 2
        if fy == 1 and np.isfinite(r.get("yr2", np.nan)) and r["yr2"] >= p["rho"] and r.get("yd2", 0) >= 0.35: fy = 2
    if rule in ("sym+H", "all", "H") and p.get("th") is not None:
        if fx == 1 and np.isfinite(r.get("Hx", np.nan)) and r["Hx"] < p["th"]: fx = 0.5
        if fy == 1 and np.isfinite(r.get("Hy", np.nan)) and r["Hy"] < p["th"]: fy = 0.5
    if rule in ("RH", "all") and p.get("rh") is not None:
        if fx == 1 and np.isfinite(r.get("xr2h", np.nan)) and r["xr2h"] < p["rh"]: fx = 0.5
        if fy == 1 and np.isfinite(r.get("yr2h", np.nan)) and r["yr2h"] < p["rh"]: fy = 0.5
    return fx, fy

def feats_synth(d, wt="snr"):
    out = []
    for r in d.to_dict("records"):
        out.append(dict(Dx=r.get(f"o_{wt}_Dx", np.nan), Dy=r.get(f"o_{wt}_Dy", np.nan), C=r.get(f"o_{wt}_C", np.nan),
                        Hx=r.get(f"o_{wt}_Hx", np.nan), Hy=r.get(f"o_{wt}_Hy", np.nan),
                        xr2=r.get("o_x_r2", np.nan), xd2=r.get("o_x_d2", np.nan), yr2=r.get("o_y_r2", np.nan), yd2=r.get("o_y_d2", np.nan),
                        xr2h=r.get("o_x_r2h", np.nan), yr2h=r.get("o_y_r2h", np.nan)))
    return out

def eval_synth(d, F, rule, p):
    fs = np.array([decide_single(f, rule, p) for f in F])
    h = d.ours_h.values * fs[:, 0]; v = d.ours_v.values * fs[:, 1]
    res = {}
    for s in ("A", "B"):
        m = (d.set == s).values
        r = np.r_[h[m] / d.true_h.values[m], v[m] / d.true_v.values[m]]
        c = cat(r)
        res[s] = dict(mape=100 * np.mean(np.abs(r - 1)), ok=np.mean(c == "ok"), half=int((c == "half").sum()), dbl=int((c == "dbl").sum()),
                      changed=int((fs[m] != 1).sum()))
    return res

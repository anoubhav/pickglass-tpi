import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from synthan import load, cat
from consist import consistency, MAN
R_MIN = 18.0

def factors(r, ax, rule, p, cnt, hi):
    r2 = r.get(f"o_{ax}_r2", np.nan); r2h = r.get(f"o_{ax}_r2h", np.nan)
    c = json.loads(r.get(f"o_{ax}_c") or "null")
    f = 1.0
    if "D" in rule:
        ok = np.isfinite(r2) and r2 >= p["rho"]
        if "K" in rule and c:
            ok = ok or (c[1] / max(c[0], 1e-9) >= p["kappa"])
        if ok and 2 * cnt <= hi:
            f = 2.0
    if "H" in rule and f == 1.0:
        if np.isfinite(r2h) and r2h < p["rh"] and cnt / 2 >= R_MIN:
            f = 0.5
    return f

def apply(df, rule, p):
    fx = []; fy = []
    for r in df.to_dict("records"):
        hi = r.get("hi", np.nan)
        if not np.isfinite(hi):
            hi = 0.42 * min(1024, r.get("N", 1024)) if "N" in r else 430
        fx.append(factors(r, "x", rule, p, r["ours_h"], hi)); fy.append(factors(r, "y", rule, p, r["ours_v"], hi))
    return np.array(fx), np.array(fy)

def synth_metrics(d, fx, fy):
    out = {}
    for s in ("A", "B"):
        m = (d.set == s).values
        rr = np.r_[d.ours_h.values[m] * fx[m] / d.true_h.values[m], d.ours_v.values[m] * fy[m] / d.true_v.values[m]]
        c = cat(rr)
        out[s] = dict(mape=round(100 * np.mean(np.abs(rr - 1)), 2), w10=round(100 * np.mean(c == "ok"), 1), half=int((c == "half").sum()),
                      dbl=int((c == "dbl").sum()), changed=int((fx[m] != 1).sum() + (fy[m] != 1).sum()))
    return out

def legacy_metrics(L, fx, fy):
    h = L.ours_h.values * fx; v = L.ours_v.values * fy
    ape = 100 * np.r_[np.abs(h / L.true_h.values - 1), np.abs(v / L.true_v.values - 1)]
    return dict(mape=round(ape.mean(), 2), changed=int((fx != 1).sum() + (fy != 1).sum()))

def real_metrics(Rl, fx, fy):
    x = Rl[["key"]].copy(); x["id"] = x.key.astype(int)
    x["h"] = Rl.ours_h.values * fx; x["v"] = Rl.ours_v.values * fy
    s, _ = consistency(x.fillna(1.0), "h", "v")
    reps = set(MAN["scenes"]["reps"]); m = x.id.isin(reps).values
    return dict(reshoot_oct=int(s.loc["reshoot", "oct"]), res_oct=int(s.loc["res", "oct"]), rot_oct=int(s.loc["rot", "oct"]),
                copy_oct=int(s.loc["copy", "oct"]), reshoot_ag5=round(float(s.loc["reshoot", "agree5"]), 3),
                res_ag5=round(float(s.loc["res", "agree5"]), 3), rot_ag5=round(float(s.loc["rot", "agree5"]), 3),
                rep_up=int((fx[m] > 1).sum() + (fy[m] > 1).sum()), rep_down=int((fx[m] < 1).sum() + (fy[m] < 1).sum()))

if __name__ == "__main__":
    d = load(); L = pd.read_pickle("ridge_legacy.pkl"); Rl = pd.read_pickle("ridge_real.pkl")
    L["hi"] = 0.42 * L.N; Rl["hi"] = 0.42 * Rl.N
    grid = [("none", {})]
    for rho in (0.05, 0.1, 0.15, 0.2, 0.3):
        grid.append(("D", dict(rho=rho)))
    for rho in (0.1, 0.2):
        for kappa in (0.35, 0.5, 0.7):
            grid.append(("DK", dict(rho=rho, kappa=kappa)))
    for rh in (0.03, 0.05, 0.1):
        grid.append(("H", dict(rh=rh)))
        grid.append(("DH", dict(rho=0.1, rh=rh)))
    rows = []
    for rule, p in grid:
        fx, fy = apply(d, rule, p); sm = synth_metrics(d, fx, fy)
        lx, ly = apply(L, rule, p); lm = legacy_metrics(L, lx, ly)
        rx, ry = apply(Rl, rule, p); rm = real_metrics(Rl, rx, ry)
        rows.append(dict(rule=rule, **{k: v for k, v in p.items()}, A_mape=sm["A"]["mape"], A_w10=sm["A"]["w10"], A_half=sm["A"]["half"], A_dbl=sm["A"]["dbl"],
                         B_mape=sm["B"]["mape"], B_w10=sm["B"]["w10"], B_half=sm["B"]["half"], B_dbl=sm["B"]["dbl"],
                         leg_mape=lm["mape"], leg_chg=lm["changed"], **rm))
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(rows).to_string())

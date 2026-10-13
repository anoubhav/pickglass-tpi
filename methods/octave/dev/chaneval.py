import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from synthan import cat
from consist import consistency, MAN
CH = ("pca", "b", "g", "r")
def r2eff(row, tag, ax, chs, which="r2"):
    vals = []
    for c in chs:
        r2 = row.get(f"{tag}_{ax}_{c}_{which}", np.nan); d2 = row.get(f"{tag}_{ax}_{c}_d2", np.nan)
        if which == "r2":
            if np.isfinite(r2) and not (np.isfinite(d2) and d2 >= 0.35): r2 = 0.0
        vals.append(r2)
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else np.nan
def fac(row, ax, cnt, N, rho, chs, rh=None, chs_h=("pca",)):
    hi = 0.42 * N
    r2 = r2eff(row, "o", ax, chs)
    if np.isfinite(r2) and r2 >= rho and 2 * cnt <= hi: return 2.0
    if rh is not None:
        vals = [row.get(f"o_{ax}_{c}_r2h", np.nan) for c in chs_h]
        if all(np.isfinite(vals)) and max(vals) < rh and cnt / 2 >= 18: return 0.5
    return 1.0
def apply(df, rho, chs, rh=None, chs_h=("pca",)):
    fx = np.array([fac(r, "x", r["ours_h"], r["N"], rho, chs, rh, chs_h) for r in df.to_dict("records")])
    fy = np.array([fac(r, "y", r["ours_v"], r["N"], rho, chs, rh, chs_h) for r in df.to_dict("records")])
    return fx, fy
if __name__ == "__main__":
    d = pd.read_pickle("chan_synth.pkl"); lab = pd.read_csv("synth/labels.csv"); d = d.merge(lab[["file", "set", "weave", "colour_style"]], left_on="key", right_on="file")
    L = pd.read_pickle("chan_legacy.pkl"); R = pd.read_pickle("chan_real.pkl"); R["id"] = R.key.astype(int)
    reps = set(MAN["scenes"]["reps"])
    rows = []
    for chs_name, chs in (("pca", ("pca",)), ("all4", CH)):
        for rho in (0.05, 0.1, 0.15, 0.2, 0.3):
            for rh, chs_h_name in ((None, "-"), (0.05, "pca"), (0.05, "all4")):
                chs_h = ("pca",) if chs_h_name == "pca" else CH
                fx, fy = apply(d, rho, chs, rh, chs_h)
                out = dict(rep=chs_name, rho=rho, rh=rh, rep_h=chs_h_name)
                for s in ("A", "B"):
                    m = (d.set == s).values
                    rr = np.r_[d.ours_h.values[m] * fx[m] / d.true_h.values[m], d.ours_v.values[m] * fy[m] / d.true_v.values[m]]
                    c = cat(rr)
                    out.update({f"{s}_mape": round(100 * np.mean(np.abs(rr - 1)), 2), f"{s}_w10": round(100 * np.mean(c == "ok"), 1),
                                f"{s}_half": int((c == "half").sum()), f"{s}_dbl": int((c == "dbl").sum())})
                lx, ly = apply(L, rho, chs, rh, chs_h)
                ape = 100 * np.r_[np.abs(L.ours_h * lx / L.true_h - 1), np.abs(L.ours_v * ly / L.true_v - 1)]
                out.update(leg_mape=round(ape.mean(), 2), leg_chg=int((lx != 1).sum() + (ly != 1).sum()))
                rx, ry = apply(R, rho, chs, rh, chs_h)
                x = R[["id"]].copy(); x["h"] = R.ours_h * rx; x["v"] = R.ours_v * ry
                s_, _ = consistency(x.fillna(1.0), "h", "v")
                m = R.id.isin(reps).values
                out.update(reshoot_oct=int(s_.loc["reshoot", "oct"]), res_oct=int(s_.loc["res", "oct"]), rot_oct=int(s_.loc["rot", "oct"]),
                           copy_oct=int(s_.loc["copy", "oct"]), reshoot_ag5=round(float(s_.loc["reshoot", "agree5"]), 3),
                           res_ag5=round(float(s_.loc["res", "agree5"]), 3), rot_ag5=round(float(s_.loc["rot", "agree5"]), 3),
                           rep_up=int((rx[m] > 1).sum() + (ry[m] > 1).sum()), rep_down=int((rx[m] < 1).sum() + (ry[m] < 1).sum()))
                rows.append(out)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    T = pd.DataFrame(rows); print(T.to_string()); T.to_csv("chaneval_grid.csv", index=False)
    # frozen baseline
    x = R[["id"]].copy(); x["h"] = R.ours_h; x["v"] = R.ours_v
    s_, _ = consistency(x.fillna(1.0), "h", "v"); print(s_)

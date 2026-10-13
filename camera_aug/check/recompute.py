"""Independent recomputation of headline numbers from the raw CSVs (read-only)."""
import json, os
import numpy as np, pandas as pd
H = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "camera_aug")
R = os.path.join(H, "results")
pd.set_option("display.width", 200)

# ---------- (A) legacy-13 MAPE from raw predictions + labels.csv ----------
lab = pd.read_csv(os.path.join(H, "stress/legacy13/labels.csv"), dtype={"source_id": str})
pc = pd.read_csv(os.path.join(R, "stress_legacy13_classical.csv"), dtype={"source_id": str})
cn = pd.read_csv(os.path.join(R, "stress_legacy13_cnn.csv"), dtype={"source_id": str})
pr = pd.concat([pc[["source_id", "condition", "rep", "estimator", "h", "v"]], cn[["source_id", "condition", "rep", "estimator", "h", "v"]]])
src = lab.drop_duplicates("source_id").set_index("source_id")[["h_src", "v_src"]]
rows = []
for (est, cond), g in pr.groupby(["estimator", "condition"]):
    if cond == "clean":
        m = g.merge(src, left_on="source_id", right_index=True)
        m["hl"], m["vl"] = m.h_src, m.v_src
    else:
        m = g.merge(lab[["source_id", "condition", "rep", "h_label", "v_label"]], on=["source_id", "condition", "rep"])
        m["hl"], m["vl"] = m.h_label, m.v_label
    long = pd.concat([pd.DataFrame(dict(sid=m.source_id, ax="h", ape=100 * (m.h - m.hl).abs() / m.hl)),
                      pd.DataFrame(dict(sid=m.source_id, ax="v", ape=100 * (m.v - m.vl).abs() / m.vl))])
    per = long.groupby(["sid", "ax"]).ape.mean()          # mean over reps per count
    w10 = (long.ape <= 10).mean() * 100                    # over count x rep
    rows.append(dict(est=est, cond=cond, n_counts=len(per), n_rows=len(long), MAPE=per.mean(), w10_rows=w10,
                     w10_permean=(per <= 10).mean() * 100))
A = pd.DataFrame(rows)
clean = A[A.cond == "clean"].set_index("est").MAPE
A["dMAPE"] = A.MAPE - A.est.map(clean)
print("(A) legacy-13 recomputed:")
print(A[A.cond.isin(["clean", "budget_binned_12mp", "budget_fullres_50mp", "midrange_binned", "flagship",
                     "legacy_16mp_2019", "budget_lowlight", "budget+gblur6", "budget+glare0.3"])].round(2).to_string(index=False))
tab = pd.read_csv(os.path.join(R, "stress_legacy13_metrics.csv"))
mm = A.merge(tab[["estimator", "condition", "MAPE", "dMAPE", "w10_pct"]], left_on=["est", "cond"], right_on=["estimator", "condition"], suffixes=("", "_tab"))
print("max |MAPE - table| =", (mm.MAPE - mm.MAPE_tab).abs().max(), " max |w10 - table| (rows) =", (mm.w10_rows - mm.w10_pct).abs().max(),
      " (per-mean) =", (mm.w10_permean - mm.w10_pct).abs().max())
cam5 = ["budget_binned_12mp", "budget_fullres_50mp", "midrange_binned", "flagship", "legacy_16mp_2019"]
o = A[(A.est == "ours") & A.cond.isin(cam5)]
print("ours w10 range over 5 profiles (rows):", o.w10_rows.min(), o.w10_rows.max(), " permean:", o.w10_permean.min(), o.w10_permean.max())

# ---------- (B) real150: gross flips over 8 camera conditions + confidence quartiles ----------
SYN = {131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144}
raw = pd.read_csv(os.path.join(R, "audit_real150_raw.csv"), dtype={"source_id": str})
cl = raw[raw.cond == "clean"].set_index("source_id")
s = pd.read_csv(os.path.join(R, "stress_real150_classical.csv"), dtype={"source_id": str})
man = pd.read_csv(os.path.join(H, "stress/real150/manifest.csv"), dtype={"source_id": str})
CAM8 = cam5 + ["budget_lowlight", "budget_whatsapp", "legacy_quarter"]
s = s[(s.estimator == "ours") & s.condition.isin(CAM8)].merge(
    man[["source_id", "condition", "S_src", "S_out", "label_factor_h", "label_factor_v"]], on=["source_id", "condition"])
L = []
for ax, f in (("h", "label_factor_h"), ("v", "label_factor_v")):
    t = pd.DataFrame(dict(sid=s.source_id, cond=s.condition, obs=s[ax], fac=s[f], S_out=s.S_out, S_src=s.S_src))
    t["clean"] = t.sid.map(cl[ax]); t["conf"] = t.sid.map(cl["conf"])
    t["pred"] = t.fac * t.clean
    t["drift"] = t.obs / t.pred - 1
    t.loc[~np.isfinite(t.obs), "drift"] = 1.0
    t["app"] = (t.pred >= 19.8) & (t.pred <= 0.4 * np.minimum(t.S_out, 1024)) & (t.clean <= 0.4 * np.minimum(t.S_src, 1024)) & np.isfinite(t.clean)
    L.append(t)
x = pd.concat(L)
x = x[~x.sid.astype(int).isin(SYN)]
print("\n(B) real150, 8 camera conditions: rows", len(x), "applicable", int(x.app.sum()), "not applicable", int((~x.app).sum()))
xa = x[x.app]
print("   gross >10%% overall: %.1f %%" % (100 * (xa.drift.abs() > .1).mean()))
print("   per condition:", (xa.groupby("cond").drift.apply(lambda d: round(100 * (d.abs() > .1).mean(), 1))).to_dict())
print("   median abs drift per cond:", (xa.groupby("cond").drift.apply(lambda d: round(100 * d.abs().median(), 2))).to_dict())
print("   not-applicable per cond:", (~x.app).groupby(x.cond).sum().to_dict())
# confidence quartiles of the clean conf (per image), quartiles over images or counts
q = pd.qcut(xa.conf, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("   gross by clean-conf quartile (quartiles over applicable counts):", xa.groupby(q, observed=True).drift.apply(lambda d: round(100 * (d.abs() > .1).mean(), 1)).to_dict(),
      " Q4 lower edge:", round(xa.conf.quantile(.75), 3))
imgconf = cl.loc[[i for i in cl.index if int(i) not in SYN], "conf"]
qi = pd.qcut(imgconf, 4, labels=["Q1", "Q2", "Q3", "Q4"])
xa2 = xa.assign(qi=xa.sid.map(qi))
print("   gross by clean-conf quartile (quartiles over images):", xa2.groupby("qi", observed=True).drift.apply(lambda d: round(100 * (d.abs() > .1).mean(), 1)).to_dict())
fl = xa[xa.drift.abs() > .1]
r = (1 + fl.drift)
harm = lambda lo, hi: ((r > lo) & (r < hi))
print("   flips near x0.5/x2: %.0f %%, near x0.67/x1.5: %.0f %%" % (100 * (harm(.45, .55) | harm(1.85, 2.15)).mean(), 100 * (harm(.6, .72) | harm(1.4, 1.6)).mean()))

# ---------- (C) real full/quarter pairs vs matched simulation ----------
P = pd.read_csv(os.path.join(R, "label_audit_pairs.csv"))
a = P.drift.abs(); b = P["sim_iso_resave_q75_x0.25"].abs().dropna()
print("\n(C) real pairs: n=%d counts, %d pairs, %d scenes; median %.2f%%, p90 %.2f%%, >10%% %.1f%%" % (len(P), P[["full_res", "quarter_res"]].drop_duplicates().shape[0], P.scene.nunique(), 100 * a.median(), 100 * a.quantile(.9), 100 * (a > .1).mean()))
print("    sim iso q75x0.25: n=%d; median %.2f%%, p90 %.2f%%, >10%% %.1f%%; S_sim median %.0f" % (len(b), 100 * b.median(), 100 * b.quantile(.9), 100 * (b > .1).mean(), P.S_sim.median()))
print("    real S_quarter median", P.S_quarter.median(), "; gross counts real:", int((a > .1).sum()), " sim:", int((b > .1).sum()))
both = P[(P.drift.abs() > .1) & (P["sim_iso_resave_q75_x0.25"].abs() > .1)]
print("    coinciding flips:", len(both))
# scene-level (each scene counted once): average per scene of the gross indicator
sc = P.assign(g=(a > .1)).groupby("scene").g.mean()
scs = P.assign(g=(P["sim_iso_resave_q75_x0.25"].abs() > .1)).groupby("scene").g.mean()
print("    scene-weighted gross: real %.1f%%, sim %.1f%%" % (100 * sc.mean(), 100 * scs.mean()))
# full-res duplicates: how many distinct full-res crops underlie the 62 pairs
print("    distinct full-res crops in pairs:", P.full_res.nunique(), " distinct quarter:", P.quarter_res.nunique())

# ---------- (D) k1 ----------
D = pd.read_csv(os.path.join(R, "distortion_measured.csv"))
c = D[D.rms_px < 8]
ps = c.groupby("scene").k1.median()
print("\n(D) k1: clean edges n=%d, images %d, scenes %d; pooled median %.4f IQR [%.4f, %.4f]; per-scene median %.4f; negative scenes %d/%d" % (
    len(c), c.id.nunique(), c.scene.nunique(), c.k1.median(), c.k1.quantile(.25), c.k1.quantile(.75), ps.median(), (ps < 0).sum(), len(ps)))
print("    per-scene IQR [%.4f, %.4f]; all edges per-scene median %.4f" % (ps.quantile(.25), ps.quantile(.75), D.groupby("scene").k1.median().median()))
k1 = ps.median(); print("    centre factor (1+k1)/(1+2k1) = %.4f; area (1+4k1/3)/(1+2k1) = %.4f" % ((1 + k1) / (1 + 2 * k1), (1 + 4 * k1 / 3) / (1 + 2 * k1)))

# ---------- (E) sweep claims ----------
S = pd.read_csv(os.path.join(R, "label_audit_sweep.csv"))
print("\n(E) sweep rows at 3.5-5 px/yarn:")
print(S[S.ppt_bin.isin(["3.5", "4.0", "5.0", "6.0", ">7"])][["route", "ppt_bin", "n", "med_abs", "gt10"]].round(1).to_string(index=False))

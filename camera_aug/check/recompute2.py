exec(open("recompute.py").read().split("# ---------- (C)")[0].split("# ---------- (B)")[0])  # load (A) imports only
import numpy as np, pandas as pd, os
SYN = {131,132,134,135,136,137,138,139,141,142,144}
raw = pd.read_csv(os.path.join(R, "audit_real150_raw.csv"), dtype={"source_id": str}); cl = raw[raw.cond=="clean"].set_index("source_id")
s = pd.read_csv(os.path.join(R, "stress_real150_classical.csv"), dtype={"source_id": str})
man = pd.read_csv(os.path.join(H, "stress/real150/manifest.csv"), dtype={"source_id": str})
CAM8 = ["budget_binned_12mp","budget_fullres_50mp","midrange_binned","flagship","legacy_16mp_2019","budget_lowlight","budget_whatsapp","legacy_quarter"]
s = s[(s.estimator=="ours") & s.condition.isin(CAM8)].merge(man[["source_id","condition","S_src","S_out","label_factor_h","label_factor_v"]], on=["source_id","condition"])
L=[]
for ax,f in (("h","label_factor_h"),("v","label_factor_v")):
    t = pd.DataFrame(dict(sid=s.source_id, cond=s.condition, obs=s[ax], fac=s[f], S_out=s.S_out, S_src=s.S_src))
    t["clean"]=t.sid.map(cl[ax]); t["conf"]=t.sid.map(cl["conf"]); t["drift"]=t.obs/(t.fac*t.clean)-1
    t["app"]=(t.fac*t.clean>=19.8)&(t.fac*t.clean<=0.4*np.minimum(t.S_out,1024))&(t.clean<=0.4*np.minimum(t.S_src,1024))&np.isfinite(t.clean)
    L.append(t)
x = pd.concat(L); x = x[~x.sid.astype(int).isin(SYN)]
g_all = (x.drift.abs()>.1) | ~np.isfinite(x.drift) | ~x.app
print("gross incl. non-applicable as gross: %.1f%%" % (100*g_all.mean()), " NaN drifts:", int((~np.isfinite(x.drift)).sum()))
print("gross all rows, NaN dropped: %.1f%%" % (100*(x.drift.abs()>.1)[np.isfinite(x.drift)].mean()))
q = pd.qcut(x.conf, 4, labels=["Q1","Q2","Q3","Q4"])
print("quartiles, all rows, non-applicable as gross:", g_all.groupby(q, observed=True).mean().mul(100).round(1).to_dict())
print("non-applicable rows by quartile:", (~x.app).groupby(q, observed=True).sum().to_dict())
xa = x[x.app]; fl = xa[xa.drift.abs()>.1]; lr = np.log(1+fl.drift)
near = lambda r: (np.abs(lr-np.log(r))<0.1)
print("n flips", len(fl), "images with any flip:", fl.sid.nunique(), "of", x.sid.nunique())
print("log-tol 0.1: x0.5/x2 %.0f%%, x0.67/x1.5 %.0f%%" % (100*(near(.5)|near(2)).mean(), 100*(near(2/3)|near(1.5)).mean()))
near15 = lambda r: (np.abs(lr-np.log(r))<0.15)
print("log-tol 0.15: x0.5/x2 %.0f%%, x0.67/x1.5 %.0f%%" % (100*(near15(.5)|near15(2)).mean(), 100*(near15(2/3)|near15(1.5)).mean()))
# images that flip on >= half of their 16 counts
per = xa.assign(g=xa.drift.abs()>.1).groupby("sid").g.agg(["sum","count"])
print("images flipping on >= half of counts:", int((per["sum"]>=per["count"]/2).sum()))
# number of distinct scenes contributing flips
sc = {str(i): gi for gi, g in enumerate(json.load(open(os.path.join(H,"originals_provenance.json")))["groups"]) for i in g}
print("scenes with any flip:", fl.sid.map(sc).nunique(), "of", x.sid.map(sc).nunique())

import os, numpy as np, pandas as pd
H = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "camera_aug")
SYN = {131,132,134,135,136,137,138,139,141,142,144}
CAM8 = ["budget_binned_12mp","budget_fullres_50mp","midrange_binned","flagship","legacy_16mp_2019","budget_lowlight","budget_whatsapp","legacy_quarter"]
man = pd.read_csv(os.path.join(H, "stress/real150/manifest.csv"), dtype={"source_id": str})
man = man[man.condition.isin(CAM8) & ~man.source_id.astype(int).isin(SYN)]
raw = pd.read_csv(os.path.join(H, "results/audit_real150_raw.csv"), dtype={"source_id": str})
cl = raw[raw.cond=="clean"].set_index("source_id")
for ax in ("h","v"):
    man["tpi_"+ax] = man.source_id.map(cl[ax])
man["tpi_fine"] = man[["tpi_h","tpi_v"]].max(axis=1)
man["ppy_eff"] = np.minimum(man.S_src, man.S_out) / man.tpi_fine
man["lo"] = man.S_src < 1500
print("renders:", len(man), " lo-res sources:", man[~man.duplicated("source_id")].lo.sum(), "of", man.source_id.nunique())
print("share of camera renders with effective px/yarn (finer axis) < 4: %.1f%%; < 7: %.1f%%" % (100*(man.ppy_eff<4).mean(), 100*(man.ppy_eff<7).mean()))
print("share with scene upsampled:", 100*man.scene_upsampled.astype(str).eq("True").mean())
print(man.groupby("condition").apply(lambda g: pd.Series(dict(upsampled=100*g.scene_upsampled.astype(str).eq("True").mean(), ppy_eff_lt4=100*(g.ppy_eff<4).mean()))).round(1).to_string())

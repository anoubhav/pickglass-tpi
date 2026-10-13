import json, os, numpy as np, pandas as pd
H = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "camera_aug")
SYN = {131,132,134,135,136,137,138,139,141,142,144}
CAM8 = ["budget_binned_12mp","budget_fullres_50mp","midrange_binned","flagship","legacy_16mp_2019","budget_lowlight","budget_whatsapp","legacy_quarter"]
def analyse(set_name, tpi):  # tpi: DataFrame index source_id cols h,v
    man = pd.read_csv(os.path.join(H, f"stress/{set_name}/manifest.csv"), dtype={"source_id": str})
    man = man[man.condition.isin(CAM8 + ["budget+glare0.05","budget+glare0.3","budget+gblur3","budget+gblur6"])]
    P = man.params_json.apply(json.loads)
    man["L"] = P.apply(lambda p: p.get("motion_u", 0.0)); man["ang"] = P.apply(lambda p: p.get("motion_angle_deg", 0.0))
    rows = []
    for ax, c in (("h", np.cos), ("v", np.sin)):
        t = man[["source_id","condition","rep","S_out","L","ang"]].copy(); t["ax"] = ax
        t["tpi"] = t.source_id.map(tpi[ax])
        t["Lproj_over_pitch"] = t.L * np.abs(c(np.radians(t.ang))) * t.tpi / 1000.0
        t["ppy"] = t.S_out / t.tpi
        rows.append(t)
    x = pd.concat(rows).dropna(subset=["tpi"])
    x["motion_flag"] = x.Lproj_over_pitch >= 0.5
    x["res_flag"] = x.ppy < 4.0
    g = x.groupby("condition").agg(n=("ax","size"), motion_ge_half_pitch=("motion_flag","mean"), ppy_lt4=("res_flag","mean"), Lmax=("L","max"))
    g[["motion_ge_half_pitch","ppy_lt4"]] *= 100
    print(set_name); print(g.round(1).to_string())
    cam = x[x.condition.isin(CAM8)]
    print("  camera conds: counts with motion>=0.5 pitch: %.1f%%, ppy<4: %.1f%%, either: %.1f%%" % (100*cam.motion_flag.mean(), 100*cam.res_flag.mean(), 100*(cam.motion_flag|cam.res_flag).mean()))
lab = pd.read_csv(os.path.join(H, "stress/legacy13/labels.csv"), dtype={"source_id": str}).drop_duplicates("source_id").set_index("source_id")
analyse("legacy13", pd.DataFrame(dict(h=lab.h_src, v=lab.v_src)))
raw = pd.read_csv(os.path.join(H, "results/audit_real150_raw.csv"), dtype={"source_id": str})
cl = raw[(raw.cond=="clean") & ~raw.source_id.astype(int).isin(SYN)].set_index("source_id")[["h","v"]]
analyse("real150", cl)

import pandas as pd, numpy as np, sys
KEYS = ["Dx0","Dxp","Dxm","Dy0","Dyp","Dym","Cp","Cm","Cpx","Cmx","Cpy","Cmy","Hx0","Hxp","Hxm","Hy0","Hyp","Hym"]
HYPS = {"t": "true", "hx": "half_x", "hy": "half_y", "dx": "dbl_x", "dy": "dbl_y", "hh": "half_xy", "dd": "dbl_xy"}
def tab(d, wt="snr"):
    rows = []
    for r in d.to_dict("records"):
        if f"t_{wt}_Dx0" not in r or pd.isna(r.get("true_h")): continue
        for tag, lab in HYPS.items():
            row = dict(src=r["src"], key=r["key"], hyp=lab, W=r.get("W"))
            for k in KEYS: row[k] = r.get(f"{tag}_{wt}_{k}", np.nan)
            row["Dx"] = np.nanmax([row["Dx0"], row["Dxp"], row["Dxm"]])
            row["Dy"] = np.nanmax([row["Dy0"], row["Dyp"], row["Dym"]])
            row["C"] = np.nanmax([row["Cp"], row["Cm"]])
            row["Hx"] = np.nanmax([row["Hx0"], row["Hxp"], row["Hxm"]])
            row["Hy"] = np.nanmax([row["Hy0"], row["Hyp"], row["Hym"]])
            rows.append(row)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    d = pd.read_pickle(sys.argv[1]); X = tab(d, sys.argv[2] if len(sys.argv) > 2 else "snr")
    print(X[["key","hyp","W","Dx","Dy","C","Hx","Hy","Dx0","Dxp","Dxm","Cp","Cm"]].round(2).to_string())

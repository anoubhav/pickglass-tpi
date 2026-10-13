"""Label-free consistency metrics on the 150 real crops (no human counts read).

Pairs:
  reshoot : 41 accepted same-patch pairs of scene reps (paper/tables/fabric_groups_v2.json);
            |angle|<15 -> same axes, |angle+-90|<15 -> swapped axes, else 'oblique' (axis-set compare)
  res     : quarter_res_copy member vs its scene rep (manifest; h/v swapped if scene_swap_hv)
  rot     : any member with scene_rot_k != 0 vs its scene rep
  copy    : png_copy / byte_identical / notebook_synthetic vs rep
Per axis ratio r = max/min; 'octave' if 1.7 <= r <= 2.35 (or 2.6..3.4 for 3x), 'agree5' if r <= 1.05.
"""
import os  # added by the release export (path configuration)
import json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
MAN = json.load(open(f"{ROOT}/annotation_tool/data/manifest.json"))
GRP = json.load(open(f"{ROOT}/paper/tables/fabric_groups_v2.json"))
IM = {i["id"]: i for i in MAN["images"]}
REP_OF_SCENE = {i["scene_id"]: i["id"] for i in MAN["images"] if i["is_scene_rep"]}


def pairs():
    out = []
    for p in GRP["accepted_pairs"]:
        a, b, ang = REP_OF_SCENE[p["scene_a"]], REP_OF_SCENE[p["scene_b"]], p["angle_ab_deg"]
        aa = abs(((ang + 90) % 180) - 90)  # angle mod 180 in [0, 90]
        if aa < 15:
            mode = "same"
        elif aa > 75:
            mode = "swap"
        else:
            mode = "oblique"
        out.append(("reshoot", a, b, mode))
    for i in MAN["images"]:
        if i["is_scene_rep"]:
            continue
        rep = REP_OF_SCENE[i["scene_id"]]
        mode = "swap" if i["scene_swap_hv"] else "same"
        if i["kind"] == "quarter_res_copy":
            out.append(("res", rep, i["id"], mode))
        if i["scene_rot_k"] != 0:
            out.append(("rot", rep, i["id"], mode))
        if i["kind"] in ("png_copy", "byte_identical", "notebook_synthetic"):
            out.append(("copy", rep, i["id"], mode))
    return out


PAIRS = pairs()


def _ratios(H, V, a, b, mode):
    ha, va, hb, vb = H[a], V[a], H[b], V[b]
    if mode == "swap":
        hb, vb = vb, hb
    if mode == "oblique":  # compare as sorted sets
        x, y = sorted([ha, va]), sorted([hb, vb])
        return [max(x[0], y[0]) / min(x[0], y[0]), max(x[1], y[1]) / min(x[1], y[1])]
    return [max(ha, hb) / min(ha, hb), max(va, vb) / min(va, vb)]


def consistency(df, hcol="h_count", vcol="v_count", verbose=False):
    H = dict(zip(df["id"], df[hcol]))
    V = dict(zip(df["id"], df[vcol]))
    rows = []
    for typ, a, b, mode in PAIRS:
        rs = _ratios(H, V, a, b, mode)
        for ax, r in zip("hv", rs):
            rows.append(dict(type=typ, a=a, b=b, mode=mode, axis=ax, r=r))
    R = pd.DataFrame(rows)
    R["oct"] = R.r.between(1.7, 2.35)
    R["x3"] = R.r.between(2.6, 3.4)
    R["agree5"] = R.r <= 1.05
    R["agree10"] = R.r <= 1.10
    summ = R.groupby("type").agg(n_axes=("r", "size"), oct=("oct", "sum"), x3=("x3", "sum"),
                                 agree5=("agree5", "mean"), agree10=("agree10", "mean"))
    # pair-level: any axis octave
    P = R.groupby(["type", "a", "b"]).agg(oct=("oct", "any"), both5=("agree5", "all")).reset_index()
    ps = P.groupby("type").agg(n_pairs=("oct", "size"), pairs_oct=("oct", "sum"), pairs_both5=("both5", "mean"))
    summ = summ.join(ps)
    if verbose:
        print(summ.round(3).to_string())
    return summ, R


if __name__ == "__main__":
    d = pd.read_csv(f"{ROOT}/methods/predictions_ours_150.csv")
    for nm, (hc, vc) in {"ours": ("h_count", "v_count"), "lattice": ("h_lattice", "v_lattice"),
                         "fft": ("h_fft", "v_fft"), "profile": ("h_profile", "v_profile"),
                         "tiles": ("h_tiles", "v_tiles"), "gabor": ("h_gabor", "v_gabor"),
                         "alt": ("alt_h", "alt_v")}.items():
        dd = d.copy()
        dd[hc] = dd[hc].fillna(1.0)
        dd[vc] = dd[vc].fillna(1.0)
        print("==", nm)
        consistency(dd, hc, vc, verbose=True)

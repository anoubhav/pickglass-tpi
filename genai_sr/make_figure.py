"""make_figure.py -- qualitative figure of the genai_sr study.

    <venv2>/bin/python make_figure.py
-> paper/figs/genai_hallucination.pdf/.png        rows: reference / degraded x8 / bicubic / SR / diffusion / AI-enhance
   paper/figs/genai_hallucination_all.pdf/.png    all 9 rows (adds Real-ESRGAN, InvSR, FLUX.2 klein)

Examples (fixed rule, R-LEG at D8, keys with every restoration available and ours correct (+-10 %) on the clean
reference): A = the largest OVER-count of ours among the generative restorations (seedvr2, invsr, flux2klein, nbpro);
B = the largest UNDER-count (another crop).  Per example three panels:
full crop (zoom box = central 1/6 inch), zoom, and the counter's whitened spectrum (z, clipped 0..12) with the
reference yarn lattice (open circles) and, for degraded rows, the LR Nyquist square (dashed).
Label on each crop: ours' count (vertical x horizontal yarns per inch), green if both counts are within 10 % of the
hand count, red otherwise; the reference row also shows the hand count.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gsr_common as G  # noqa: E402

sys.path.insert(0, os.path.join(G.WVAQ, "methods"))
import counting as C  # noqa: E402

FIGS = os.path.join(G.WVAQ, "paper", "figs")
RES = os.path.join(G.GSR, "results")
COND = "D8"
ROWS_MAIN = ["clean", "none", "bicubic", "swinir", "seedvr2", "nbpro"]
ROWS_ALL = ["clean", "none", "bicubic", "swinir", "realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
LABEL = {"clean": "Reference\n(~1800 px/in)", "none": "Degraded \u00d78\n(~225 px/in)", "bicubic": "Bicubic",
         "swinir": "SwinIR-M\n(regression SR)", "realesrgan": "Real-ESRGAN\n(GAN SR)",
         "seedvr2": "SeedVR2-3B\n(1-step diffusion)", "invsr": "InvSR\n(diffusion inv.)",
         "flux2klein": "FLUX.2 klein\n(open AI edit)", "nbpro": "Nano Banana Pro\n(commercial AI edit)"}
GENR = ["seedvr2", "invsr", "flux2klein", "nbpro"]
OK, BAD, INK = "#1b8a3a", "#d0312d", "#222222"
plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})


def fetch(rest, key):
    """local path of the S x S image (fetched from gpu-server if only there)."""
    if rest == "clean":
        return G.ref_path(key)
    if rest == "none":
        return G.lr_path(COND, key)
    p = G.out_path(rest, COND, key)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        rel = os.path.relpath(p, G.GSR)
        subprocess.run(["scp", "-q", f"{os.environ.get('PICKGLASS_GPU_HOST', 'gpu-server')}:{os.environ.get('PICKGLASS_REMOTE_ROOT', 'pickglass')}/genai_sr/{rel}", p], check=True)
    return p


def counts():
    d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "counts_classical_*.csv"))
                   if "_s1" not in p and "_s2" not in p], ignore_index=True)
    gt = pd.read_csv(os.path.join(G.INPUTS, "gt_legacy13.csv")).set_index("key")
    return d, gt


def pick(d, gt):
    x = d[(d.set == "R-LEG") & ((d.cond == COND) | (d.restorer == "clean"))].copy()
    have = x.groupby("key").restorer.apply(set)
    keys = [k for k, s in have.items() if set(ROWS_ALL) <= s]
    # only crops that ours counts correctly (both counts within 10 % of the hand count) on the CLEAN reference,
    # so that every change in a row is attributable to the degradation / restoration
    cl = x[x.restorer == "clean"].set_index("key")
    good = [k for k in keys if abs(cl.loc[k, "ours_h"] / gt.loc[k, "warp"] - 1) <= 0.10
            and abs(cl.loc[k, "ours_v"] / gt.loc[k, "weft"] - 1) <= 0.10]
    g = x[x.restorer.isin(GENR) & x.key.isin(good)].copy()
    g["eh"] = g.ours_h / g.key.map(gt.warp) - 1
    g["ev"] = g.ours_v / g.key.map(gt.weft) - 1
    g["emax"] = g[["eh", "ev"]].max(axis=1)
    g["emin"] = g[["eh", "ev"]].min(axis=1)
    a = g.sort_values("emax", ascending=False).iloc[0]
    rest = g[g.key != a.key].sort_values("emin").iloc[0]
    return [a.key, rest.key], {a.key: f"largest over-count ({a.restorer})", rest.key: f"largest under-count ({rest.restorer})"}


def spectrum_view(img, B_ref, L, lr_side=None):
    g, w, N, info = C.preprocess(img)
    sp = C.Spectrum(g)
    c = sp.c
    Lp = int(min(L, c - 1))
    Z = np.full((2 * L + 1, 2 * L + 1), np.nan)
    Z[L - Lp:L + Lp + 1, L - Lp:L + Lp + 1] = sp.Z[c - Lp:c + Lp + 1, c - Lp:c + Lp + 1]
    return Z


def draw(rows, out, keys, why, d, gt):
    ncol = 3 * len(keys)
    cw = 0.98
    fig_w = 0.95 + ncol * cw
    fig_h = 0.42 + len(rows) * cw
    fig, axes = plt.subplots(len(rows), ncol, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.95 / fig_w, right=0.995, top=1 - 0.40 / fig_h, bottom=0.005, wspace=0.04, hspace=0.06)
    man = pd.read_csv(G.MANIFEST).set_index(["key", "cond"])
    for j, key in enumerate(keys):
        ref = cv2.imread(G.ref_path(key), cv2.IMREAD_COLOR)
        S = ref.shape[0]
        lat = C.estimate(ref, method="ours", keep_arrays=True)["debug"]["lattice"]
        B = np.array(lat["B"])
        gh, gv = gt.loc[key, "warp"], gt.loc[key, "weft"]
        L = int(2.4 * max(gh, gv))
        z = S // 6
        y0 = x0 = S // 2 - z // 2
        lr_side = int(man.loc[(key, COND), "lr_w"])
        top = axes[0, 3 * j]
        top.text(0.0, 1.22, f"legacy {key[4:]}: hand count {gh:g}\u00d7{gv:g}", transform=top.transAxes,
                 ha="left", va="bottom", fontsize=6.5, weight="bold")
        for i, r in enumerate(rows):
            img = cv2.imread(fetch(r, key), cv2.IMREAD_COLOR)
            disp = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST) if r == "none" else img
            a_full, a_zoom, a_sp = axes[i, 3 * j], axes[i, 3 * j + 1], axes[i, 3 * j + 2]
            a_full.imshow(cv2.cvtColor(cv2.resize(disp, (600, 600), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            s = 600 / S
            a_full.add_patch(plt.Rectangle((x0 * s, y0 * s), z * s, z * s, fill=False, ec="white", lw=0.6))
            a_zoom.imshow(cv2.cvtColor(disp[y0:y0 + z, x0:x0 + z], cv2.COLOR_BGR2RGB), interpolation="nearest"
                          if r == "none" else "antialiased")
            rr = d[(d.key == key) & (d.restorer == r) & ((d.cond == COND) | (r == "clean"))].iloc[0]
            h, v = rr.ours_h, rr.ours_v
            good = abs(h / gh - 1) <= 0.10 and abs(v / gv - 1) <= 0.10
            txt = f"ours {h:.0f}\u00d7{v:.0f}"
            a_full.text(0.03, 0.04, txt, transform=a_full.transAxes, fontsize=6, color="white", weight="bold",
                        bbox=dict(boxstyle="square,pad=0.15", fc=OK if good else BAD, ec="none", alpha=0.9))
            a_full.text(0.97, 0.96, f"c={rr.ours_conf:.2f}", transform=a_full.transAxes, fontsize=5, color="white",
                        ha="right", va="top", bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.55))
            Z = spectrum_view(img, B, L)
            a_sp.imshow(np.clip(Z, 0, 12), cmap="magma", vmin=0, vmax=12, extent=(-L, L, L, -L), interpolation="nearest")
            pts = [B @ np.array(p, float) for p in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]]
            P = np.array(pts + [-p for p in pts])
            a_sp.plot(P[:, 0], P[:, 1], "o", ms=3.2, mfc="none", mec="#55d6ff", mew=0.6)
            if r != "clean":
                f = lr_side / 2
                a_sp.plot([-f, f, f, -f, -f], [-f, -f, f, f, -f], "--", color="white", lw=0.5)
            a_sp.set_xlim(-L, L)
            a_sp.set_ylim(L, -L)
            for a in (a_full, a_zoom, a_sp):
                a.set_xticks([])
                a.set_yticks([])
                for sp_ in a.spines.values():
                    sp_.set_linewidth(0.3)
            if j == 0:
                a_full.set_ylabel(LABEL[r], fontsize=6.3, rotation=0, ha="right", va="center", labelpad=3)
            if i == 0:
                a_full.text(0.5, 1.02, "1-inch crop", transform=a_full.transAxes, ha="center", va="bottom",
                            fontsize=5.5)
                a_zoom.text(0.5, 1.02, "zoom (1/6 in)", transform=a_zoom.transAxes, ha="center", va="bottom",
                            fontsize=5.5)
                a_sp.text(0.5, 1.02, "spectrum + ref. lattice", transform=a_sp.transAxes, ha="center", va="bottom",
                          fontsize=5.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{out}.{ext}"))
    plt.close(fig)
    print("wrote", out, keys)


def main():
    d, gt = counts()
    keys, why = pick(d, gt)
    draw(ROWS_MAIN, "genai_hallucination", keys, why, d, gt)
    draw(ROWS_ALL, "genai_hallucination_all", keys, why, d, gt)
    G.write_json(os.path.join(G.GSR, "logs", "figure_examples.json"), dict(keys=keys, why=why, cond=COND))


if __name__ == "__main__":
    main()

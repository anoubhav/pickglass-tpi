"""Compact one-row version of methods/figs/fig_lattice.pdf for the 8-page paper.

Same content and code path as methods/make_figs.fig_lattice (crop with guide-line
crossings + whitened spectrum with the selected lattice), but two examples side by side:
legacy a1 (2/1 twill, hand count 76 x 64) and PickGlass image #86.
The legacy panel reads the canonical crop vlm/crops/legacy/a1.png (the one crop rule of
paper Tab. 2 / tables/legacy13_unified.csv), not the original 2019 file (Oct 7 change).

    PYTHONDONTWRITEBYTECODE=1 <venv2>/bin/python tools/make_lattice_compact.py
Writes paper/figs/fig_lattice_compact.pdf/.png. Only reads methods/ (imports its modules).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, os.path.join(ROOT, "methods"))
import make_figs as MF  # noqa: E402  (sets rcParams, imports counting/rectify)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

plt, C, CAT = MF.plt, MF.C, MF.CAT


def _load(src):
    """('legacyvlm', 'a1') -> canonical crop (vlm/crops/legacy); anything else -> make_figs._load."""
    if src[0] == "legacyvlm":
        return cv2.imread(os.path.join(ROOT, "vlm", "crops", "legacy", f"{src[1]}.png")), f"legacy {src[1]}"
    return MF._load(src)


def panel(ax_img, ax_sp, src, gt=None, legend=False):
    img, title = _load(src)
    e = C.estimate(img, method="ours", keep_arrays=True)
    d = e["debug"]
    sp, lat = d["spectrum"], d["lattice"]
    B = np.array(lat["B"]) if lat else None
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if lat:
        gp = C.guide_positions(img, e["h_count"], e["v_count"], -lat["angle_x"], -lat["angle_y"])
        ax_img.axhline(gp["h_line_y"], color=CAT[0], lw=0.6)
        ax_img.axvline(gp["v_line_x"], color=CAT[1], lw=0.6)
        ax_img.plot(gp["h_positions"], [gp["h_line_y"]] * len(gp["h_positions"]), "o", ms=1.6, mfc=CAT[0],
                    mec="white", mew=0.3)
        ax_img.plot([gp["v_line_x"]] * len(gp["v_positions"]), gp["v_positions"], "o", ms=1.6, mfc=CAT[1],
                    mec="white", mew=0.3)
    ax_img.set_xlim(-0.5, img.shape[1] - 0.5)
    ax_img.set_ylim(img.shape[0] - 0.5, -0.5)
    t = f"{title}: h={e['h_count']:.1f}, v={e['v_count']:.1f}, {e['weave']}, conf {e['confidence']:.2f}"
    if gt:
        t += f"\nhand count: warp {gt[0]:g}, weft {gt[1]:g}"
    ax_img.set_title(t, loc="left", fontsize=6.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.grid(False)
    c = sp.c
    L = int(min(c, 2.6 * max(e["h_count"], e["v_count"]) if np.isfinite(e["h_count"]) else c))
    Z = sp.Z[c - L:c + L + 1, c - L:c + L + 1]
    ax_sp.imshow(np.clip(Z, -1, 16), cmap="Greys", extent=(-L - .5, L + .5, L + .5, -L - .5),
                 interpolation="nearest")
    if B is not None:
        pts = [(m, k) for m in range(-3, 4) for k in range(-3, 4) if (m, k) != (0, 0)]
        q = np.array([B @ np.array(p) for p in pts])
        ok = (np.abs(q) <= L).all(1)
        ax_sp.plot(q[ok, 0], q[ok, 1], "o", ms=3.5, mfc="none", mec=CAT[2], mew=0.7,
                   label="yarn lattice $n\\mathbf{a}+m\\mathbf{b}$")
        W = lat["weave_W"]
        if W != "none":
            wp = []
            for r in C._REPS[W][1:]:
                for m in range(-2, 3):
                    for k in range(-2, 3):
                        wp.append(B @ (np.array(r) + np.array([m, k])))
            wp = np.array(wp)
            ok = (np.abs(wp) <= L).all(1)
            ax_sp.plot(wp[ok, 0], wp[ok, 1], "^", ms=3, mfc="none", mec=CAT[3], mew=0.7,
                       label="weave sub-lattice")
        for vec, col, lab in ((B[:, 0], CAT[0], "$\\mathbf{a}$ (vertical yarns)"),
                              (B[:, 1], CAT[1], "$\\mathbf{b}$ (horizontal yarns)")):
            ax_sp.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
            ax_sp.plot([vec[0]], [vec[1]], "o", ms=5, mfc="none", mec=col, mew=1.1, label=lab)
    ax_sp.set_xlim(-L, L)
    ax_sp.set_ylim(L, -L)
    ax_sp.set_xlabel("u [cycles/inch]", fontsize=6.5, labelpad=1)
    ax_sp.set_ylabel("v [cycles/inch]", fontsize=6.5, labelpad=1)
    ax_sp.tick_params(labelsize=6)
    ax_sp.grid(False)
    return ax_sp.get_legend_handles_labels() if legend else None


def main():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1.05, 1, 1.05]})
    hl = panel(axs[0], axs[1], ("legacyvlm", "a1"), gt=(76, 64), legend=True)
    panel(axs[2], axs[3], ("150", 86))
    fig.legend(*hl, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handletextpad=0.3, columnspacing=1.2)
    out = os.path.join(PAPER, "figs", "fig_lattice_compact")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=220)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()

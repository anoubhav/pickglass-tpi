"""make_figs.py -- paper figures for the counting method + capture-quality ablations (PDF + PNG).

    python make_figs.py lattice      # fig_lattice_*.pdf: crop + whitened spectrum + lattice overlay
    python make_figs.py profile      # fig_profile.pdf: deskewed projection profile + YIN + comb
    python make_figs.py drift        # fig_degradation_drift.pdf  (needs results/all150_degrade.csv,
                                     #  results/legacy13_degrade.csv)
    python make_figs.py confdrift    # fig_conf_vs_drift.pdf
    python make_figs.py legacybar    # fig_legacy_methods.pdf (needs results/legacy13_metrics.csv)
    python make_figs.py all
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import counting as C  # noqa: E402
import degrade as D  # noqa: E402

FIGS = os.path.join(HERE, "figs")
RES = os.path.join(HERE, "results")
# reference categorical order (validated palette): blue, orange, aqua, yellow, magenta, green, violet, red
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"ours": "ours: lattice + fusion", "lattice": "lattice only", "gabor": "log-Gabor bank",
         "fft": "FFT max near-axis peak", "fft_harm": "FFT harmonic sum", "tiles": "FFT tile voting",
         "profile": "deskewed profile + YIN", "peaks": "naive peak count", "fusion3": "v0 fusion (FFT)",
         "2019 doc3+count (faithful)": "2019 dot map + transition count",
         "2019 test.py RGB irregularity (weft only)": "2019 RGB irregularity (weft only)"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIGS, name + ".pdf/.png"))


def _load(src):
    """src: ('legacy', 'a1') | ('150', 86) -> (bgr, title)"""
    kind, key = src
    if kind == "legacy":
        sys.path.insert(0, os.path.join(HERE, "legacy_2019"))
        import ablate as A
        return cv2.imread(A.legacy_files()[key]), f"legacy {key}"
    import rectify as R
    return R.rectify(int(key)), f"#{key}"


# ------------------------------------------------------------------------------ lattice
def fig_lattice(srcs=(("legacy", "a1"), ("legacy", "a5"), ("150", 55)), gts=None, name="fig_lattice"):
    n = len(srcs)
    fig, axs = plt.subplots(n, 2, figsize=(6.8, 3.4 * n), gridspec_kw={"width_ratios": [1, 1.08], "wspace": 0.35})
    axs = np.atleast_2d(axs)
    for i, src in enumerate(srcs):
        img, title = _load(src)
        e = C.estimate(img, method="ours", keep_arrays=True)
        d = e["debug"]
        sp, lat = d["spectrum"], d["lattice"]
        B = np.array(lat["B"]) if lat else None
        S = img.shape[0]
        ax = axs[i, 0]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if lat:
            gp = C.guide_positions(img, e["h_count"], e["v_count"], -lat["angle_x"], -lat["angle_y"])
            ax.axhline(gp["h_line_y"], color=CAT[0], lw=0.8)
            ax.axvline(gp["v_line_x"], color=CAT[1], lw=0.8)
            ax.plot(gp["h_positions"], [gp["h_line_y"]] * len(gp["h_positions"]), "o", ms=2.2, mfc=CAT[0],
                    mec="white", mew=0.4)
            ax.plot([gp["v_line_x"]] * len(gp["v_positions"]), gp["v_positions"], "o", ms=2.2, mfc=CAT[1],
                    mec="white", mew=0.4)
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        gt = (gts or {}).get(src)
        t = f"{title}  final h={e['h_count']:.1f}, v={e['v_count']:.1f}\n{e['weave']}, conf={e['confidence']:.2f}"
        if lat:
            t += f"; lattice basis {abs(B[0, 0]):.1f}/{abs(B[1, 1]):.1f}"
        if gt:
            t += f"\nhand count: warp={gt[0]:g}, weft={gt[1]:g}"
        ax.set_title(t, loc="left", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        # spectrum
        ax = axs[i, 1]
        c = sp.c
        L = int(min(c, 2.6 * max(e["h_count"], e["v_count"]) if np.isfinite(e["h_count"]) else c))
        Z = sp.Z[c - L:c + L + 1, c - L:c + L + 1]
        sc = S / sp.N  # cycles per crop are resolution independent: axes in threads per inch
        ax.imshow(np.clip(Z, -1, 16), cmap="Greys", extent=(-L - .5, L + .5, L + .5, -L - .5), interpolation="nearest")
        if B is not None:
            pts = [(m, k) for m in range(-3, 4) for k in range(-3, 4) if (m, k) != (0, 0)]
            q = np.array([B @ np.array(p) for p in pts])
            ok = (np.abs(q) <= L).all(1)
            ax.plot(q[ok, 0], q[ok, 1], "o", ms=5, mfc="none", mec=CAT[2], mew=0.8, label="yarn lattice $n\\mathbf{a}+m\\mathbf{b}$")
            W = lat["weave_W"]
            if W != "none":
                wp = []
                for r in C._REPS[W][1:]:
                    for m in range(-2, 3):
                        for k in range(-2, 3):
                            wp.append(B @ (np.array(r) + np.array([m, k])))
                wp = np.array(wp)
                ok = (np.abs(wp) <= L).all(1)
                ax.plot(wp[ok, 0], wp[ok, 1], "^", ms=4, mfc="none", mec=CAT[3], mew=0.8,
                        label=f"weave sub-lattice ({W})")
            for vec, col, lab in ((B[:, 0], CAT[0], "$\\mathbf{a}$ (vertical yarns)"),
                                  (B[:, 1], CAT[1], "$\\mathbf{b}$ (horizontal yarns)")):
                ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
                ax.plot([vec[0]], [vec[1]], "o", ms=7, mfc="none", mec=col, mew=1.4, label=lab)
        ax.set_xlim(-L, L)
        ax.set_ylim(L, -L)
        ax.set_xlabel("u [cycles / inch]")
        ax.set_ylabel("v [cycles / inch]")
        ax.grid(False)
        ax.set_title("whitened spectrum + selected lattice", loc="left", fontsize=7)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderpad=0.3,
                      handletextpad=0.3)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------------------ profile
def fig_profile(src=("legacy", "a5"), name="fig_profile"):
    img, title = _load(src)
    e = C.estimate(img, method="ours")
    lat = e["debug"]["lattice"]
    g, w, N, info = C.preprocess(img)
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for r, (axis, cnt, ang, col, lab) in enumerate((("x", e["h_count"], -lat["angle_x"], CAT[0], "vertical yarns (h)"),
                                                    ("y", e["v_count"], -lat["angle_y"], CAT[1], "horizontal yarns (v)"))):
        pr = C.est_profile(g, w, axis, angle_deg=ang)
        p = pr["profile"]
        T = N / cnt
        sm = C.ndimage.gaussian_filter1d(p, max(0.6, T / 6))
        x = np.arange(len(p)) / N  # inch
        ax = axs[r, 0]
        i0 = int(0.04 * N)
        seg = slice(i0, int(min(len(p), i0 + 12 * T)))  # ~12 periods, away from the crop edge
        ax.plot(x[seg], p[seg], color="#b9b8b2", lw=0.8, label="deskewed projection")
        ax.plot(x[seg], sm[seg], color=col, lw=1.4, label="smoothed")
        gp = C.guide_positions(img, e["h_count"], e["v_count"], -lat["angle_x"], -lat["angle_y"])
        pos = np.array(gp["h_positions"] if axis == "x" else gp["v_positions"]) / img.shape[1 if axis == "x" else 0]
        pos = pos[(pos >= x[seg][0]) & (pos <= x[seg][-1])]
        yv = np.interp(pos * N, np.arange(len(sm)), sm)
        ax.plot(pos, yv, "o", ms=4, mfc=col, mec="white", mew=0.6, label="yarn crossings (proposal)")
        ax.set_xlabel("position along the guide line [inch]")
        ax.set_ylabel("contrast-normalised grey")
        ax.set_title(f"{title}: {lab}, {cnt:.1f} per inch (period {T * img.shape[0] / N:.1f} px)", loc="left")
        if r == 0:
            hdl = ax.get_legend_handles_labels()
        # YIN cumulative-mean-normalised difference
        ax = axs[r, 1]
        lo, hi = C._band(N)
        tau_max = int(min(N / lo, N // 2))
        dn = np.mean([C._yin_dprime(C._project(g, w, np.tan(np.radians(ang)), axis, rows=(a, b)), tau_max)
                      for a, b in zip(np.linspace(0, N, 5).astype(int)[:-1], np.linspace(0, N, 5).astype(int)[1:])],
                     axis=0)
        taus = np.arange(len(dn))
        ax.plot(N / np.maximum(taus[2:], 1), dn[2:], color=col, lw=1.2)
        ax.axvline(N / pr["period_px"], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(lo, min(hi, 3.2 * cnt))
        ax.set_xlabel("candidate density [per inch]")
        ax.set_ylabel("YIN $d'(\\tau)$")
        ax.set_title(f"YIN: {N / pr['period_px']:.1f} per inch", loc="left")
    fig.legend(*hdl, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, name)


# ------------------------------------------------------------------------------ drift
def _drift_table(df):
    """degraded predictions -> drift vs the clean prediction (shared with summarise_drift.py;
    a failed prediction has |drift| = inf)."""
    import summarise_drift
    return summarise_drift.drift_table(df)


def fig_drift(name="fig_degradation_drift", est=("ours", "lattice", "fft"), p150=None, pleg=None):
    p150 = p150 or os.path.join(RES, "all150_degrade.csv")
    pleg = pleg or os.path.join(RES, "legacy13_degrade.csv")
    d150 = _drift_table(pd.read_csv(p150)) if os.path.exists(p150) else None
    dleg = pd.read_csv(pleg) if os.path.exists(pleg) else None
    kinds = [k for k in D.LEVELS]
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.9), sharey=True)
    for ax, k in zip(axs.ravel(), kinds):
        for j, m in enumerate(est):
            if d150 is not None:
                g = d150[(d150.kind == k) & (d150.estimator == m)]
                if len(g):
                    s = g.groupby("level").absdrift.agg([lambda x: np.mean(x <= 0.05) * 100, "count"])
                    s.columns = ["stable", "n"]
                    ax.plot(s.index, s.stable, "-o", color=CAT[j], ms=3, lw=1.6,
                            label=f"{NAMES.get(m, m)}: 150 crops, % within 5 % of own clean prediction")
            if dleg is not None:
                g = dleg[(dleg.estimator == m) & (dleg.gt_usable == True)]  # noqa: E712
                g = g[(g.kind == k) | (g.kind == "clean")]
                if len(g):
                    t = np.r_[g.gt_warp.values, g.gt_weft.values]
                    pr = np.r_[g.h.values, g.v.values]
                    lv = np.r_[g.level.values, g.level.values]
                    kk = np.r_[g.kind.values, g.kind.values]
                    ok10 = np.abs(pr / t - 1) <= 0.10
                    acc = pd.DataFrame({"lv": lv, "kind": kk, "ok": ok10})
                    a0 = acc[acc.kind == "clean"].ok.mean() * 100
                    a = acc[acc.kind == k].groupby("lv").ok.mean() * 100
                    ax.plot(a.index, a.values, "--", color=CAT[j], lw=1.0, alpha=0.9,
                            label=f"{NAMES.get(m, m)}: 13 legacy, % within 10 % of hand count")
                    ax.axhline(a0, color=CAT[j], lw=0.6, ls=":", alpha=0.7)
        ax.set_title(k, loc="left")
        ax.set_xlabel(D.UNITS[k], fontsize=6.5)
        if k == "jpeg":
            ax.invert_xaxis()
        if k == "down":
            ax.set_xscale("log", base=2)
        ax.set_ylim(-3, 103)
    axs[0, 0].set_ylabel("%")
    axs[1, 0].set_ylabel("%")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, name)


def fig_conf_drift(name="fig_conf_vs_drift", est="ours", p150=None):
    p150 = p150 or os.path.join(RES, "all150_degrade.csv")
    d = _drift_table(pd.read_csv(p150))
    d = d[d.estimator == est]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axs[0]
    y = np.clip(np.where(np.isfinite(d.absdrift), d.absdrift * 100, 300), 0.05, 300)
    ax.scatter(d.conf, y, s=4, color=CAT[0], alpha=0.25, lw=0)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(d.conf, bins) - 1
    med = [np.median(y[idx == b]) if (idx == b).sum() >= 10 else np.nan for b in range(10)]
    ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-o", color=CAT[1], ms=3, lw=1.6, label="binned median")
    ax.set_yscale("log")
    rho, pv = stats.spearmanr(d.conf, np.where(np.isfinite(d.absdrift), d.absdrift, 1e3))  # failures rank last
    ax.set_xlabel("confidence (no-reference)")
    ax.set_ylabel("|prediction drift vs clean| [%]")
    ax.set_title(f"{d.image.nunique()} crops x {d.kind.nunique()} degradation types x levels (n={len(d)})\n"
                 f"Spearman $\\rho$ = {rho:.2f}", loc="left")
    ax.legend(loc="upper right")
    # stability rate per confidence decile
    ax = axs[1]
    q = pd.qcut(d.conf.rank(method="first"), 5, labels=False)
    s = d.groupby(q).absdrift.apply(lambda x: np.mean(x <= 0.05) * 100)
    cm = d.groupby(q).conf.median()
    ax.bar(np.arange(5), s.values, color=CAT[0], width=0.7)
    for i, (v, c) in enumerate(zip(s.values, cm.values)):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7, color=INK)
    ax.set_xticks(np.arange(5), [f"Q{i + 1}\n({c:.2f})" for i, c in enumerate(cm.values)])
    ax.set_xlabel("confidence quintile (median conf.)")
    ax.set_ylabel("% within 5 % of clean")
    ax.set_ylim(0, 105)
    ax.set_title("stability by confidence quintile", loc="left")
    fig.tight_layout()
    save(fig, name)


def fig_legacy_bar(name="fig_legacy_methods"):
    m = pd.read_csv(os.path.join(RES, "legacy13_metrics.csv"))
    m = m[(m.subset == "all13")]
    clean = m[(m.kind.isna()) | (m.kind == "clean")]
    rows = []
    for (meth, assign, axis), g in clean.groupby(["method", "assign", "axis"], dropna=False):
        if g.get("crop") is not None and g.crop.notna().any():
            g = g[(g.crop == "gt") & (g.preproc == "lcn") & (g.glare == "mask")]
        if len(g):
            rows.append(dict(method=meth, assign=assign, axis=axis, MAPE=g.MAPE.iloc[0], w10=g.n_w10.iloc[0], n=g.n.iloc[0]))
    t = pd.DataFrame(rows)
    sel = t[((t["assign"] == "axis") & (t["axis"] == "both")) | (t["method"].str.contains("test.py") & (t["assign"] == "axis"))]
    sel = sel.sort_values("MAPE")
    fig, ax = plt.subplots(figsize=(4.6, 0.28 * len(sel) + 0.8))
    yy = np.arange(len(sel))
    cols = [CAT[1] if "2019" in s else (CAT[0] if s == "ours" else "#9a9993") for s in sel.method]
    ax.barh(yy, np.minimum(sel.MAPE, 200), color=cols, height=0.65)
    for y, (v, w10, n) in zip(yy, sel[["MAPE", "w10", "n"]].values):
        ax.text(min(v, 200) + 2, y, f"{v:.0f}%  ({int(w10)}/{int(n)} within 10%)", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(yy, [NAMES.get(m_, m_) for m_ in sel.method])
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute % error vs hand count (13 legacy images, axis-matched)")
    ax.set_xlim(0, 260)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, name)


def fig_degradations(i=86, name="fig_degradation_examples"):
    """One rectified crop under one level of every degradation (centre 1/4-inch zoom)."""
    import rectify as R
    crop = R.rectify(i)
    conds = [("clean", 0), ("gblur", 3), ("motion", 10), ("down", 4), ("jpeg", 20), ("noise", 20),
             ("glare", 0.10), ("shadow", 0.50), ("corners", 0.05)]
    photo, corners = R.load_image(i), R.load_corners(i)
    fig, axs = plt.subplots(2, len(conds), figsize=(7.0, 1.9))
    for k, (kind, lv) in enumerate(conds):
        d = D.apply(crop, kind, lv, seed=1, photo=photo, corners=corners)
        S = d.shape[0]
        z = d[S // 2 - S // 8:S // 2 + S // 8, S // 2 - S // 8:S // 2 + S // 8]
        for r, im in enumerate((d, z)):
            ax = axs[r, k]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation="nearest" if r else "antialiased")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
        axs[0, k].set_title(f"{kind}\n{lv:g}" if kind != "clean" else "clean\n", fontsize=6.5)
    axs[0, 0].set_ylabel("1 inch", fontsize=6.5)
    axs[1, 0].set_ylabel("1/4 in zoom", fontsize=6.5)
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    save(fig, name)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("lattice", "all"):
        fig_lattice()
    if what in ("profile", "all"):
        fig_profile()
    if what in ("drift", "all"):
        fig_drift()
    if what in ("confdrift", "all"):
        fig_conf_drift()
    if what in ("legacybar", "all"):
        fig_legacy_bar()
    if what in ("degradations", "all"):
        fig_degradations()

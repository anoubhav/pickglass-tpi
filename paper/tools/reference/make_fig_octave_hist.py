"""make_fig_octave_hist.py -- main Fig. 3 (figs/fig_octave_hist_H2.pdf): every count of PG-Lattice, GPT-6.1-sol P1 and
the non-expert annotator (A1) divided by the textile-trained reference annotator's (A2's) count on the same line, as a
histogram over the 152 lines (h and v of 76 scenes), with the per-series tallies of main Sec. 5.2.

  * every count is divided by A2's count on the same line;
  * the annotator series is A1's count / A2's count, legend "non-expert annotator";
  * bins of 1/8 octave from -2 to 2, values clipped at +-1.99 (below 1/4: first bin); half-or-double band
    | |log2 r| - 1 | <= 0.3 (OCT_TOL, the paper's Sec. 5.2 band); size 3.25 x 1.25 in;
  * per-series tallies (within 5%, within 10%, about half, about double, between, beyond the band, about threefold) are
    printed and written to --tallies-json with every line beyond the band listed; the same tallies against A1 are
    printed for comparison (not plotted) and checked against the band counts against A1 (PG-Lattice 42 + 22, the 64
    lines of rule 4; GPT-6.1-sol P1 37 + 8; A2 30 + 18).
Predictions: methods/predictions_ours_150.csv (PG-Lattice) and vlm/results/rect150_P1_predictions_wide.csv
(GPT-6.1-sol P1, <model>__v = vertical yarns = our h), read through paper/tools/score150.py load_predictions() and
rounded to 3 decimals, as paper/tools/label_day.py stores them in the per-scene table the paper's figure was drawn from.
The annotators' counts are read with Python's csv module (h_tpi, v_tpi). Deterministic (no creation date in the PDF);
refuses to overwrite an existing figure file.

    python paper/tools/reference/make_fig_octave_hist.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR] [--tallies-json PATH]

Needs numpy, pandas and matplotlib. Post hoc: choosing A2 as the reference was decided after both annotators' counts
were opened; the band itself is the paper's Sec. 5.2 band, fixed before.
Outputs: <out>/fig_octave_hist_H2.pdf/.png and the tallies (default <out>/octave_tallies.json).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into paper/tools

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi (the 'other' series)")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi (the reference)")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
ap.add_argument("--tallies-json", default=None, help="tallies output (default <out>/octave_tallies.json)")
a = ap.parse_args()

# figure style (the paper's octave figures; Okabe-Ito, one color per actor: counter, GPT, the other annotator)
OCT_TOL = 0.3   # the paper's half-or-double band (Sec. 5.2)
FS = 6.5
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                     "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.3,
                     "ytick.minor.width": 0.3, "pdf.fonttype": 42})
C_METHODS = ("#0072B2", "#CC79A7")
LS_METHODS = ("-", "--")
C_SECOND = "#E69F00"

METHODS = ["Ours (lattice + fusion)", "GPT-6.1-sol, P1"]
LABELS = ["PG-Lattice", "GPT-6.1-sol P1"]
OTHER_LABEL = "non-expert annotator"
XLABEL = "count / reference annotator's count (log scale; 152 lines)"
OUTNAME = "fig_octave_hist_H2"
CREATOR = "paper/tools/reference/make_fig_octave_hist.py"


def read_ref(path):
    """scene_id -> {'h': count, 'v': count} from a per-scene counts CSV (h_tpi, v_tpi)."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["scene_id"])] = {"h": float(r["h_tpi"]), "v": float(r["v_tpi"])}
    return out


def _score150():
    spec = importlib.util.spec_from_file_location("score150_ro", os.path.join(ROOT, "paper", "tools", "score150.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PREDS = _score150().load_predictions()


def load(method, ref):
    """[(axis, reference count, predicted count, scene)] for one method, scenes in ascending order (label_day's
    per-scene table: scenes with both counts predicted, predictions to 3 decimals)."""
    p = PREDS[method][["h", "v"]].copy()
    p.index = p.index.astype(int)
    p = p.dropna().round(3)
    pts = []
    for s in sorted(ref):
        if s not in p.index:
            continue
        for ax in ("h", "v"):
            hu, pr = ref[s][ax], float(p.at[s, ax])
            if hu > 0 and pr > 0:
                pts.append((ax, hu, pr, s))
    return pts


def save(fig, out):
    for ext in (".pdf", ".png"):
        if os.path.exists(out + ext):
            sys.exit(f"refusing to overwrite {out + ext}")
    fig.savefig(out + ".pdf", metadata={"CreationDate": None, "Creator": CREATOR})
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


def hist_fig(series, r2):
    fig = plt.figure(figsize=(3.25, 1.25))
    ax = fig.add_axes([0.085, 0.245, 0.905, 0.585])
    bins = np.arange(-2.0, 2.0001, 0.125)
    lr2 = np.clip(np.log2(r2), -1.99, 1.99)
    ax.hist(lr2, bins=bins, histtype="stepfilled", color=C_SECOND, alpha=0.45, lw=0, zorder=1, label=OTHER_LABEL)
    ax.hist(lr2, bins=bins, histtype="step", color=C_SECOND, lw=0.6, zorder=1)
    for lab, r, col, ls in series:
        ax.hist(np.clip(np.log2(r), -1.99, 1.99), bins=bins, histtype="step", color=col, ls=ls, lw=0.9, label=lab, zorder=3)
    for xv in (-1, 0, 1):
        ax.axvline(xv, color="0.45", lw=0.5, ls=":" if xv else "-", zorder=0)
    ax.set_xlim(-2, 2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    ax.set_xlabel(XLABEL, labelpad=1)
    ax.set_ylabel("lines", labelpad=1)
    ax.tick_params(length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    hs, ls_ = ax.get_legend_handles_labels()
    order = [ls_.index(x) for x in [s[0] for s in series] + [OTHER_LABEL]]
    ax.legend([hs[i] for i in order], [ls_[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=FS - 0.5, handlelength=1.6, borderaxespad=0.1, columnspacing=1.4, handletextpad=0.4)
    save(fig, os.path.join(a.out, OUTNAME))


def tally(lines):
    """lines: [(scene, axis, ref count, count)]. Partition of the lines by r = count / ref count:
    within 10% (|r - 1| <= 0.10; 'within 5%' is a subset), about half / about double (the Sec. 5.2 band,
    | log2 r -/+ 1 | <= 0.3), between (outside 10%, |log2 r| < 0.7) and beyond the band (|log2 r| > 1.3)."""
    r = np.array([c / h for _, _, h, c in lines])
    lr = np.log2(r)
    w5 = np.abs(r - 1) <= 0.05
    w10 = np.abs(r - 1) <= 0.10
    half = np.abs(lr + 1) <= OCT_TOL
    dbl = np.abs(lr - 1) <= OCT_TOL
    beyond = np.abs(lr) > 1 + OCT_TOL
    between = ~w10 & (np.abs(lr) < 1 - OCT_TOL)
    assert int(w10.sum() + half.sum() + dbl.sum() + between.sum() + beyond.sum()) == len(r), "not a partition"
    sc = lambda m: len({lines[i][0] for i in np.flatnonzero(m)})  # noqa: E731
    out = dict(n_lines=len(r), within_5pct=int(w5.sum()), within_10pct=int(w10.sum()),
               about_half=int(half.sum()), about_double=int(dbl.sum()), band_total=int((half | dbl).sum()),
               band_scenes=sc(half | dbl), between_10pct_and_band=int(between.sum()),
               beyond_band=int(beyond.sum()), beyond_band_scenes=sc(beyond),
               beyond_above=int((lr > 1 + OCT_TOL).sum()), beyond_below=int((lr < -1 - OCT_TOL).sum()),
               about_threefold=int(((np.abs(r / 3 - 1) <= 0.10) | (np.abs(3 * r - 1) <= 0.10)).sum()),
               about_threefold_scenes=sc((np.abs(r / 3 - 1) <= 0.10) | (np.abs(3 * r - 1) <= 0.10)),
               clipped_in_plot=int((np.abs(lr) >= 1.99).sum()),
               differ_gt10pct=int((~w10).sum()), differ_gt10pct_in_band=int((~w10 & (half | dbl)).sum()),
               median_abs_log2=round(float(np.median(np.abs(lr))), 3),
               beyond_band_lines=[dict(scene=lines[i][0], axis=lines[i][1], ref=lines[i][2], count=round(lines[i][3], 3),
                                       ratio=round(float(r[i]), 3)) for i in np.flatnonzero(beyond)])
    return out


def fmt(lab, t):
    return (f"{lab}: {t['n_lines']} lines | within 5% {t['within_5pct']}, within 10% {t['within_10pct']} | "
            f"about half {t['about_half']}, about double {t['about_double']} (band {t['band_total']} lines, "
            f"{t['band_scenes']} scenes) | between 10% and band {t['between_10pct_and_band']} | beyond band "
            f"{t['beyond_band']} ({t['beyond_above']} above, {t['beyond_below']} below; {t['beyond_band_scenes']} scenes; "
            f"{t['clipped_in_plot']} drawn in an edge bin) | about threefold (r within 10% of 3 or 1/3, any class) "
            f"{t['about_threefold']} lines, {t['about_threefold_scenes']} scenes")


ref, other = read_ref(a.a2), read_ref(a.a1)
series, res = [], {}
for i, (m, lab) in enumerate(zip(METHODS, LABELS)):
    pts = load(m, ref)
    assert len(pts) == 152, (m, len(pts))
    r = np.array([pr / hu for _, hu, pr, _ in pts])
    series.append((lab, r, C_METHODS[i], LS_METHODS[i]))
    res[lab] = tally([(s, ax_, hu, pr) for ax_, hu, pr, s in pts])
    print(fmt(lab, res[lab]))
lines2 = [(s, ax_, ref[s][ax_], other[s][ax_]) for s in sorted(ref) for ax_ in "hv"
          if s in other and ref[s][ax_] > 0 and other[s][ax_] > 0]
r2 = np.array([c / h for _, _, h, c in lines2])
assert len(r2) == 152, len(r2)
res[OTHER_LABEL] = tally(lines2)
print(fmt(OTHER_LABEL, res[OTHER_LABEL]))
# For comparison only (not plotted): the same tallies against A1; the band counts must equal PG-Lattice 42 + 22 (the 64
# lines of rule 4), GPT-6.1-sol P1 37 + 8, A2 30 + 18.
cmp_ = {}
for m, lab in zip(METHODS, LABELS):
    cmp_[lab] = tally([(s, ax_, hu, pr) for ax_, hu, pr, s in load(m, other)])
cmp_["A2 vs A1"] = tally([(s, ax_, other[s][ax_], ref[s][ax_]) for s in sorted(other) for ax_ in "hv"
                          if s in ref and other[s][ax_] > 0 and ref[s][ax_] > 0])
got = [(cmp_[k]["about_half"], cmp_[k]["about_double"]) for k in (*LABELS, "A2 vs A1")]
assert got == [(42, 22), (37, 8), (30, 18)], got
print("check: against A1 the band tallies are", got)
for k, t in cmp_.items():
    print("  [vs A1, not plotted] " + fmt(k, t))
os.makedirs(a.out, exist_ok=True)
hist_fig(series, r2)
tj = a.tallies_json or os.path.join(a.out, "octave_tallies.json")
meta = dict(script=CREATOR, reference="A2 per-scene counts", other="A1 per-scene counts",
            predictions="paper/tools/score150.py load_predictions(): methods/predictions_ours_150.csv, "
                        "vlm/results/rect150_P1_predictions_wide.csv; rounded to 3 decimals as label_day.py's per-scene table",
            oct_tol=OCT_TOL,
            definitions="r = count / reference count per line (h and v of 76 scenes = 152 lines); within 10%: "
                        "|r-1|<=0.10; about half: |log2 r + 1|<=0.3; about double: |log2 r - 1|<=0.3; between: "
                        "outside 10% and |log2 r|<0.7; beyond band: |log2 r|>1.3; the five classes partition the "
                        "152 lines; clipped_in_plot: |log2 r|>=1.99, drawn in the -2 or +2 edge bin",
            post_hoc="choice of A2 as reference is post hoc; the band is the paper's Sec. 5.2 band")
with open(tj, "w") as f:
    json.dump(dict(meta=meta, series=res, vs_A1_not_plotted=cmp_), f, indent=1)
print("wrote", tj)

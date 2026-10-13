"""Paper figure: best-assignment MAE (TPI) per roster model and prompt on the 13 legacy GT images.

Cleveland dot plot; P1/P2/P3 use the first three categorical slots of the reference palette
(validated all-pairs); the leave-one-out mean predictor is a dashed reference line.
Run with any python that has matplotlib (e.g. the shared venv2):

    python make_summary_plot.py   -> figs/mae_by_model_prompt.{png,pdf}
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "stageA_legacy_metrics.csv"))))
summ = json.load(open(os.path.join(HERE, "results", "summary.json")))
loo = summ["loo_mean_baseline"]["best"]["mae"]

SERIES = [("P1", "P1 direct JSON", "#2a78d6", "o"),
          ("P2", "P2 step-by-step", "#eb6834", "s"),
          ("P3", "P3 quarter-inch patch x4", "#1baf7a", "^")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

models = []
for r in rows:
    if "@" in r["model"] or r["method"] not in ("P1", "P2", "P3"):
        continue
    if r["label"] not in models:
        models.append(r["label"])
val = {(r["label"], r["method"]): (float(r["mae_best"]) if r["mae_best"] else None,
                                   int(r["n_ok"]), int(r["n_gt"]), int(r.get("not_run_gt") or 0)) for r in rows}
best = {m: min(v[0] for (mm, p), v in val.items() if mm == m and v[0] is not None and p in ("P1", "P2", "P3"))
        for m in models}
models.sort(key=lambda m: best[m], reverse=True)       # best at the top

fig, ax = plt.subplots(figsize=(6.8, 3.7), dpi=200)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=8)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for i, m in enumerate(models):
    ax.axhline(i, color=GRID, linewidth=0.6, zorder=0)
for k, (pid, lab, col, mk) in enumerate(SERIES):
    xs, ys = [], []
    for i, m in enumerate(models):
        v = val.get((m, pid))
        if v and v[0] is not None:
            xs.append(v[0])
            ys.append(i + (1 - k) * 0.22)
            if v[1] < v[2]:          # annotate coverage: parse failures or a partial (not yet run) set
                lab_ = f"run {v[2] - v[3]}/{v[2]}" if v[3] else f"{v[1]}/{v[2]}"
                ax.annotate(lab_, (v[0], i + (1 - k) * 0.22), xytext=(5, -3),
                            textcoords="offset points", fontsize=6.5, color=INK2)
    ax.scatter(xs, ys, s=34, color=col, marker=mk, edgecolor="white", linewidth=1.2, label=lab, zorder=3)
ax.axvline(loo, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
ax.text(loo, len(models) - 0.45, f" mean predictor (LOO) {loo:.1f}", fontsize=7, color=INK2, va="bottom")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=8, color=INK)
ax.set_ylim(-0.6, len(models) - 0.1)
ax.set_xlim(0, None)
ax.set_xlabel("MAE, threads per inch (best axis assignment, 13 legacy images)", fontsize=8, color=INK)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=7.5,
          labelcolor=INK, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figs", f"mae_by_model_prompt.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figs/mae_by_model_prompt.{png,pdf}")

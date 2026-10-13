"""analyze_real.py -- statistics for the one-shot real evaluation (eval_real.py -> results/real_counts.csv).

References (as in the genai_sr protocol, README section 2 / 6):
  R-LEG / cascade legacy-13: human hand counts genai_sr/inputs/gt_legacy13.csv (axis: h <-> warp, v <-> weft)
  R-HR: ours (UNGUARDED) on the clean reference, computed in the same run (pseudo-GT); a second
        variant uses the guarded clean prediction (identical when the guard does not fire on the ref)
APE = |pred - ref| / ref x 100 per count; a missing count = APE 100.  MAPE = mean over both counts
of all images; 95 % percentile bootstrap over images (10,000 resamples, both counts of an image kept
together, seed 20261008).  Guarded vs unguarded: paired bootstrap of dMAPE (same resamples), Wilcoxon
signed-rank on per-image mean APE (zero differences dropped; p = 1 when all are zero).
Clean crops: share whose count changes by > 1 % / > 5 % (max over the two axes, guarded vs unguarded).

    <venv2>/bin/python analyze_real.py      -> results/real_table_*.csv, results/real_tables.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GSR = os.path.join(ROOT, "genai_sr")
RES = os.path.join(HERE, "results")
B = 10000
SEED = 20261008
REST_ORDER = ["none", "bicubic", "swinir", "realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
CASCADE_CONDS = ["clean", "gblur3", "gblur6", "down4", "down8", "jpeg20", "glare0.05", "glare0.3", "corners0.05"]


def load():
    df = pd.read_csv(os.path.join(RES, "real_counts.csv"))
    gt = pd.read_csv(os.path.join(GSR, "inputs", "gt_legacy13.csv"))
    gt_key = {r.key: (r.warp, r.weft) for r in gt.itertuples()}
    gt_img = {r.image: (r.warp, r.weft) for r in gt.itertuples()}
    clean = df[(df.group == "genai") & (df.restorer == "clean")].set_index("key")
    ref_h, ref_v, ref2_h, ref2_v = [], [], [], []
    for r in df.itertuples():
        if r.group == "genai" and r.set == "R-LEG":
            a = gt_key[r.key]
            b = a
        elif r.group == "genai":
            a = (clean.loc[r.key, "ug_h"], clean.loc[r.key, "ug_v"])
            b = (clean.loc[r.key, "g_h"], clean.loc[r.key, "g_v"])
        elif r.group == "cascade":
            a = gt_img[r.key]
            b = a
        else:
            a = b = (np.nan, np.nan)
        ref_h.append(a[0]), ref_v.append(a[1]), ref2_h.append(b[0]), ref2_v.append(b[1])
    df["ref_h"], df["ref_v"], df["ref2_h"], df["ref2_v"] = ref_h, ref_v, ref2_h, ref2_v
    for p in ("ug", "g"):
        for ax in ("h", "v"):
            ape = 100 * (df[f"{p}_{ax}"] / df[f"ref_{ax}"] - 1).abs()
            df[f"ape_{p}_{ax}"] = ape.where(df[f"{p}_{ax}"].notna(), 100.0)
            ape2 = 100 * (df[f"{p}_{ax}"] / df[f"ref2_{ax}"] - 1).abs()
            df[f"ape2_{p}_{ax}"] = ape2.where(df[f"{p}_{ax}"].notna(), 100.0)
        df[f"ape_{p}"] = 0.5 * (df[f"ape_{p}_h"] + df[f"ape_{p}_v"])
        df[f"ape2_{p}"] = 0.5 * (df[f"ape2_{p}_h"] + df[f"ape2_{p}_v"])
        df[f"w10_{p}"] = (df[f"ape_{p}_h"] <= 10).astype(float) + (df[f"ape_{p}_v"] <= 10).astype(float)
    chg = np.fmax((df.g_h / df.ug_h - 1).abs(), (df.g_v / df.ug_v - 1).abs())
    df["change"] = chg.where(np.isfinite(chg), np.where(df.fired, 1.0, 0.0))
    return df


def boot_idx(n, rng):
    return rng.integers(0, n, size=(B, n))


def cell_stats(d, col="ape", cluster=None):
    """d: one row per image (or, with cluster=<column>, several rows per cluster: the bootstrap then
    resamples clusters, e.g. the 13 legacy photos across conditions, as the paper's cascade table)."""
    rng = np.random.default_rng(SEED)
    u, g = d[f"{col}_ug"].to_numpy(float), d[f"{col}_g"].to_numpy(float)
    n = len(d)
    if cluster is None:
        idx = boot_idx(n, rng)
        bu, bg = u[idx].mean(1), g[idx].mean(1)
    else:
        keys = d[cluster].to_numpy()
        uk = sorted(set(keys))
        members = [np.nonzero(keys == k)[0] for k in uk]
        bu, bg = np.empty(B), np.empty(B)
        for i in range(B):
            sel = np.concatenate([members[j] for j in rng.integers(0, len(uk), len(uk))])
            bu[i], bg[i] = u[sel].mean(), g[sel].mean()
    bd = bg - bu
    diff = g - u
    nz = diff[np.abs(diff) > 1e-9]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    return dict(n=n, mape_ug=u.mean(), ug_lo=np.percentile(bu, 2.5), ug_hi=np.percentile(bu, 97.5),
                mape_g=g.mean(), g_lo=np.percentile(bg, 2.5), g_hi=np.percentile(bg, 97.5),
                d=g.mean() - u.mean(), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5), wilcoxon_p=p,
                n_better=int((diff < -1e-9).sum()), n_worse=int((diff > 1e-9).sum()),
                w10_ug=100 * d.w10_ug.sum() / (2 * n), w10_g=100 * d.w10_g.sum() / (2 * n),
                fire=100 * d.fired.mean(), eligible=100 * d.eligible.mean(), flag_up=100 * d.upsampled.mean(),
                flag_unsup=100 * d.unsupported.mean(), flag_insuff=100 * d.insufficient.mean(),
                chg1=100 * (d.change > 0.01).mean(), chg5=100 * (d.change > 0.05).mean(),
                ppy_med=float(np.nanmedian(d.ppy)), neff_rel_med=float(np.nanmedian(d.n_eff / d.S)))


def to_md(df, nd=2):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.{nd}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def fmt_ci(m, lo, hi, nd=1):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def genai_table(df):
    rows = []
    G = df[df.group == "genai"]
    for s in ("R-LEG", "R-HR"):
        for rest in ["clean"] + REST_ORDER:
            for cond in (["clean"] if rest == "clean" else ["D4", "D8", "D4J"]):
                d = G[(G.set == s) & (G.restorer == rest) & (G.cond == cond)]
                if d.empty:
                    continue
                st = cell_stats(d)
                st2 = cell_stats(d, "ape2") if s == "R-HR" else {}
                rows.append(dict(set=s, restorer=rest, cond=cond, **st,
                                 mape_g_vs_guarded_ref=st2.get("mape_g", np.nan)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "real_table_genai.csv"), index=False)
    return T


def clean_table(df):
    rows = []
    sets = [("clean150 (all 150 clean crops)", df[df.group == "clean150"]),
            ("clean150, S < 1000 px (85 low-res captures)", df[(df.group == "clean150") & (df.S < 1000)]),
            ("clean150, S >= 1000 px", df[(df.group == "clean150") & (df.S >= 1000)]),
            ("genai clean references R-HR (53)", df[(df.group == "genai") & (df.restorer == "clean") & (df.set == "R-HR")]),
            ("genai clean references R-LEG (13)", df[(df.group == "genai") & (df.restorer == "clean") & (df.set == "R-LEG")]),
            ("cascade legacy-13 clean", df[(df.group == "cascade") & (df.cond == "clean")]),
            ("genai 'none' LR inputs (198, native low-res)", df[(df.group == "genai") & (df.restorer == "none")])]
    for name, d in sets:
        rows.append(dict(set=name, n=len(d), fired=int(d.fired.sum()), fire_pct=100 * d.fired.mean(),
                         eligible_pct=100 * d.eligible.mean(), unsupported_pct=100 * d.unsupported.mean(),
                         chg1_pct=100 * (d.change > 0.01).mean(), chg5_pct=100 * (d.change > 0.05).mean(),
                         n_chg1=int((d.change > 0.01).sum()), n_chg5=int((d.change > 0.05).sum()),
                         dead_band_pct=100 * d.dead_band.mean(), flag_up_pct=100 * d.upsampled.mean(),
                         flag_insuff_pct=100 * d.insufficient.mean(), ppy_median=float(np.nanmedian(d.ppy)),
                         neff_over_S_median=float(np.nanmedian(d.n_eff / d.S)),
                         neff_over_S_p05=float(np.nanpercentile(d.n_eff / d.S, 5))))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "real_table_clean.csv"), index=False)
    return T


def cascade_table(df):
    C = df[df.group == "cascade"]
    rows = []
    for cond in CASCADE_CONDS + sorted(set(C.cond) - set(CASCADE_CONDS)):
        d = C[C.cond == cond]
        if d.empty:
            continue
        rows.append(dict(cond=cond, in_cascade_vlm_set=cond in CASCADE_CONDS, **cell_stats(d)))
    # pooled over the 9 cascade conditions (117 photos), as cascade_legacy13.csv
    d = C[C.cond.isin(CASCADE_CONDS)]
    rows.append(dict(cond="ALL 9 cascade conditions (cluster bootstrap over the 13 photos)", in_cascade_vlm_set=True,
                     **cell_stats(d, cluster="key")))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "real_table_cascade.csv"), index=False)
    return T


COND_TAG = {"clean": "clean", "defocus 3u": "gblur3", "defocus 6u": "gblur6", "down x4": "down4", "down x8": "down8",
            "JPEG q20": "jpeg20", "glare 5%": "glare0.05", "glare 30%": "glare0.3", "corners 5%": "corners0.05"}


def cascade_vs_vlm(df):
    """Guarded ours (no VLM) next to the stored cascade policies on the same 117 legacy photos.
    The VLM numbers are the stored GPT-6.1-sol P1 predictions of paper/tables/cascade_legacy13_inputs.csv
    (no new API call).  'guard + flag -> VLM' rows are EXPLORATORY (not pre-planned): the guard's own
    flags used as the escalation router instead of the cascade's (conf, rho) gate."""
    inp = pd.read_csv(os.path.join(ROOT, "paper", "tables", "cascade_legacy13_inputs.csv"))
    inp["tag"] = inp.cond.map(COND_TAG)
    C = df[df.group == "cascade"].merge(inp[["image", "tag", "gt_w", "gt_f", "ours_w", "ours_f", "conf", "rho",
                                             "gpt_w", "gpt_f", "gpt_cost"]],
                                        left_on=["key", "cond"], right_on=["image", "tag"], how="inner")
    assert len(C) == 117, len(C)
    thr = __import__("json").load(open(os.path.join(ROOT, "paper", "tables", "cascade_thresholds.json")))
    g = thr["fits"]["vlm9_a10"]["gate"]
    esc_gate = (C.conf < g["t_c"]) | (C.rho < g["r_min"])

    def ape(h, v):
        return 0.5 * (100 * np.abs(h / C.gt_w - 1) + 100 * np.abs(v / C.gt_f - 1))
    pol = {
        "ours unguarded (this run)": (C.ug_h, C.ug_v, np.zeros(len(C), bool)),
        "ours guarded (no VLM)": (C.g_h, C.g_v, np.zeros(len(C), bool)),
        "cascade gate: ours -> VLM if conf < t_c or rho < r_min (paper)": (
            np.where(esc_gate, C.gpt_w, C.ug_h), np.where(esc_gate, C.gpt_f, C.ug_v), esc_gate.to_numpy()),
        "VLM only (GPT-6.1-sol P1, stored)": (C.gpt_w, C.gpt_f, np.ones(len(C), bool)),
    }
    for name, esc in (("EXPLORATORY guarded -> VLM if guard fired (unsupported claim)", C.fired.to_numpy()),
                      ("EXPLORATORY guarded -> VLM if fired or insufficient_resolution",
                       (C.fired | C.insufficient).to_numpy())):
        pol[name] = (np.where(esc, C.gpt_w, C.g_h), np.where(esc, C.gpt_f, C.g_v), esc)
    rng = np.random.default_rng(SEED)
    imgs = sorted(C.key.unique())
    idx_of = {k: np.nonzero(C.key.to_numpy() == k)[0] for k in imgs}
    boots = [np.concatenate([idx_of[imgs[j]] for j in rng.integers(0, len(imgs), len(imgs))]) for _ in range(B)]
    base = ape(C.ug_h, C.ug_v).to_numpy()
    rows = []
    for name, (h, v, esc) in pol.items():
        a = ape(np.asarray(h, float), np.asarray(v, float)).to_numpy()
        bm = np.array([a[b].mean() for b in boots])
        bd = np.array([a[b].mean() - base[b].mean() for b in boots])
        per = pd.Series(a).groupby(C.tag.to_numpy()).mean()
        cost = float(np.where(esc, C.gpt_cost, 0).sum() / len(C) * 1000)
        rows.append(dict(policy=name, MAPE=a.mean(), lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5),
                         dMAPE_vs_unguarded=a.mean() - base.mean(), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5),
                         worst_cond=str(per.idxmax()), worst_MAPE=float(per.max()), escalated_pct=100 * np.mean(esc),
                         usd_per_1k=cost, **{f"MAPE_{t}": float(per.get(t, np.nan)) for t in CASCADE_CONDS}))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "real_table_cascade_vs_vlm.csv"), index=False)
    return T


def repro_checks(df):
    """Unguarded counts of this run vs the stored study counts (same frozen estimator)."""
    out = []
    G = df[df.group == "genai"]
    for rest in ["clean"] + REST_ORDER:
        p = os.path.join(GSR, "results", f"counts_classical_{rest}.csv")
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p)
        s = s.rename(columns={"md5": "md5_study"})
        m = G[G.restorer == rest].merge(s[["key", "cond", "ours_h", "ours_v", "md5_study"]], on=["key", "cond"], how="inner")
        if m.empty:
            continue
        rel = np.fmax((m.ug_h / m.ours_h - 1).abs(), (m.ug_v / m.ours_v - 1).abs())
        out.append(dict(source=f"genai_sr counts_classical_{rest}", n=len(m), md5_equal=int((m.md5_study == m.md5_read).sum()),
                        identical=int((rel < 1e-9).sum()), max_rel_diff=float(np.nanmax(rel))))
    s = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_degrade.csv"))
    s = s[s.estimator == "ours"]
    m = df[df.group == "cascade"].merge(s[["image", "tag", "h", "v"]], left_on=["key", "cond"], right_on=["image", "tag"])
    rel = np.fmax((m.ug_h / m.h - 1).abs(), (m.ug_v / m.v - 1).abs())
    out.append(dict(source="methods/results/legacy13_vlmcrops_degrade.csv (ours)", n=len(m), md5_equal=np.nan,
                    identical=int((rel < 1e-9).sum()), max_rel_diff=float(np.nanmax(rel))))
    s = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv"))
    m = df[df.group == "clean150"].assign(id=lambda x: x.key.astype(int)).merge(s[["id", "h_count", "v_count"]], on="id")
    rel = np.fmax((m.ug_h / m.h_count - 1).abs(), (m.ug_v / m.v_count - 1).abs())
    out.append(dict(source="methods/predictions_ours_150.csv", n=len(m), md5_equal=np.nan,
                    identical=int((rel < 1e-9).sum()), max_rel_diff=float(np.nanmax(rel))))
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(RES, "real_repro_checks.csv"), index=False)
    return T


def md_genai(T):
    lines = []
    for s in ("R-LEG", "R-HR"):
        lines.append(f"\n**{s}** ({'human hand counts' if s == 'R-LEG' else 'pseudo-GT = unguarded ours on the clean reference'})\n")
        lines.append("| input | cond | n | MAPE unguarded [95 % CI] | MAPE guarded [95 % CI] | dMAPE [95 % CI] | Wilcoxon p | better / worse | dead band (eligible) | fires | flag insufficient |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in T[T.set == s].itertuples():
            lines.append(f"| {r.restorer} | {r.cond} | {r.n} | {fmt_ci(r.mape_ug, r.ug_lo, r.ug_hi)} | "
                         f"{fmt_ci(r.mape_g, r.g_lo, r.g_hi)} | {fmt_ci(r.d, r.d_lo, r.d_hi)} | {r.wilcoxon_p:.2g} | "
                         f"{r.n_better} / {r.n_worse} | {r.eligible:.0f} % | {r.fire:.0f} % | {r.flag_insuff:.0f} % |")
    return "\n".join(lines)


def main():
    df = load()
    T = genai_table(df)
    CT = clean_table(df)
    K = cascade_table(df)
    R = repro_checks(df)
    V = cascade_vs_vlm(df)
    pd.set_option("display.width", 250)
    md = ["## genai_sr inputs", md_genai(T), "\n## clean crops", to_md(CT),
          "\n## cascade degraded legacy-13", to_md(K),
          "\n## cascade legacy-13 (117 photos): guarded ours vs the stored VLM cascade", to_md(V),
          "\n## reproduction of stored unguarded counts", to_md(R, 6)]
    open(os.path.join(RES, "real_tables.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())

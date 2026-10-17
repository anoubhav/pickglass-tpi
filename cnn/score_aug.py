#!/usr/bin/env python
"""score_aug.py -- report numbers of the augmentation study of train_aug.py (run ONCE on the Mac after all
selection; venv2).

Reads the synced GPU results in cnn/runs_aug/ (selection.json, final_spec.json, final/<variant>/...) and
writes cnn/runs_aug/report/:
  legacy13_aug.csv           legacy-13 scores per final variant (axis-matched primary, 'sorted' secondary):
                             MAE, MAPE + image-bootstrap 95 % CI (10k), within 5 / 10 %, bias
  legacy13_aug_paired.csv    paired tests vs the MPS synthetic-only CNN (cnn:synth_r18) and vs ours
                             (paper/tools/legacy_unified.paired) + Holm over the 10 final variants
  legacy13_aug_per_count.csv every scored count
  confirmation_bias_*.csv    student vs teacher (ours) where ours is unreliable: synth_val_v2 (exact
                             labels), real150 (scene representatives: disagreement by the scene split /
                             ours' confidence), legacy-13 (ours' failures)
  ablation_stage2.csv        drop-one in the V4-R18 fine-tuning (+ fine-tuning seeds), selection metrics
  ablation_stage1.csv        drop-one synthetic pre-training (adopted whitelist_ext runs), selection metrics
  real150_aug_wide.csv       150-crop predictions of every final variant in score150.py's --extra-wide
                             format (key__v = our h); real150_aug_split.csv says which files were
                             pseudo-label TRAINING scenes for V3-V5 (exclude them when scoring vs humans);
                             real150_aug_wide_heldout.csv = the same with V3-V5 blanked on those scenes
  gpu_hours.json, parity.json, tables.md
Scene split: see train_aug.py and runs_aug/PREREG.json.
The legacy GT, crop rule, scoring and paired tests are imported from paper/tools/legacy_unified.py, so
the numbers are directly comparable with tables/legacy13_unified.csv (CNN 16.0 %, ours 11.6 %).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs_aug")
OUT = os.path.join(RUNS, "report")
sys.path.insert(0, os.path.join(ROOT, "paper", "tools"))
import legacy_unified as LU  # noqa: E402

LABEL = {"V1": "V1 synthetic, existing aug (GPU re-run)", "V2": "V2 synthetic + whitelist aug",
         "V3": "V3 pseudo-label self-training (real only)", "V4": "V4 synthetic -> pseudo-real fine-tune",
         "V5": "V5 = V4 + TTA"}
BB = {"r18": "ResNet-18", "b0": "EfficientNet-B0"}
RULE_CONF = None


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def boot_ci_mean(x, B=10000, seed=0):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f.format(v) if (f and isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(open(os.path.join(RUNS, "final_spec.json")))
    sel = json.load(open(os.path.join(RUNS, "selection.json")))
    prereg = json.load(open(os.path.join(RUNS, "PREREG.json")))
    conf_cut = prereg["split_info"]["conf_median_76_reps"]
    gt = LU.load_gt()
    cnn_ref = LU.cnn()[0]
    ours = [m for m in LU.classical() if m["key"] == "est:ours"][0]
    md = []

    # ------------------------------------------------------------------ legacy-13 (once per final variant)
    variants = [v for v in spec if v["name"] != "REF_mps_r18"]
    ms = []
    for v in variants:
        d = pd.read_csv(os.path.join(RUNS, "final", v["name"], "legacy_predictions_vlm.csv")).set_index("name")
        k, bb = v["name"].split("_")
        ms.append(LU.P("learned", f"{LABEL[k]}, {BB[bb]}", "--", d.pred_h, d.pred_v, key=f"aug:{v['name']}",
                       input_px="cap 1024, " + ("TTA 3 scales x 5 views x 3 rotations" if v.get("tta") else
                                                "3 scales x 3 flips"), note=f"ckpt {v['ckpt']}"))
    rows, counts = [], []
    for m in [cnn_ref, ours] + ms:
        for assign in ("axis", "sorted"):
            r, d = LU.score(m, gt, assign)
            rows.append({"key": m["key"], "method": m["method"], "assign": assign, **r})
            counts.append(d.assign(key=m["key"], assign=assign))
    S = pd.DataFrame(rows)
    S.round(3).to_csv(os.path.join(OUT, "legacy13_aug.csv"), index=False)
    pd.concat(counts).round(3).to_csv(os.path.join(OUT, "legacy13_aug_per_count.csv"), index=False)
    PR = []
    for comp_name, comp in (("CNN synthetic-only (MPS, 16.0%)", cnn_ref), ("ours (11.6%)", ours)):
        for assign in ("axis", "sorted"):
            block = [LU.paired(m, comp, gt, assign, f"{m['key']} vs {comp_name}") for m in ms]
            b = pd.DataFrame(block)
            b["comparator"] = comp_name
            b["holm_p_boot_MAPE"] = holm(b.p_boot_MAPE)
            b["holm_p_wilcoxon_abs_pct_err_26"] = holm(b.p_wilcoxon_abs_pct_err_26.fillna(1.0))
            b["holm_p_wilcoxon_image_13"] = holm(b.p_wilcoxon_image_mean_pct_err_13.fillna(1.0))
            PR.append(b)
    PR = pd.concat(PR)
    PR.round(4).to_csv(os.path.join(OUT, "legacy13_aug_paired.csv"), index=False)
    ax = S[S["assign"] == "axis"].copy()
    ax["MAPE [95% CI]"] = [f"{r.MAPE:.1f} [{r.MAPE_lo:.1f}, {r.MAPE_hi:.1f}]" for r in ax.itertuples()]
    pa = PR[PR["assign"] == "axis"]

    def dcell(key, comp):
        r = pa[(pa.A == key) & (pa.comparator == comp)]
        if not len(r):
            return "--"
        r = r.iloc[0]
        return (f"{r.dMAPE:+.1f} [{r.dMAPE_lo:+.1f}, {r.dMAPE_hi:+.1f}]; p_W={r.p_wilcoxon_abs_pct_err_26:.3f} "
                f"(Holm {r.holm_p_wilcoxon_abs_pct_err_26:.2f})")
    ax["vs CNN-synth: dMAPE [CI]; Wilcoxon"] = [dcell(k, "CNN synthetic-only (MPS, 16.0%)") for k in ax.key]
    ax["vs ours: dMAPE [CI]; Wilcoxon"] = [dcell(k, "ours (11.6%)") for k in ax.key]
    so = S[S["assign"] == "sorted"].set_index("key").MAPE
    ax["MAPE sorted"] = [so[k] for k in ax.key]
    md.append("### Legacy-13 (axis-matched, 26 counts; reported once per final variant)\n")
    md.append(md_table(ax, ["method", "MAE", "MAPE [95% CI]", "n_w5", "n_w10", "bias", "MAPE sorted",
                            "vs CNN-synth: dMAPE [CI]; Wilcoxon", "vs ours: dMAPE [CI]; Wilcoxon"],
                       {"MAE": "{:.2f}", "bias": "{:+.2f}", "MAPE sorted": "{:.1f}"}))

    # forest plot (legacy-13, axis-matched MAPE with image-bootstrap 95 % CI)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axp = plt.subplots(figsize=(6.8, 4.4))
        rr = ax.reset_index(drop=True)
        yy = np.arange(len(rr))[::-1]
        axp.errorbar(rr.MAPE, yy, xerr=[rr.MAPE - rr.MAPE_lo, rr.MAPE_hi - rr.MAPE], fmt="o", color="#333", ms=4,
                     capsize=2, lw=1)
        for x_, c_, l_ in ((rr.set_index("key").MAPE["cnn:synth_r18"], "#c44", "CNN synthetic-only (MPS)"),
                           (rr.set_index("key").MAPE["est:ours"], "#36c", "ours")):
            axp.axvline(x_, color=c_, ls="--", lw=0.8, label=f"{l_}: {x_:.1f}%")
        axp.set_yticks(yy)
        axp.set_yticklabels([m.replace("synthetic", "synth.").replace(" (GPU re-run)", "") for m in rr.method], fontsize=7)
        axp.set_xlabel("legacy-13 MAPE (%), axis-matched\n95% CI, bootstrap over images", fontsize=8)
        axp.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figs", "aug_training_legacy13.pdf"))
        fig.savefig(os.path.join(HERE, "figs", "aug_training_legacy13.png"), dpi=160)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)

    # ------------------------------------------------------------------ selection metrics of the final variants
    st = pd.read_csv(os.path.join(RUNS, "selection_table.csv"))
    md.append("\n### Selection metrics (synth_val + held-out pseudo-real; the only basis for every choice)\n")
    md.append(md_table(st, ["variant", "run", "step", "synth_MAPE", "synth_acc10", "synth_logratio_r", "pseudo_MAPE",
                            "pseudo_acc10", "pseudo_bias", "score"],
                       {c: "{:.2f}" for c in ("synth_MAPE", "synth_acc10", "synth_logratio_r", "pseudo_MAPE",
                                              "pseudo_acc10", "pseudo_bias", "score")}))
    cand = []
    for k, v in sel.items():
        for c in v.get("candidates", []):
            if isinstance(c, dict):
                cand.append({"variant": k, "candidate": c["run"], "step": c["step"], "synth_MAPE": c["synth_MAPE"],
                             "pseudo_MAPE": c["pseudo_MAPE"], "score": c["score"], "chosen": c["run"] == v["run"]})
    if cand:
        cd = pd.DataFrame(cand)
        cd.to_csv(os.path.join(OUT, "selection_candidates.csv"), index=False)
        md.append("\nCandidates compared by the pre-declared score:\n")
        md.append(md_table(cd, ["variant", "candidate", "step", "synth_MAPE", "pseudo_MAPE", "score", "chosen"],
                           {"synth_MAPE": "{:.2f}", "pseudo_MAPE": "{:.2f}", "score": "{:.2f}"}))

    # ------------------------------------------------------------------ synth_val_v2 (exact labels, never used for selection)
    osy = pd.read_csv(os.path.join(RUNS, "ours_synth_val_v2.csv"))
    osy["acc"] = (osy.confidence >= conf_cut) & (osy.f_fused_lattice == 1)
    osy["ape_ours"] = 50 * ((osy.ours_h - osy.h_count).abs() / osy.h_count + (osy.ours_v - osy.v_count).abs() / osy.v_count)
    cb, sv = [], []
    for v in spec:
        p = pd.read_csv(os.path.join(RUNS, "final", v["name"], "synth_val_v2_predictions.csv"))
        p["ape"] = 50 * ((p.pred_h - p.h_count).abs() / p.h_count + (p.pred_v - p.v_count).abs() / p.v_count)
        p["ape_single"] = 50 * ((p.pred_h_single - p.h_count).abs() / p.h_count +
                                (p.pred_v_single - p.v_count).abs() / p.v_count)
        e = np.concatenate([(p.pred_h - p.h_count).abs() / p.h_count, (p.pred_v - p.v_count).abs() / p.v_count])
        sv.append({"variant": v["name"], "MAPE": 100 * e.mean(), "acc10": 100 * (e <= .1).mean(),
                   "MAPE_single_scale": p.ape_single.mean()})
        j = p.merge(osy[["file", "ape_ours", "acc", "confidence"]], on="file")
        for subset, mk in (("ours accepted (rule)", j.acc), ("ours rejected (rule)", ~j.acc),
                           ("ours gross (APE>25%)", j.ape_ours > 25), ("all", np.ones(len(j), bool))):
            g = j[mk]
            if not len(g):
                continue
            dlt = g.ape - g.ape_ours
            lo, hi = boot_ci_mean(dlt)
            w = stats.wilcoxon(dlt).pvalue if len(g) > 5 and np.any(dlt != 0) else np.nan
            cb.append({"variant": v["name"], "subset": subset, "n_images": len(g), "MAPE_student": g.ape.mean(),
                       "MAPE_ours": g.ape_ours.mean(), "dMAPE": dlt.mean(), "dMAPE_lo": lo, "dMAPE_hi": hi,
                       "p_wilcoxon": w, "student_better_share": float((g.ape < g.ape_ours).mean()),
                       "student_gross_share": float((g.ape > 25).mean())})
    sv = pd.DataFrame(sv)
    sv.to_csv(os.path.join(OUT, "synth_val_v2.csv"), index=False)
    cb = pd.DataFrame(cb)
    cb.to_csv(os.path.join(OUT, "confirmation_bias_synth_val_v2.csv"), index=False)
    md.append("\n### synth_val_v2 (300, exact labels, not used for selection)\n")
    md.append(md_table(sv, ["variant", "MAPE", "acc10", "MAPE_single_scale"],
                       {"MAPE": "{:.2f}", "acc10": "{:.1f}", "MAPE_single_scale": "{:.2f}"}))
    md.append("\n### Confirmation bias, synthetic (student vs teacher where the teacher is unreliable)\n")
    md.append(md_table(cb[cb.subset != "all"], ["variant", "subset", "n_images", "MAPE_student", "MAPE_ours", "dMAPE",
                                                "dMAPE_lo", "dMAPE_hi", "p_wilcoxon", "student_gross_share"],
                       {c: "{:.2f}" for c in ("MAPE_student", "MAPE_ours", "dMAPE", "dMAPE_lo", "dMAPE_hi",
                                              "student_gross_share")} | {"p_wilcoxon": "{:.3g}"}))

    # ------------------------------------------------------------------ real150: disagreement with ours by ours' confidence
    rb, wide = [], None
    for v in spec:
        r = pd.read_csv(os.path.join(RUNS, "final", v["name"], "real150_predictions.csv"))
        key = "cnn-aug-" + v["name"].replace("_", "-")
        w = pd.DataFrame({"image_id": r.image, f"{key}__v": r.pred_h, f"{key}__h": r.pred_v,
                          f"{key}__weave": r.pred_weave})
        wide = w if wide is None else wide.merge(w, on="image_id")
        r = r[r.is_rep].copy()                 # one file per scene (76)
        r["confidence"] = r.scene_conf
        r = r.merge(pd.read_csv(os.path.join(RUNS, "pseudo_labels.csv"))[["id", "weave_ours"]].rename(
            columns={"id": "image"}), on="image")
        r["ape_pseudo"] = 50 * (np.abs(r.pred_h - r.pseudo_h) / r.pseudo_h + np.abs(r.pred_v - r.pseudo_v) / r.pseudo_v)
        r["dis"] = 50 * (np.abs(np.log(r.pred_h / r.ours_h)) + np.abs(np.log(r.pred_v / r.ours_v)))
        r["gross"] = (np.maximum(np.abs(np.log(r.pred_h / r.ours_h)), np.abs(np.log(r.pred_v / r.ours_v))) > np.log(1.25))
        rho, p = stats.spearmanr(r.confidence, r.dis)
        hv = r[r.split == "val"].dis
        lc = r[r.split == "lowconf"].dis
        mw = stats.mannwhitneyu(lc, hv, alternative="greater").pvalue
        row = {"variant": v["name"], "n_scenes": len(r), "spearman_conf_vs_disagreement": rho, "p_spearman": p,
               "p_MWU_lowconf_gt_heldout": mw}
        for s_ in ("train", "val", "lowconf"):
            g = r[r.split == s_]
            row[f"median_dis_{s_}"] = float(g.dis.median())
            row[f"gross_share_{s_}"] = float(g.gross.mean())
        # POST HOC (added after the first selection metrics showed a ~2x CNN / ours gap on scenes whose weave
        # ours labels 'unknown'): stratify by ours' weave
        unk = r.weave_ours == "unknown"
        for tag, mk in (("weave_unknown", unk), ("weave_known", ~unk)):
            row[f"n_{tag}"] = int(mk.sum())
            row[f"median_logratio_{tag}"] = float(np.median(np.log(np.concatenate([r.pred_h[mk] / r.ours_h[mk],
                                                                                   r.pred_v[mk] / r.ours_v[mk]]))))
            vv = r[mk & (r.split == "val")]
            row[f"pseudoval_MAPE_{tag}"] = float(vv.ape_pseudo.mean()) if len(vv) else np.nan
            row[f"pseudoval_n_{tag}"] = int(len(vv))
        rb.append(row)
    rb = pd.DataFrame(rb)
    rb.to_csv(os.path.join(OUT, "confirmation_bias_real150.csv"), index=False)
    wide.to_csv(os.path.join(OUT, "real150_aug_wide.csv"), index=False)
    ps_ = pd.read_csv(os.path.join(RUNS, "pseudo_labels.csv"))
    # same, with the pseudo-label-trained variants (V3-V5) blanked on EVERY file of the 28 pseudo-train scenes,
    # so that score150.py (which skips missing predictions) never scores a model on a scene it was fitted to
    trs = set(ps_.scene[ps_.use_train])
    in_tr = wide.image_id.map(dict(zip(ps_.id, ps_.scene))).isin(trs)
    wh = wide.copy()
    for c in wh.columns:
        if c.startswith(("cnn-aug-V3", "cnn-aug-V4", "cnn-aug-V5")):
            wh.loc[in_tr, c] = np.nan
    wh.to_csv(os.path.join(OUT, "real150_aug_wide_heldout.csv"), index=False)
    ps_[["id", "scene", "is_rep", "kind", "split", "use_train", "use_val"]].rename(columns={"id": "image_id"}).to_csv(
        os.path.join(OUT, "real150_aug_split.csv"), index=False)
    md.append("\n### Real 150, scene representatives (76): disagreement with ours (|log ratio|, %, mean of the two "
              "axes) by scene split (train = pseudo-train scenes, val = held-out accepted scenes, lowconf = rejected "
              "by the rule)\n")
    md.append(md_table(rb, ["variant", "median_dis_train", "median_dis_val", "median_dis_lowconf", "gross_share_val",
                            "gross_share_lowconf", "spearman_conf_vs_disagreement", "p_MWU_lowconf_gt_heldout"],
                       {c: "{:.2f}" for c in ("median_dis_train", "median_dis_val", "median_dis_lowconf",
                                              "gross_share_val", "gross_share_lowconf",
                                              "spearman_conf_vs_disagreement")} | {"p_MWU_lowconf_gt_heldout": "{:.2g}"}))
    md.append("\n### POST HOC: real 150 representatives by ours' weave label ('unknown' = ours' weave classifier "
              "abstains); median log(pred / ours) over both axes, and the pseudo-val MAPE split the same way\n")
    md.append(md_table(rb, ["variant", "n_weave_unknown", "median_logratio_weave_unknown", "median_logratio_weave_known",
                            "pseudoval_n_weave_unknown", "pseudoval_MAPE_weave_unknown", "pseudoval_MAPE_weave_known"],
                       {c: "{:.2f}" for c in ("median_logratio_weave_unknown", "median_logratio_weave_known",
                                              "pseudoval_MAPE_weave_unknown", "pseudoval_MAPE_weave_known")}))

    # ------------------------------------------------------------------ legacy-13: does the student copy ours' failures?
    pc = pd.concat(counts)
    pc = pc[pc["assign"] == "axis"]
    o = pc[pc.key == "est:ours"].set_index(["image", "axis"])
    lconf = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_estimators.csv"))
    lconf = lconf[(lconf.source == "vlmcrop") & (lconf.estimator == "ours")].set_index("image")
    lb = []
    for m in ms:
        s_ = pc[pc.key == m["key"]].set_index(["image", "axis"])
        j = s_.join(o, rsuffix="_ours")
        bad = j.ape_ours > 25
        r_, p_ = stats.spearmanr(j.ape, j.ape_ours)
        close_to_ours = (np.abs(np.log(j.pred / j.pred_ours)) < np.abs(np.log(j.pred / j.true)))
        lb.append({"variant": m["key"].split(":")[1], "n_ours_gross": int(bad.sum()),
                   "student_APE_on_ours_gross": float(j[bad].ape.mean()), "ours_APE_on_ours_gross": float(j[bad].ape_ours.mean()),
                   "student_also_gross": int((j[bad].ape > 25).sum()),
                   "student_APE_elsewhere": float(j[~bad].ape.mean()), "ours_APE_elsewhere": float(j[~bad].ape_ours.mean()),
                   "spearman_student_vs_ours_error": r_, "p": p_,
                   "share_closer_to_ours_than_to_GT": float(close_to_ours.mean())})
    lb = pd.DataFrame(lb)
    lb.to_csv(os.path.join(OUT, "confirmation_bias_legacy13.csv"), index=False)
    md.append("\n### Legacy-13: student errors on the counts where ours fails (ours APE > 25%)\n")
    md.append(md_table(lb, ["variant", "n_ours_gross", "student_APE_on_ours_gross", "ours_APE_on_ours_gross",
                            "student_also_gross", "student_APE_elsewhere", "ours_APE_elsewhere",
                            "spearman_student_vs_ours_error", "share_closer_to_ours_than_to_GT"],
                       {c: "{:.2f}" for c in ("student_APE_on_ours_gross", "ours_APE_on_ours_gross",
                                              "student_APE_elsewhere", "ours_APE_elsewhere",
                                              "spearman_student_vs_ours_error", "share_closer_to_ours_than_to_GT")}))
    lc_ = lconf[["conf"]].rename(columns={"conf": "ours_conf_legacy"})
    lc_.to_csv(os.path.join(OUT, "ours_conf_legacy13.csv"))

    # ------------------------------------------------------------------ ablation (selection metrics only)
    def best_of(name):
        bj = os.path.join(RUNS, name, "best.json")
        return json.load(open(bj)) if os.path.exists(bj) else None

    v4 = sel["V4_r18"]
    rho_tag = v4["run"].split("_")[-1]
    ab = []
    drop_runs = [n for n in sorted(os.listdir(RUNS)) if n.startswith("ABLFT_") and not n.startswith("ABLFT_seed")
                 and os.path.isdir(os.path.join(RUNS, n))]
    ref0 = ("ABLFT_seed0_r18", "none (seed 0)") if os.path.isdir(os.path.join(RUNS, "ABLFT_seed0_r18")) else \
        (v4["run"], "none (V4-R18, seed 0)")
    for name, fam in ([ref0] + [(f"ABLFT_seed{sd}_r18", f"none (seed {sd})") for sd in (1, 2)]
                      + [(n, n[len("ABLFT_"):-len("_r18")]) for n in drop_runs]):
        b = best_of(name)
        if b is None:
            continue
        ab.append({"dropped_family": fam, "run": name, "best_step": b["best_step"], "synth_MAPE": b["synth_MAPE"],
                   "pseudo_MAPE": b["pseudo_MAPE"], "score": b["score"]})
    ab = pd.DataFrame(ab)
    if len(ab):
        base = ab[ab.dropped_family.str.startswith("none")]
        lo, hi, mu = base.score.min(), base.score.max(), base.score.mean()
        ab["d_score_vs_seed_mean"] = ab.score - mu
        ab["outside_seed_range"] = (ab.score < lo) | (ab.score > hi)
        ab.to_csv(os.path.join(OUT, "ablation_stage2.csv"), index=False)
        md.append(f"\n### Ablation, stage 2: one family dropped in the V4-R18 fine-tuning (init {sel['V2_r18']['run']}, "
                  f"policy {sel['V2_r18'].get('policy')}, rho = {rho_tag}); selection metrics only; seed range "
                  f"[{lo:.2f}, {hi:.2f}]\n")
        md.append(md_table(ab, ["dropped_family", "best_step", "synth_MAPE", "pseudo_MAPE", "score",
                                "d_score_vs_seed_mean", "outside_seed_range"],
                           {c: "{:.2f}" for c in ("synth_MAPE", "pseudo_MAPE", "score", "d_score_vs_seed_mean")}))
    a1 = []
    for name, fam in ([("V2E_r18", "none (seed 0)"), ("SEED1_pre_r18", "none (seed 1)"), ("SEED2_pre_r18", "none (seed 2)")]
                      + [(f"ABL_{f}_pre_r18", f) for f in ("geometry", "photometric", "compression_blur", "illumination",
                                                             "zoom", "tiling")]):
        b = best_of(name)
        if b is not None:
            a1.append({"dropped_family": fam, "run": name, "synth_MAPE": b["synth_MAPE"],
                       "pseudo_MAPE": b["pseudo_MAPE"], "score": b["score"]})
    a1 = pd.DataFrame(a1)
    if len(a1):
        base = a1[a1.dropped_family.str.startswith("none")]
        lo, hi, mu = base.score.min(), base.score.max(), base.score.mean()
        a1["d_score_vs_seed_mean"] = a1.score - mu
        a1["outside_seed_range"] = (a1.score < lo) | (a1.score > hi)
        a1.to_csv(os.path.join(OUT, "ablation_stage1.csv"), index=False)
        md.append(f"\n### Ablation, stage 1: one family dropped in synthetic-only pre-training ('whitelist_ext', "
                  f"4750 steps, final step; adopted from the superseded round); seed range [{lo:.2f}, {hi:.2f}]\n")
        md.append(md_table(a1, ["dropped_family", "synth_MAPE", "pseudo_MAPE", "score", "d_score_vs_seed_mean",
                                "outside_seed_range"],
                           {c: "{:.2f}" for c in ("synth_MAPE", "pseudo_MAPE", "score", "d_score_vs_seed_mean")}))

    # ------------------------------------------------------------------ GPU hours (both rounds)
    gh = {}
    for tag, led in (("superseded_round", os.path.join(RUNS, "_superseded_autogroup_split", "runs", "_queue", "ledger.jsonl")),
                     ("scene_round", os.path.join(RUNS, "_queue", "ledger.jsonl"))):
        if os.path.exists(led):
            gh[tag + "_train_h"] = sum(json.loads(l)["seconds"] for l in open(led) if l.strip()) / 3600
    for tag, d_ in (("superseded_round", os.path.join(RUNS, "_superseded_autogroup_split", "runs", "final")),
                    ("scene_round", os.path.join(RUNS, "final"))):
        if os.path.isdir(d_):
            gh[tag + "_final_h"] = sum(json.load(open(os.path.join(d_, n, "meta.json"))).get("seconds", 0)
                                       for n in os.listdir(d_) if os.path.exists(os.path.join(d_, n, "meta.json"))) / 3600
    gh["total_h"] = sum(gh.values())
    json.dump(gh, open(os.path.join(OUT, "gpu_hours.json"), "w"), indent=1)
    md.append(f"\nGPU hours (RTX 4090; queue ledgers + final predictions; benches excluded): "
              f"{json.dumps({k: round(v, 2) for k, v in gh.items()})}\n")

    # ------------------------------------------------------------------ parity of the GPU pipeline (real150 only)
    par = {}
    ref = os.path.join(RUNS, "final", "REF_mps_r18", "real150_predictions.csv")
    mac = os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "eval_final", "real150_predictions.csv")
    if os.path.exists(ref) and os.path.exists(mac):
        a_, b_ = pd.read_csv(ref).set_index("image"), pd.read_csv(mac).set_index("image")
        lr = np.abs(np.log(np.concatenate([a_.pred_h / b_.pred_h, a_.pred_v / b_.pred_v])))
        par = {"n": int(len(a_)), "max_abs_log_ratio": float(lr.max()), "median_abs_log_ratio": float(np.median(lr))}
    json.dump(par, open(os.path.join(OUT, "parity.json"), "w"), indent=1)
    md.append(f"\nGPU-vs-MPS parity of the existing checkpoint on real150: {json.dumps(par)}\n")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

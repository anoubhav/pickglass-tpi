"""Score the scaffolded-visual-prompting runs on legacy-13, exactly like the VLM baselines.

* Point metrics: evaluate_vlm.metrics (axis-matched: vertical_yarns <-> warp, horizontal_yarns <->
  weft; best assignment: sorted pair vs sorted pair), MAE / MAPE / within-10 % / bias over the parsed
  images (failures not imputed).
* CIs and paired tests: paper/tools/legacy_unified.py (score / paired): percentile bootstrap over
  IMAGES (10 000 resamples, both counts of an image together), paired bootstrap on dMAPE, Wilcoxon on
  per-count absolute % errors (n = 26) and on per-image mean % errors (n = 13), exact McNemar on <=10 %.
* Baselines: each model's OWN P1 (and P2 / P3 / P3s / P4q where they exist) from
  vlm/results/legacy_predictions.csv (API models) and gpu_results/results/legacy_predictions.csv
  (Molmo2 bf16); same pixels (S1t input is byte-identical to the P1 input for the API models and
  pixel-identical to the Molmo2 P1 input).
* Pre-declared on the SYNTHETIC pilot before any legacy run (scaffold/FROZEN.json): primary scaffold
  = S3 for every model (confirmatory test: S3 vs own P1, one per model, Holm over the 4 models);
  every other contrast is exploratory (Holm over all exploratory scaffold-vs-P1 contrasts).

    <venv2>/bin/python evaluate_scaffolds.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "paper", "tools"))
import evaluate_vlm as EV                 # noqa: E402
import legacy_unified as LU               # noqa: E402
import scaffolds as SC                    # noqa: E402
from gt_legacy import legacy_gt           # noqa: E402
from parse_output import parse_counts     # noqa: E402

RAW = os.path.join(SC.SC_DIR, "raw")
OUT = os.path.join(SC.SC_DIR, "results")
P_VLM = os.path.join(HERE, "results", "legacy_predictions.csv")
P_MOLMO = next(p for p in (os.path.join(ROOT, "gpu_results", "results", "legacy_predictions.csv"),
                           os.path.join(ROOT, "paper", "tables", "src", "molmo2_bf16_legacy_predictions.csv"))
               if os.path.exists(p))
MODELS = ["gpt-6.1-sol", "gemini-3.8-flash", "qwen3.8-27b", "molmo2_8b_bf16"]
LABEL = {"gpt-6.1-sol": "GPT-6.1-sol", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "qwen3.8-27b": "Qwen3.8-27B (open)", "molmo2_8b_bf16": "Molmo2-8B bf16 (open)"}
OPEN = {"qwen3.8-27b", "molmo2_8b_bf16"}
CONDS = ["P1", "S1t", "S1", "S2", "S3", "S4"]
PRIMARY = "S3"
LEGACY_NAMES = EV.LEGACY_NAMES


# ----------------------------------------------------------------------------- loading
def load_scaffold_records():
    """-> list of dict(model, set, tag, repeat, scaffold, image, v, h, ok, cell_v, cell_h, ...)."""
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            model = r["model"].split(":", 1)[1] if r["model"].startswith("or:") else r["model"]
            p = parse_counts(r.get("raw") or "")                 # always re-parse from the raw text
            cv_, ch_ = SC.cell_estimate(r["scaffold"], r.get("raw") or "")
            out.append(dict(model=model, set=r["set"], tag=r.get("tag", "gpu"), repeat=int(r.get("repeat", 0)),
                            scaffold=r["scaffold"], image=str(r["image_id"]), v=p["vertical"], h=p["horizontal"],
                            weave=p["weave"], ok=bool(p["ok"]) and not r.get("error"), cell_v=cv_, cell_h=ch_,
                            error=r.get("error"), finish=r.get("finish_reason"), raw=r.get("raw") or "",
                            usd=float(r.get("cost_usd") or 0.0), sec=r.get("seconds"),
                            gen_tokens=r.get("generation_tokens"), reasoning_tokens=r.get("reasoning_tokens", 0)))
    # keep the last successful record per key
    best = {}
    for r in out:
        k = (r["model"], r["set"], r["tag"], r["repeat"], r["scaffold"], r["image"])
        if k not in best or (best[k]["error"] and not r["error"]):
            best[k] = r
    return list(best.values())


def baseline_rows():
    """Existing runs: {(model, method, repeat): {image: dict(v, h, ok)}} on the legacy set."""
    out = defaultdict(dict)
    v = pd.read_csv(P_VLM)
    v = v[(v.set == "legacy") & v.model.isin(MODELS)]
    m = pd.read_csv(P_MOLMO)
    # Molmo2 'repeat' 1/2 are the resize-side stability runs (588 / 700 px), not repeats -> excluded
    m = m[(m.set == "legacy") & (m["repeat"] == 0)]
    for d in (v, m):
        for _, r in d.iterrows():
            ok = str(r["ok"]) == "True"
            out[(r["model"], r["method"], int(r["repeat"]))][str(r["image"])] = dict(
                v=float(r["vertical"]) if ok else None, h=float(r["horizontal"]) if ok else None, ok=ok,
                weave=r["weave"] if isinstance(r["weave"], str) else None)
    return out


# ----------------------------------------------------------------------------- scoring helpers
def as_lu(rows, key, label):
    """evaluate_vlm-style rows -> legacy_unified method dict (h = vertical <-> warp, v = horizontal)."""
    ok = {i: e for i, e in rows.items() if e.get("ok")}
    return LU.P("vlm", label, key.split("|")[1] if "|" in key else "--",
                {i: e["v"] for i, e in ok.items()}, {i: e["h"] for i, e in ok.items()}, key=key)


def export150(recs):
    """Wide CSV in the VLM format read by paper/tools/score150.py --extra-wide
    (<key>__v = VERTICAL yarns, <key>__h = horizontal), keys '<model>@S3' and 's4-peaks', plus
    label-free diagnostics (parse rate, template-ness, rank agreement with ours / own P1)."""
    from scipy.stats import spearmanr
    man = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    side = {str(im["id"]): im["rect_size"][0] for im in man["images"]}
    wide = {str(i): dict(image_id=i, native_crop_px=side.get(str(i))) for i in range(150)}
    for r in recs:
        if r["set"] == "rect150" and r["scaffold"] == "S3" and r["repeat"] == 0:
            k = f"{r['model']}@S3"
            wide[r["image"]].update({f"{k}__v": r["v"] if r["ok"] else None, f"{k}__h": r["h"] if r["ok"] else None,
                                     f"{k}__weave": r["weave"]})
    for i in range(150):
        ov, oh = SC.profile_oracle_counts(SC.load_crop("rect150", str(i)))
        wide[str(i)].update({"s4-peaks__v": ov, "s4-peaks__h": oh})
    W = pd.DataFrame(list(wide.values()))
    W.to_csv(os.path.join(OUT, "rect150_S3_predictions_wide.csv"), index=False)
    ours = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    p1 = pd.read_csv(os.path.join(HERE, "results", "rect150_P1_predictions_wide.csv")).set_index("image_id")
    rows = []
    for k in [c[:-3] for c in W.columns if c.endswith("__v")]:
        d = W.set_index("image_id")[[f"{k}__v", f"{k}__h"]].dropna()
        m = d.mean(1)
        o = ((ours.h_count + ours.v_count) / 2).reindex(d.index)
        base = k.split("@")[0]
        row = dict(key=k, n_parsed=len(d), frac_v_eq_h=float((d[f"{k}__v"] == d[f"{k}__h"]).mean()),
                   n_distinct_pairs=int(len({tuple(x) for x in d.values})), median=float(m.median()),
                   spearman_vs_ours=float(spearmanr(m, o).correlation))
        if f"{base}__v" in p1.columns:
            q = p1[[f"{base}__v", f"{base}__h"]].reindex(d.index).dropna()
            row["spearman_vs_own_P1"] = float(spearmanr(m.reindex(q.index), q.mean(1)).correlation)
            row["spearman_P1_vs_ours"] = float(spearmanr(q.mean(1), o.reindex(q.index)).correlation)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rect150_S3_labelfree_summary.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def holm(ps):
    ps = np.asarray(ps, float)
    idx = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = legacy_gt()
    gt_df = LU.load_gt()
    assert set(gt) == set(gt_df.index) and all(gt[k]["warp"] == gt_df.loc[k, "warp"] and
                                                gt[k]["weft"] == gt_df.loc[k, "weft"] for k in gt)
    recs = load_scaffold_records()
    base = baseline_rows()

    # conditions available per model: {(model, cond, repeat): rows}
    cond = {}
    for (m, meth, rep), rows in base.items():
        if meth in ("P1", "P2", "P3", "P3s", "P4", "P4q"):
            cond[(m, meth, rep)] = rows
    for r in recs:
        if r["set"] != "legacy":
            continue
        rep = r["repeat"]
        if r["scaffold"] == "P1":
            # only the extra P1 repeats of the scaffold runs (Qwen3.8-27B r1/r2, tag p1rep); never overwrite an existing run
            if r["tag"] != "p1rep" or (r["model"], "P1", rep) in base:
                continue
        cond.setdefault((r["model"], r["scaffold"], rep), {})[r["image"]] = dict(
            v=r["v"], h=r["h"], ok=r["ok"], weave=r["weave"], usd=r["usd"], sec=r["sec"], finish=r["finish"],
            cell_v=r["cell_v"], cell_h=r["cell_h"], raw=r["raw"], gen_tokens=r["gen_tokens"])
    # mean of the available repeats (P1: 3 repeats for GPT / Flash; scaffolds: r0 + r1 when run)
    for m in MODELS:
        for c in CONDS + ["P3"]:
            reps = sorted(rep for (mm, cc, rep) in cond if mm == m and cc == c and isinstance(rep, int))
            if len(reps) >= 2:
                rows = {}
                for i in LEGACY_NAMES:
                    es = [cond[(m, c, rep)].get(i) for rep in reps]
                    if all(e is not None and e["ok"] for e in es):
                        rows[i] = dict(v=float(np.mean([e["v"] for e in es])), h=float(np.mean([e["h"] for e in es])),
                                       ok=True)
                cond[(m, c, "mean")] = rows

    # ------------------------------------------------------------ metrics table
    T = []
    for (m, c, rep), rows in sorted(cond.items(), key=lambda x: (MODELS.index(x[0][0]) if x[0][0] in MODELS else 9,
                                                                   x[0][1], str(x[0][2]))):
        if m not in MODELS:
            continue
        key = f"{m}|{c}|r{rep}"
        lu = as_lu(rows, key, LABEL[m])
        gtrows = [rows[i] for i in gt if i in rows]
        for assign, mode in (("axis", "axis"), ("sorted", "best")):
            pm = EV.metrics(rows, gt, mode)
            sc = LU.score(lu, gt_df, assign)
            r = dict(model=m, label=LABEL[m], cond=c, repeat=rep, assign=assign, n_gt=len(gt),
                     n_ok=pm.get("n_ok", 0), n_fail=pm.get("n_fail", 0), n_missing=pm.get("n_missing", 0),
                     MAE=pm.get("mae"), MAPE=pm.get("mape"), within10=pm.get("within10"), n_values=pm.get("n_values"),
                     bias=pm.get("bias"), medAE=pm.get("medae"), MAE_warp=pm.get("mae_warp"), MAE_weft=pm.get("mae_weft"))
            if sc:
                s, _ = sc
                r.update(MAPE_lo=s["MAPE_lo"], MAPE_hi=s["MAPE_hi"], MAE_lo=s["MAE_lo"], MAE_hi=s["MAE_hi"],
                         n_w5=s["n_w5"], MAPE_plain=s["MAPE_plain"], MAPE_noformula=s["MAPE_noformula"])
                assert abs(s["MAPE"] - pm["mape"]) < 1e-6, (key, s["MAPE"], pm["mape"])
            # harness re-computation from the per-cell lists (secondary)
            if c.startswith("S") and isinstance(rep, int):
                crow = {i: dict(v=e.get("cell_v"), h=e.get("cell_h"),
                                ok=e.get("cell_v") is not None and e.get("cell_h") is not None) for i, e in rows.items()}
                cm = EV.metrics(crow, gt, mode)
                r.update(MAPE_cells=cm.get("mape"), n_ok_cells=cm.get("n_ok", 0))
                # template diagnostics over all 20 legacy images
                al = [rows[i] for i in LEGACY_NAMES if i in rows and rows[i]["ok"]]
                r.update(frac_v_eq_h=float(np.mean([e["v"] == e["h"] for e in al])) if al else None,
                         n_distinct_pairs_20=len({(e["v"], e["h"]) for e in al}),
                         n_truncated=sum(1 for i in LEGACY_NAMES if i in rows and rows[i].get("finish") == "length"),
                         usd_per_img=float(np.mean([e.get("usd") or 0 for e in rows.values()])),
                         s_per_img=float(np.nanmean([e.get("sec") or np.nan for e in rows.values()])),
                         gen_tokens=float(np.mean([e.get("gen_tokens") or 0 for e in rows.values()])))
            T.append(r)
    T = pd.DataFrame(T)
    T.round(4).to_csv(os.path.join(OUT, "legacy13_scaffold_metrics.csv"), index=False)

    # ------------------------------------------------------------ paired tests
    PR = []

    def pair(ka, kb, label, kind):
        a, b = cond.get(ka), cond.get(kb)
        if not a or not b:
            return
        ma = as_lu(a, f"{ka[0]}|{ka[1]}|r{ka[2]}", "")
        mb = as_lu(b, f"{kb[0]}|{kb[1]}|r{kb[2]}", "")
        for assign in ("axis", "sorted"):
            d = LU.paired(ma, mb, gt_df, assign, label)
            d.update(kind=kind, model_A=ka[0], cond_A=ka[1], rep_A=ka[2], model_B=kb[0], cond_B=kb[1], rep_B=kb[2])
            PR.append(d)

    for m in MODELS:
        for c in ["S1t", "S1", "S2", "S3", "S4"]:
            kind = "confirmatory" if c == PRIMARY else "exploratory"
            pair((m, c, 0), (m, "P1", 0), f"{LABEL[m]}: {c} vs P1", kind)
            pair((m, c, "mean"), (m, "P1", "mean"), f"{LABEL[m]}: {c} vs P1 (mean of repeats)", "robustness")
            pair((m, c, 0), (m, "P1", "mean"), f"{LABEL[m]}: {c} r0 vs P1 mean of repeats", "robustness")
        pair((m, "S1", 0), (m, "S1t", 0), f"{LABEL[m]}: S1 (ruler drawn) vs S1t (same text, no ruler)",
             "mechanism")
        pair((m, "S3", 0), (m, "P3", 0), f"{LABEL[m]}: S3 (4 tenth-inch insets) vs P3 (central quarter inch x4)",
             "mechanism")
        if m != "gpt-6.1-sol":
            for c in ["P1", "S1t", "S1", "S2", "S3", "S4"]:
                pair((m, c, 0), ("gpt-6.1-sol", "P1", 0), f"gap: {LABEL[m]} {c} vs GPT-6.1-sol P1", "gap")
            pair((m, PRIMARY, 0), ("gpt-6.1-sol", PRIMARY, 0), f"gap: {LABEL[m]} S3 vs GPT-6.1-sol S3", "gap")
    PR = pd.DataFrame(PR)
    if len(PR):
        PR["p_holm"] = np.nan
        for assign in ("axis", "sorted"):
            for kind in ("confirmatory", "exploratory", "mechanism"):
                sel = (PR["assign"] == assign) & (PR.kind == kind)
                if sel.any():
                    PR.loc[sel, "p_holm"] = holm(PR.loc[sel, "p_wilcoxon_abs_pct_err_26"].fillna(1.0).to_numpy())
    PR.round(4).to_csv(os.path.join(OUT, "legacy13_scaffold_paired.csv"), index=False)

    # ------------------------------------------------------------ stability (r0 vs r1 on all 20 images)
    ST = []
    for m in MODELS:
        for c in CONDS:
            a, b = cond.get((m, c, 0)), cond.get((m, c, 1))
            if not a or not b:
                continue
            both = [i for i in LEGACY_NAMES if i in a and i in b and a[i]["ok"] and b[i]["ok"]]
            if not both:
                continue
            d = [abs(a[i][x] - b[i][x]) for i in both for x in ("v", "h")]
            rel = [abs(a[i][x] - b[i][x]) / max(1.0, (a[i][x] + b[i][x]) / 2) * 100 for i in both for x in ("v", "h")]
            unl = [i for i in both if i not in gt]
            du = [abs(a[i][x] - b[i][x]) for i in unl for x in ("v", "h")]
            ST.append(dict(model=m, cond=c, n_images=len(both), n_unlabelled=len(unl),
                           mean_abs_change_tpi=float(np.mean(d)), mean_rel_change_pct=float(np.mean(rel)),
                           max_abs_change_tpi=float(np.max(d)),
                           mean_abs_change_tpi_unlabelled=float(np.mean(du)) if du else None,
                           frac_identical_pairs=float(np.mean([a[i]["v"] == b[i]["v"] and a[i]["h"] == b[i]["h"]
                                                               for i in both]))))
    ST = pd.DataFrame(ST)
    ST.round(4).to_csv(os.path.join(OUT, "legacy20_scaffold_stability.csv"), index=False)

    # ------------------------------------------------------------ S4: does the plot encode the count? can models read it?
    S4 = []
    for i in LEGACY_NAMES:
        crop = SC.load_crop("legacy", i)
        ov, oh = SC.profile_oracle_counts(crop)
        row = dict(image=i, oracle_v=ov, oracle_h=oh, gt_warp=gt[i]["warp"] if i in gt else None,
                   gt_weft=gt[i]["weft"] if i in gt else None)
        for m in MODELS:
            e = cond.get((m, "S4", 0), {}).get(i)
            row[f"{m}_v"] = e["v"] if e and e["ok"] else None
            row[f"{m}_h"] = e["h"] if e and e["ok"] else None
        S4.append(row)
    S4 = pd.DataFrame(S4)
    S4.to_csv(os.path.join(OUT, "legacy20_S4_profile_oracle.csv"), index=False)
    orows = {r.image: dict(v=float(r.oracle_v), h=float(r.oracle_h), ok=True) for r in S4.itertuples()}
    cond[("profile-oracle", "S4o", 0)] = orows
    s4sum = dict(oracle_axis=EV.metrics(orows, gt, "axis"), oracle_best=EV.metrics(orows, gt, "best"))
    olu = as_lu(orows, "oracle|S4o|r0", "S4 plot peak count (find_peaks)")
    s4sum["oracle_axis_ci"] = LU.score(olu, gt_df, "axis")[0]
    for m in MODELS:
        rr = S4[S4[f"{m}_v"].notna()]
        if len(rr):
            dv = np.abs(np.r_[rr[f"{m}_v"] - rr.oracle_v, rr[f"{m}_h"] - rr.oracle_h])
            rel = dv / np.r_[rr.oracle_v, rr.oracle_h] * 100
            s4sum[f"{m}_vs_oracle"] = dict(n_images=int(len(rr)), mean_abs_diff=float(dv.mean()),
                                           mean_rel_diff_pct=float(rel.mean()),
                                           frac_within10pct=float((rel <= 10).mean()))
    # paired: can the models read the plot (VLM S4 vs the plot's own peak count), and how does the
    # plot's peak count compare with the classical counters of the unified table (same crops)?
    by_key = {x["key"]: x for x in LU.classical()}
    for m in MODELS:
        if (m, "S4", 0) in cond:
            for assign in ("axis", "sorted"):
                d = LU.paired(as_lu(cond[(m, "S4", 0)], f"{m}|S4|r0", ""), olu, gt_df, assign,
                              f"{LABEL[m]} reading S4 vs S4 plot peak count")
                d.update(kind="S4-oracle", model_A=m, cond_A="S4", rep_A=0, model_B="profile-oracle", cond_B="S4o", rep_B=0)
                PR.loc[len(PR)] = d
    for k, lab in (("est:ours", "ours (frozen)"), ("est:peaks", "classical profile peak count"),
                   ("est:gabor", "log-Gabor")):
        for assign in ("axis", "sorted"):
            d = LU.paired(olu, by_key[k], gt_df, assign, f"S4 plot peak count vs {lab}")
            d.update(kind="S4-oracle", model_A="profile-oracle", cond_A="S4o", rep_A=0, model_B=k, cond_B="--", rep_B=0)
            PR.loc[len(PR)] = d
    for assign in ("axis", "sorted"):
        d = LU.paired(olu, as_lu(cond[("gpt-6.1-sol", "P1", 0)], "gpt|P1|r0", ""), gt_df, assign,
                      "S4 plot peak count vs GPT-6.1-sol P1")
        d.update(kind="S4-oracle", model_A="profile-oracle", cond_A="S4o", rep_A=0, model_B="gpt-6.1-sol", cond_B="P1", rep_B=0)
        PR.loc[len(PR)] = d
    # does a scaffolded model beat a know-nothing predictor (leave-one-out mean of the other 12 hand counts)?
    loo = LU.loo_mean(gt_df)
    for m in MODELS:
        for c in ("P1", "S3"):
            if (m, c, 0) in cond:
                for assign in ("axis", "sorted"):
                    d = LU.paired(as_lu(cond[(m, c, 0)], f"{m}|{c}|r0", ""), loo, gt_df, assign,
                                  f"{LABEL[m]} {c} vs leave-one-out mean")
                    d.update(kind="vs-LOO-mean", model_A=m, cond_A=c, rep_A=0, model_B="loo_mean", cond_B="--", rep_B=0)
                    PR.loc[len(PR)] = d
    PR.round(4).to_csv(os.path.join(OUT, "legacy13_scaffold_paired.csv"), index=False)
    json.dump(s4sum, open(os.path.join(OUT, "legacy20_S4_oracle_summary.json"), "w"), indent=1, default=float)

    # ------------------------------------------------------------ 150 photos: pre-declared S3 (+ S4 plot peaks), label-free
    export150(recs)

    # ------------------------------------------------------------ cost
    cost = pd.read_csv(os.path.join(SC.SC_DIR, "logs", "openrouter_costs_job2.csv"))
    csum = cost.groupby(["model", "set", "tag", "repeat"]).agg(n_calls=("cost_usd", "size"),
                                                                usd=("cost_usd", "sum")).reset_index()
    csum.to_csv(os.path.join(OUT, "job2_costs.csv"), index=False)

    # ------------------------------------------------------------ print
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    show = T[(T["assign"] == "axis") & T.cond.isin(CONDS + ["P3", "P3s", "P4q"])]
    print(show[["label", "cond", "repeat", "n_ok", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "within10", "bias",
                "MAPE_cells", "frac_v_eq_h", "n_distinct_pairs_20", "n_truncated", "usd_per_img"]]
          .round(2).to_string(index=False))
    print()
    if len(PR):
        print(PR[PR["assign"] == "axis"][["comparison", "kind", "n_counts", "MAPE_A", "MAPE_B", "dMAPE", "dMAPE_lo",
                                       "dMAPE_hi", "p_boot_MAPE", "p_wilcoxon_abs_pct_err_26", "p_holm",
                                       "n_w10_A", "n_w10_B", "p_mcnemar_w10"]].round(3).to_string(index=False))
    print()
    print(ST.round(2).to_string(index=False))
    print()
    print(json.dumps(s4sum, indent=1, default=float))
    print()
    print(csum.to_string(index=False), "\nTOTAL job-2 logged spend: $%.4f" % cost.cost_usd.sum())
    return T, PR, ST


if __name__ == "__main__":
    main()

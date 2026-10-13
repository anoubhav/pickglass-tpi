"""Score the Molmo2-4B rows (base and synthetic-pointing LoRA) on legacy-13 with the paper's unified
scorer (paper/tools/legacy_unified.py: same GT parser, same canonical crops, same 'axis' assignment,
same image-level bootstrap CIs and paired tests), and compare with the existing rows.

Reads   ft_results/results/legacy_predictions.csv  (summarize_ft.py, synced from the server)
        ft_results/results/selection.json           (primary readout fixed on SYNTHETIC validation)
Writes  ft_results/tables/legacy13_ft.csv            one row per (model, readout, assignment)
        ft_results/tables/legacy13_ft_per_count.csv
        ft_results/tables/legacy13_ft_paired.csv     paired tests (primary FT readout vs others)

    <venv2>/bin/python ft_results/code/score_ft_legacy.py
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FTR = os.path.dirname(HERE)
ROOT = os.path.dirname(FTR)
spec = importlib.util.spec_from_file_location("LU", os.path.join(ROOT, "paper", "tools", "legacy_unified.py"))
LU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LU)

LABEL = {"molmo2_4b_base": "Molmo2-4B (base, zero-shot)",
         "molmo2_4b_synthft": "Molmo2-4B + synthetic-pointing LoRA (ours)"}
READOUTS = ["P1", "P4", "P4q", "P4s", "P4q_sp", "P4s_sp"]


def ft_methods(pred_csv):
    d = pd.read_csv(pred_csv, dtype={"image": str})
    out = []
    for model in [m for m in LABEL if m in set(d.model)]:
        for meth in READOUTS:
            g = d[(d.model == model) & (d.method == meth) & d.ok]
            if not len(g):
                continue
            g = g.set_index("image")
            out.append(LU.P("vlm-ft", LABEL[model], meth, g.vertical.astype(float), g.horizontal.astype(float),
                            key=f"ft:{model}|{meth}", input_px="644", usd=0.0,
                            sec=float(d[(d.model == model) & (d.method == meth)].seconds.mean()),
                            sec_hw="RTX 4090 (transformers bf16, batched greedy), mean", offline="yes (open weights)",
                            note=""))
    return out


def main():
    gt = LU.load_gt()
    sp = os.path.join(FTR, "results", "selection.json")
    prim = json.load(open(sp))["primary"] if os.path.exists(sp) else None
    ft = ft_methods(os.path.join(FTR, "results", "legacy_predictions.csv"))
    ref = [LU.loo_mean(gt)] + LU.classical() + LU.cnn() + LU.vlms()
    by_key = {m["key"]: m for m in ref + ft}
    keep_ref = ["ref:loo_mean", "est:ours", "est:gabor", "cnn:synth_r18", "vlm:gpt-6.1-sol|P1|r0",
                "vlm:molmo2_8b_bf16|P1|r0", "vlm:molmo2_8b_bf16|P3|r0", "vlm:molmo2_8b_bf16|P4|r0",
                "vlm:molmo2_8b_bf16|P4q|r0"]
    rows, counts = [], []
    for m in [by_key[k] for k in keep_ref if k in by_key] + ft:
        for assign in ("axis", "sorted"):
            s = LU.score(m, gt, assign)
            if s is None:
                continue
            r, dd = s
            rows.append(dict(group=m["group"], method=m["method"], prompt=m["prompt"], key=m["key"], assign=assign,
                             primary=(m["key"] == f"ft:molmo2_4b_synthft|{prim}"), **r,
                             s_per_img=m["s_per_img"], latency_hw=m["latency_hw"], note=m["note"]))
            counts.append(dd.assign(key=m["key"], method=m["method"], prompt=m["prompt"], assign=assign))
    os.makedirs(os.path.join(FTR, "tables"), exist_ok=True)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(os.path.join(FTR, "tables", "legacy13_ft.csv"), index=False)
    pd.concat(counts).round(3).to_csv(os.path.join(FTR, "tables", "legacy13_ft_per_count.csv"), index=False)

    show = T[T["assign"] == "axis"][["method", "prompt", "n_counts", "MAPE", "MAPE_lo", "MAPE_hi", "n_w10", "MAE", "bias",
                                   "primary"]]
    print(show.round(1).to_string(index=False))
    A = by_key.get(f"ft:molmo2_4b_synthft|{prim}")
    if A is None:
        print("no fine-tuned primary readout yet -> no paired tests")
        return
    comps = [(f"FT {prim} vs base 4B {prim}", by_key.get(f"ft:molmo2_4b_base|{prim}")),
             (f"FT {prim} vs base 4B P4", by_key.get("ft:molmo2_4b_base|P4")),
             (f"FT {prim} vs base 4B P4q", by_key.get("ft:molmo2_4b_base|P4q")),
             (f"FT {prim} vs base 4B P1", by_key.get("ft:molmo2_4b_base|P1")),
             (f"FT {prim} vs Molmo2-8B bf16 P4", by_key.get("vlm:molmo2_8b_bf16|P4|r0")),
             (f"FT {prim} vs Molmo2-8B bf16 P4q", by_key.get("vlm:molmo2_8b_bf16|P4q|r0")),
             (f"FT {prim} vs Molmo2-8B bf16 P1", by_key.get("vlm:molmo2_8b_bf16|P1|r0")),
             (f"FT {prim} vs ours (PG-Lattice)", by_key.get("est:ours")),
             (f"FT {prim} vs CNN synthetic", by_key.get("cnn:synth_r18")),
             (f"FT {prim} vs GPT-6.1-sol P1", by_key.get("vlm:gpt-6.1-sol|P1|r0")),
             (f"FT {prim} vs log-Gabor", by_key.get("est:gabor")),
             (f"FT {prim} vs leave-one-out mean", by_key.get("ref:loo_mean"))]
    comps = list({c[0]: c for c in comps}.values())          # drop duplicate labels (prim == P1 etc.)
    best_pt = "P4s"   # best POINTING readout on synthetic val (selection.json order); secondary comparisons
    B = by_key.get(f"ft:molmo2_4b_synthft|{best_pt}")
    if B is not None and best_pt != prim:
        for lab, k in (("ours (PG-Lattice)", "est:ours"), ("CNN synthetic", "cnn:synth_r18"),
                       ("GPT-6.1-sol P1", "vlm:gpt-6.1-sol|P1|r0"), ("Molmo2-8B bf16 P4q", "vlm:molmo2_8b_bf16|P4q|r0"),
                       ("log-Gabor", "est:gabor"), ("leave-one-out mean", "ref:loo_mean")):
            if k in by_key:
                comps.append((f"[secondary] FT {best_pt} vs {lab}", B, by_key[k]))
        comps.append((f"[secondary] FT {best_pt} vs FT {prim}", B, A))
    for meth in READOUTS:   # every other FT readout vs its own base readout (secondary)
        if meth != prim and f"ft:molmo2_4b_synthft|{meth}" in by_key and f"ft:molmo2_4b_base|{meth}" in by_key:
            comps.append((f"[secondary] FT {meth} vs base 4B {meth}", by_key[f"ft:molmo2_4b_synthft|{meth}"],
                          by_key[f"ft:molmo2_4b_base|{meth}"]))
    PR = []
    for c in comps:
        lab, a_, b_ = (c[0], A, c[1]) if len(c) == 2 else c
        if b_ is None:
            continue
        for assign in ("axis", "sorted"):
            PR.append(LU.paired(a_, b_, gt, assign, lab))
    PR = pd.DataFrame(PR)
    PR.round(4).to_csv(os.path.join(FTR, "tables", "legacy13_ft_paired.csv"), index=False)
    print(PR[PR["assign"] == "axis"][["comparison", "n_counts", "MAPE_A", "MAPE_B", "dMAPE", "dMAPE_lo", "dMAPE_hi",
                                   "p_boot_MAPE", "p_wilcoxon_abs_pct_err_26", "n_w10_A", "n_w10_B",
                                   "p_mcnemar_w10"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

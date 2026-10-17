#!/usr/bin/env python
"""make_tab1_ref.py -- main Tab. 1: every method scored against A2, the textile-trained reference annotator, alone.

    python paper/tools/reference/make_tab1_ref.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR] [optional checks, below]

Read-only on every input; writes main150_rows_ref.tex, main150_rows_refB.tex, tab1_ref_rows.csv and
tab1_ref_checks.json in --out.

Reference: A2 (textile-trained; its counts were the blind test set, opened only after every method was frozen).
A1 (the non-expert) appears as a prediction of A2 (first row, a human baseline). Scoring against A2 alone was decided
after both annotators' counts were seen (post hoc, declared in the paper); the reference fixed before labeling is the
mean of both annotators (supplement Tabs. S18 and S13; each annotator alone: Tab. S15).

Statistics = the paper's own frozen code, imported read-only (sys.dont_write_bytecode; nothing that writes is called):
  paper/tools/score150.py   load_predictions, load_extra_wide, loo_mean, metrics (scene bootstrap 2,000, seed 0 = the
                            Tab. 1 CIs), tex_row, tex_escape
  paper/tools/label_day.py  ci_format (integer CI bounds, as in Tab. 1)
Reference table: label_day.py and score150.py score the annotation tool's CSV export, which stores <ax>_tpi to 2
decimals; here the reference is the per-scene counts rounded to 2 decimals (the same values), one row per scene in
ascending scene order (label_day's order, which fixes the bootstrap), a count <= 0 or missing being missing
(score150.load_labels). A1's counts as a prediction are built the same way.
Row format: method & MAPE [95% CI] & % of counts <= 5% & <= 10% & scenes with both counts within 5% (of those scored).
  main150_rows_ref.tex   the rows as generated;
  main150_rows_refB.tex  the paper's Tab. 1 rows: final row names and, among the methods (the two baseline rows
                         excluded), the best printed value per column in bold, ties included (done by hand for the
                         paper, scripted here).

Extra (CSV only): mean |log2(pred/ref)| over the scored counts (pre-specified against A2 in item 5 of the plan for the
reference annotator, not in this repository, SHA-256 9ae1d9ca..., see CHECKSUMS.md) and the share of counts with
|log2| > 0.5; its scene-bootstrap CI (metrics() scheme, 2,000, seed 0) is post hoc.

Optional checks and context (inputs that are not part of this repository; each step is skipped when not given):
  --a2-run / --a1-run DIR     a paper/tools/label_day.py single-annotator run on that annotator's tool export (the run
                              folder holding work/): the reference equals its per-scene table, every prediction source
                              was read by its score150 call (work/label_day.log), and every A2 row equals its
                              work/stage/score/main150.csv (2 decimals)
  --per-annotator-table CSV   per-annotator Tab. 1 (method, MAPE against A2 / A1 / the mean): fills the context column
                              MAPE_vs_mean_reference_context (MAPE against the mean of both annotators)
  --fix-numbers JSON          Molmo2-4B + LoRA scored against A2 and the mean by an independent script: checked, and
                              its mean-reference MAPE fills the context column for that row
  --per-annotator-md MD       the same table as Markdown; with --a1-run, --a2-run, --mean-run (label_day.py run
                              against the mean of both annotators) and --tab1-spec (label_day's Tab. 1 row list) it is
                              reproduced line by line (10,000 resamples, one RNG seeded 20261012), and its A2 column
                              (MAPE, both-within-5%) is checked against the rows here
  --logratio-json JSON        logratio_endpoint.py output (default <out>/logratio_endpoint.json if present): the mean
                              |log2| and the share > 0.5 of five rows are checked against it (3 decimals)
"""
import sys

sys.dont_write_bytecode = True
import argparse
import glob
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
ap.add_argument("--a2-run", default=None, help="optional: label_day.py single-annotator run on A2's export")
ap.add_argument("--a1-run", default=None, help="optional: label_day.py single-annotator run on A1's export")
ap.add_argument("--per-annotator-table", default=None, help="optional: per-annotator Tab. 1 CSV (context column)")
ap.add_argument("--fix-numbers", default=None, help="optional: independent Molmo2-4B + LoRA scores (JSON)")
ap.add_argument("--per-annotator-md", default=None, help="optional: per-annotator Tab. 1 Markdown to reproduce")
ap.add_argument("--mean-run", default=None, help="optional: label_day.py run against the mean of both annotators")
ap.add_argument("--tab1-spec", default=None, help="optional: label_day's Tab. 1 row list (JSON)")
ap.add_argument("--logratio-json", default=None, help="optional: logratio_endpoint.py output to check against")
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
SELF = os.path.relpath(os.path.abspath(__file__), ROOT)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """The optional per-annotator inputs label the two annotators by the internal codes of the original runs; their
    labels are read as A1 / A2."""
    return s.replace("FL_NG", "A2").replace("FL_SA", "A1")


S = load("score150_ro", f"{ROOT}/paper/tools/score150.py")
LD = load("label_day_ro", f"{ROOT}/paper/tools/label_day.py")

OURS, GPT, LOO, CNN0 = "Ours (lattice + fusion)", "GPT-6.1-sol, P1", "Leave-one-out mean (reference)", "CNN, synthetic only (zero-shot)"
A1KEY, CNNA1, ENS = "A1 counts", "cnn-fabric-oof-FLSA", "ens-v1"
CHK = {"code": {p: hashlib.md5(open(f"{ROOT}/{p}", "rb").read()).hexdigest()
                for p in ("paper/tools/score150.py", "paper/tools/label_day.py")}}
assert CHK["code"]["paper/tools/score150.py"] == "5879ef44698de7b2a090e8bbee973625"   # frozen scorer, byte-identical


# ------------------------------------------------------------------ reference tables (as score150.py builds them)
def scene_gt(path):
    """Per-scene counts -> the scored reference: <ax>_tpi to 2 decimals (the tool export), <= 0 or missing = missing,
    ascending scene order."""
    L = pd.read_csv(path)
    L["scene_id"] = L.scene_id.astype(int)
    L = L.set_index("scene_id").sort_index()
    gs = pd.DataFrame(index=L.index)
    for ax in ("h", "v"):
        c = pd.to_numeric(L[f"{ax}_tpi"], errors="coerce").round(2)
        if f"{ax}_count" in L:   # score150.load_labels: <ax>_tpi when present, else <ax>_count
            c = c.fillna(pd.to_numeric(L[f"{ax}_count"], errors="coerce"))
        gs[ax] = c.where(c > 0)   # a cleared line (0 dots) is missing, not 0 TPI
    gs.index.name = None
    return gs


GT = {"A2": scene_gt(args.a2), "A1": scene_gt(args.a1)}
REF = GT["A2"]
RUN = {"A2": args.a2_run, "A1": args.a1_run}
for a in ("A2", "A1"):
    lab = pd.read_csv(args.a2 if a == "A2" else args.a1).set_index("scene_id").loc[GT[a].index]
    CHK[f"gt_{a}_vs_labels_per_scene_maxdiff"] = float(max((lab.h_tpi - GT[a].h).abs().max(), (lab.v_tpi - GT[a].v).abs().max()))
    CHK[f"gt_{a}_n_scenes"] = int(len(GT[a]))
    CHK[f"gt_{a}_missing_axes"] = int(GT[a][["h", "v"]].isna().to_numpy().sum())
    # the scored reference is the tool's CSV export (<ax>_tpi to 2 decimals); the per-scene counts may keep full precision
    assert CHK[f"gt_{a}_vs_labels_per_scene_maxdiff"] < 0.005
    if RUN[a]:   # same scenes, same order, same values as the run's per-scene file
        ps = pd.read_csv(f"{RUN[a]}/work/stage/main150_per_scene.csv")
        po = ps[ps.method == OURS]
        assert list(po.scene_id.astype(int)) == list(GT[a].index), a
        CHK[f"gt_{a}_vs_run_per_scene_maxdiff"] = float(max((po.h.to_numpy() - GT[a].h.to_numpy()).__abs__().max(),
                                                           (po.v.to_numpy() - GT[a].v.to_numpy()).__abs__().max()))
        assert CHK[f"gt_{a}_vs_run_per_scene_maxdiff"] < 1e-3
assert (GT["A1"].index.sort_values() == REF.index.sort_values()).all()

# ------------------------------------------------------------------ predictions (the sources label_day.py scores)
SPECS = [f"{ROOT}/gpu_results/countgd/outputs/results/rect150_predictions_wide.csv::countgd@CG-W-ex3-lat",
         f"{ROOT}/ft_results/results/rect150_predictions_wide.csv::molmo2_4b_synthft@P1",
         f"{ROOT}/paper/tables/src/extra150_wide.csv::pg-lattice@ruler,cascade-gate",
         f"{ROOT}/cnn/runs_aug/report/real150_aug_wide_heldout.csv::cnn-aug-V4-b0",
         f"{ROOT}/methods/results/combined_v1_150_wide.csv::combined-v1"]
if RUN["A2"]:
    log = open(f"{RUN['A2']}/work/label_day.log").read()
    for sp in SPECS:   # every source is one the run's score150 call read (label_day.log)
        rel = os.path.relpath(sp.split("::")[0], ROOT)
        assert rel in log, sp
    CHK["sources_in_A2_run_log"] = True
PRED = S.load_predictions()
PRED.update(S.load_extra_wide(SPECS))
PRED[LOO] = S.loo_mean(REF)                                  # leave-one-out mean of A2's own counts
PRED[A1KEY] = GT["A1"][["h", "v"]].copy()                    # A1's counts as a prediction of A2


def read_preds(path):   # numeric h_count / v_count of a frozen prediction file
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(int)
    d = d.set_index("id")
    raw = pd.DataFrame({"h": pd.to_numeric(d.h_count, errors="coerce"), "v": pd.to_numeric(d.v_count, errors="coerce")})
    assert (raw > 0).all().all(), path      # no non-positive prediction
    return raw


PRED[CNNA1] = read_preds(f"{ROOT}/methods/ensemble/cnn_fabric_oof_FLSA_150.csv")
PRED[ENS] = read_preds(f"{ROOT}/methods/ensemble/predictions_ens_v1_150.csv")
for k in list(PRED):
    PRED[k] = PRED[k].copy()
    PRED[k].index = PRED[k].index.astype(int)

# ------------------------------------------------------------------ rows: (block, display label, key, source, status)
FROZEN = "frozen before A2's counts were opened"
ROWS = [
    (1, "Non-expert annotator (A1)", A1KEY, "A1's per-scene counts (--a1), rounded to 2 decimals as the tool export",
     "pre-specified (plan for the reference annotator, item 3: method-vs-A2 error compared with A1-vs-A2 error)"),
    (1, "Leave-one-out mean", LOO, "score150.loo_mean of A2", "Tab. 1 row (frozen label_day TAB1_SPEC)"),
    (2, "Log-Gabor bank", "Log-Gabor bank", "methods/predictions_ours_150.csv h_gabor/v_gabor", "Tab. 1 row; " + FROZEN),
    (2, r"\ours (spectral)", OURS, "methods/predictions_ours_150.csv", "Tab. 1 row; " + FROZEN),
    (2, r"Combined (\ours + log-Gabor)", "combined-v1", "methods/results/combined_v1_150_wide.csv", "Tab. 1 row; " + FROZEN),
    (2, r"\ours, ruler crop", "pg-lattice@ruler", "paper/tables/src/extra150_wide.csv", "Tab. 1 row; " + FROZEN),
    (2, r"Cascade (\ours/GPT)", "cascade-gate", "paper/tables/src/extra150_wide.csv (frozen gate)", "Tab. 1 row; " + FROZEN),
    (3, "CNN, synthetic (zero-shot)", CNN0, "cnn/runs/synth_r18_v2mix_aniso/eval_final/real150_predictions.csv",
     "Tab. 1 row; " + FROZEN),
    (3, r"CNN, + pseudo-labels$^\ast$", "cnn-aug-V4-b0", "cnn/runs_aug/report/real150_aug_wide_heldout.csv (held-out scenes)",
     "Tab. 1 row; " + FROZEN),
    (3, r"CNN, trained on A1$^\ddagger$", CNNA1, "methods/ensemble/cnn_fabric_oof_FLSA_150.csv (fabric-grouped, out of fold)",
     "built after A1's counts were seen (trained on them); " + FROZEN + "; not a Tab. 1 row of the frozen spec"),
    (3, r"CNN--VLM ensemble$^\ddagger$", ENS, "methods/ensemble/predictions_ens_v1_150.csv (ens-v1, PREREG 9cc90dea)",
     "built after A1's counts were seen; " + FROZEN + " (its pre-registered primary vs GPT failed against A2)"),
    (4, "Molmo2-4B, synthetic LoRA", "molmo2_4b_synthft@P1", "ft_results/results/rect150_predictions_wide.csv (P1 readout)",
     "readout pre-declared on synthetic validation; not a Tab. 1 row of the frozen spec; " + FROZEN),
    (4, r"CountGD, cells $\to$ lattice$^\ddagger$", "countgd@CG-W-ex3-lat",
     "gpu_results/countgd/outputs/results/rect150_predictions_wide.csv", "post-hoc variant (fixed before any human count)"),
    (5, "GPT-6.1-sol", "GPT-6.1-sol, P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Claude Fable 5.1", "Claude Fable 5.1, P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Gemini 3.1 Pro", "Gemini 3.1 Pro, P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Gemini 3.8 Flash", "Gemini 3.8 Flash, P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Qwen3.8-Max-Prime", "Qwen3.8-Max-Prime, P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Qwen3.8-27B (open)", "Qwen3.8-27B (open), P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
    (5, "Gemma-4-26B (open)", "Gemma-4-26B-A4B (open), P1", "vlm/results/rect150_P1_predictions_wide.csv", "Tab. 1 row; " + FROZEN),
]
# the paper's final row names (main150_rows_refB.tex); every other label is printed as generated
FINAL_NAME = {"Non-expert annotator (A1)": "Non-expert annotator", r"\ours (spectral)": r"\ours (ours)",
              r"CNN, trained on A1$^\ddagger$": r"CNN, trained on non-expert$^\ddagger$",
              "Molmo2-4B, synthetic LoRA": "Molmo2-4B + LoRA (ours)"}


def log2_err(gt, pr, n_boot=2000, seed=0):
    """mean |log2(pred/ref)| over the scored counts (same scenes as metrics(): both axes predicted); share > 0.5;
    scene bootstrap CI exactly as metrics() resamples (post hoc)."""
    j = gt[["h", "v"]].join(pr[["h", "v"]], rsuffix="_p", how="left")
    jj = j[j[["h_p", "v_p"]].notna().all(axis=1)]
    lh, lv = np.abs(np.log2(jj.h_p / jj.h)).to_numpy(), np.abs(np.log2(jj.v_p / jj.v)).to_numpy()
    assert np.isfinite(lh).all() and np.isfinite(lv).all()
    allc = np.r_[lh, lv]
    per = 0.5 * (lh + lv)
    rng = np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return dict(mean_abs_log2=float(allc.mean()), share_abs_log2_gt_0_5=float(np.mean(allc > 0.5)), n_counts=int(len(allc)),
                mean_abs_log2_lo_posthoc=float(np.percentile(bs, 2.5)), mean_abs_log2_hi_posthoc=float(np.percentile(bs, 97.5)))


def p5(m):
    n = m["both5"] * m["n_scored"] / 100
    assert abs(n - round(n)) < 0.01, n
    return int(round(n))


main = pd.read_csv(f"{RUN['A2']}/work/stage/score/main150.csv").set_index("method") if RUN["A2"] else None
recs, tex, cells, chk_run, grp = [], [], [], {}, None
for blk, disp, key, src, status in ROWS:
    pr = PRED[key]
    m = S.metrics(REF, pr)                           # n_boot=2000, seed=0: the Tab. 1 bootstrap
    line = S.tex_row(key, m)
    esc = S.tex_escape(key)
    assert line.startswith(esc), (key, line[:60])
    row = LD.ci_format(disp + line[len(esc):], 0)   # integer CI bounds, as label_day.py writes Tab. 1
    c = [x.strip() for x in row.rstrip()[:-2].split(" & ")]
    assert len(c) == 6, c
    if grp is not None and blk != grp:
        tex.append("\\midrule")
    grp = blk
    tex.append(" & ".join([c[0]] + c[2:5] + [str(p5(m))]) + " \\\\")   # drop MAE (c[1]) and weave (c[5]); add P5
    cells.append((blk, disp, [c[0]] + c[2:5] + [str(p5(m))]))
    if main is not None and key in main.index:        # identical to the run's score150 output (2 decimals)
        r = main.loc[key]
        d = max(abs(round(m[k], 2) - r[k]) for k in ("MAPE", "MAPE_lo", "MAPE_hi", "w5", "w10", "both5"))
        chk_run[key] = float(d)
        assert d < 0.0051 and int(r.n_scored) == m["n_scored"], (key, d)
    lg = log2_err(REF, pr)
    recs.append(dict(block=blk, label=disp, key=key, source=src, status=status, reference="A2 alone",
                     n_gt=m["n_gt"], n_scored=m["n_scored"], MAPE=m["MAPE"], MAPE_lo=m["MAPE_lo"], MAPE_hi=m["MAPE_hi"],
                     w5=m["w5"], w10=m["w10"], both5_pct=m["both5"], P5_n=p5(m), both10_pct=m["both10"], MAE=m["MAE"],
                     MdAPE=m["MdAPE"], bias_med=m["bias_med"], spread_mad=m["spread_mad"], **lg, tex_row=tex[-1]))
if main is not None:
    CHK["rows_vs_A2_run_main150_maxdiff_2dp"] = chk_run
T = pd.DataFrame(recs)

# ------------------------------------------------------------------ final rows of the paper (names, bold)
def bold_rows():
    meth = [x for x in cells if x[0] != 1]           # the two baseline rows (A1's counts, leave-one-out mean) excluded
    num = lambda s: float(s.split(" [")[0])          # noqa: E731  the printed value
    best = {1: min(num(x[2][1]) for x in meth), 2: max(num(x[2][2]) for x in meth),
            3: max(num(x[2][3]) for x in meth), 4: max(num(x[2][4]) for x in meth)}
    out, g = [], None
    for blk, disp, cc in cells:
        if g is not None and blk != g:
            out.append("\\midrule")
        g = blk
        cc = list(cc)
        lab = cc[0]
        if disp in FINAL_NAME:
            lab = FINAL_NAME[disp] + lab[len(disp):]
        cc[0] = lab
        if blk != 1:
            for i in (1, 2, 3, 4):
                if num(cc[i]) == best[i]:
                    v = cc[i].split(" [")
                    cc[i] = "\\textbf{" + v[0] + "}" + ((" [" + v[1]) if len(v) > 1 else "")
        out.append(" & ".join(cc) + " \\\\")
    return out


texB = bold_rows()

# context column: MAPE against the mean reference (per-annotator table 'mean'; Molmo2 from --fix-numbers); optional
# names of the methods in the per-annotator table (written with the original run's labels)
PAT_NAME = {LOO: "Leave-one-out mean", "Log-Gabor bank": "Log-Gabor bank", OURS: r"\ours (spectral)",
            "combined-v1": r"Combined (\ours + log-Gabor)", "pg-lattice@ruler": r"\ours, ruler crop", "cascade-gate": r"Cascade (\ours/GPT)",
            CNN0: "CNN, synthetic (zero-shot)", "cnn-aug-V4-b0": r"CNN, + pseudo-labels$^\ast$",
            CNNA1: "CNN fabric-grouped (A1-trained)", ENS: "ens-v1 = CNN x GPT (frozen)",
            "countgd@CG-W-ex3-lat": r"CountGD, cells $\to$ lattice$^\ddagger$", "GPT-6.1-sol, P1": "GPT-6.1-sol",
            "Claude Fable 5.1, P1": "Claude Fable 5.1", "Gemini 3.1 Pro, P1": "Gemini 3.1 Pro", "Gemini 3.8 Flash, P1": "Gemini 3.8 Flash",
            "Qwen3.8-Max-Prime, P1": "Qwen3.8-Max-Prime", "Qwen3.8-27B (open), P1": "Qwen3.8-27B (open)",
            "Gemma-4-26B-A4B (open), P1": "Gemma-4-26B (open)"}
if args.per_annotator_table:
    pm = pd.read_csv(args.per_annotator_table).set_index("method")["mean"]
    pm.index = [norm(i) for i in pm.index]
    T["MAPE_vs_mean_reference_context"] = [pm.get(PAT_NAME.get(k), np.nan) for k in T.key]
else:
    T["MAPE_vs_mean_reference_context"] = np.nan
if args.fix_numbers:
    fixn = json.load(open(args.fix_numbers))
    i9 = fixn["item9_lora"]
    T.loc[T.key == "molmo2_4b_synthft@P1", "MAPE_vs_mean_reference_context"] = i9["molmo2_4b_synthft@P1|mean"]["MAPE"]
    i9 = {norm(k): v for k, v in i9.items()}["molmo2_4b_synthft@P1|A2"]
    r = T[T.key == "molmo2_4b_synthft@P1"].iloc[0]
    CHK["molmo_lora_vs_fix_numbers_item9_A2"] = dict(MAPE=r.MAPE - i9["MAPE"], lo=r.MAPE_lo - i9["lo"], hi=r.MAPE_hi - i9["hi"],
                                                      both5_n=int(r.P5_n - i9["both5_n"]))
    assert max(abs(CHK["molmo_lora_vs_fix_numbers_item9_A2"][k]) for k in ("MAPE", "lo", "hi")) < 1e-3


# ------------------------------------------------------------------ check: per-annotator Tab. 1 reproduced (its own code)
def per_annotator_table_lines():
    """The per-annotator table's own logic (label_day.py per-scene tables of the two single-annotator runs and of
    the mean run; 10,000 resamples, one RNG seeded 20261012), no file written; labels with A1 / A2 (see norm())."""
    def ps(path):
        d = pd.read_csv(path); d["scene_id"] = d.scene_id.astype(int); return d
    P = {"A2": ps(f"{args.a2_run}/work/stage/main150_per_scene.csv"),
         "A1": ps(f"{args.a1_run}/work/stage/main150_per_scene.csv"),
         "mean": ps(f"{args.mean_run}/stage/main150_per_scene.csv")}
    ref = {k: v.groupby("scene_id")[["h", "v"]].first() for k, v in P.items()}
    spec = json.load(open(args.tab1_spec))
    # the Markdown table was written before one display name of the row list was changed
    rename = {"CNN, 5-fold FT (synthetic init)": "CNN, 5-fold FT"}
    keys = [(it[0], rename.get(it[1], it[1])) for it in spec if isinstance(it, list) and not it[0].startswith("@")]
    extra = {"octave-v1 (frozen)": "methods/octave/predictions_octave_150.csv", "CNN fabric-grouped (A1-trained)": "methods/ensemble/cnn_fabric_oof_FLSA_150.csv",
             "ens-v1 = CNN x GPT (frozen)": "methods/ensemble/predictions_ens_v1_150.csv", "ens-v1b (frozen, secondary)": "methods/ensemble_aug/predictions_ens_v1b_150.csv"}
    rng = np.random.default_rng(20261012)
    def ci(x):
        x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]; return np.percentile(b, [2.5, 97.5])
    def ape(pred_h, pred_v, R):
        eh = 100 * np.abs(pred_h - R.h) / R.h; ev = 100 * np.abs(pred_v - R.v) / R.v
        eh = eh.fillna(100); ev = ev.fillna(100); return (eh + ev) / 2, ((eh <= 5) & (ev <= 5))
    rows = []
    for k, name in keys:
        r = {"method": name}
        for a in ("A2", "A1", "mean"):
            d = P[a][P[a].method == k].set_index("scene_id")
            if d.empty: r[a] = None; continue
            R = ref[a].loc[d.index]; s, b5 = ape(d.h_p, d.v_p, R)
            r[a] = s.mean(); r[a + "_ci"] = ci(s); r[a + "_p5"] = int(b5.sum()); r[a + "_n"] = len(s)
        rows.append(r)
    for name, path in extra.items():
        p = pd.read_csv(f"{ROOT}/{path}").set_index("id"); r = {"method": name}
        for a in ("A2", "A1", "mean"):
            R = ref[a]; s, b5 = ape(p.h_count.reindex(R.index), p.v_count.reindex(R.index), R)
            r[a] = s.mean(); r[a + "_ci"] = ci(s); r[a + "_p5"] = int(b5.sum()); r[a + "_n"] = len(s)
        rows.append(r)
    for nm, pa, ra in (("A1 as a prediction of A2", "A1", "A2"), ("A2 as a prediction of A1", "A2", "A1")):
        s, b5 = ape(ref[pa].h, ref[pa].v, ref[ra]); r = {"method": nm, ra: s.mean(), ra + "_ci": ci(s), ra + "_p5": int(b5.sum()), ra + "_n": len(s)}
        rows.append(r)
    TT = pd.DataFrame(rows)
    f = lambda r, a: "" if pd.isna(r.get(a)) or r.get(a) is None else f"{r[a]:.1f} [{r[a+'_ci'][0]:.0f}, {r[a+'_ci'][1]:.0f}] ({r[a+'_p5']})"
    L = ["| method | MAPE vs A2 [95% CI] (P5) | vs A1 | vs mean (frozen) |", "|---|---|---|---|"]
    for _, r in TT.iterrows(): L.append(f"| {r['method']} | {f(r,'A2')} | {f(r,'A1')} | {f(r,'mean')} |")
    return L, TT


if args.per_annotator_md and args.a1_run and args.a2_run and args.mean_run and args.tab1_spec:
    L, TT = per_annotator_table_lines()
    md = [norm(x) for x in open(args.per_annotator_md).read().splitlines()]
    CHK["per_annotator_table_md_reproduced_line_for_line"] = (L == md)
    assert L == md
    # the rows here against the table's A2 column: MAPE (1 dp) and P5 must be identical; its CI is its own bootstrap
    MD_NAME = dict(PAT_NAME, **{A1KEY: "A1 as a prediction of A2"})
    cmp = {}
    for _, r in T.iterrows():
        nm = MD_NAME.get(r.key)
        if nm is None or nm not in set(TT.method):
            cmp[r.label] = "not in the per-annotator table"
            continue
        q = TT[TT.method == nm].iloc[0]
        same_mape = f"{r.MAPE:.1f}" == f"{q['A2']:.1f}"
        same_p5 = int(r.P5_n) == int(q["A2_p5"])
        ci_new = (LD.ci_format(f"{r.MAPE:.1f} [{r.MAPE_lo:.1f}, {r.MAPE_hi:.1f}]", 0))
        ci_md = f"{q['A2']:.1f} [{q['A2_ci'][0]:.0f}, {q['A2_ci'][1]:.0f}]"
        cmp[r.label] = dict(MAPE_same_1dp=same_mape, P5_same=same_p5, metrics_ci=ci_new, md_ci=ci_md,
                            ci_bounds_absdiff=[round(abs(r.MAPE_lo - q["A2_ci"][0]), 2), round(abs(r.MAPE_hi - q["A2_ci"][1]), 2)])
        assert same_mape and same_p5, (r.label, r.MAPE, q["A2"], r.P5_n, q["A2_p5"])
    CHK["vs_per_annotator_table_md_A2"] = cmp

# check: log-ratio endpoint values of logratio_endpoint.py (pre-specified item 5, 3 dp) reproduced
lrp = args.logratio_json or os.path.join(OUT, "logratio_endpoint.json")
if os.path.exists(lrp):
    lr = json.load(open(lrp))
    LRK = {OURS: "PG-Lattice vs A2", "GPT-6.1-sol, P1": "GPT-6.1-sol P1 vs A2", CNNA1: "CNN fabric-grouped vs A2",
           ENS: "ens-v1 vs A2", A1KEY: "A1 (as prediction) vs A2"}
    CHK["vs_logratio_endpoint_json"] = {}
    for k, j in LRK.items():
        r = T[T.key == k].iloc[0]
        CHK["vs_logratio_endpoint_json"][k] = dict(mean=[round(r.mean_abs_log2, 3), lr[j]["mean_abs_log2"]],
                                                   share=[round(r.share_abs_log2_gt_0_5, 3), lr[j]["share_gt_0_5"]])
        assert round(r.mean_abs_log2, 3) == lr[j]["mean_abs_log2"] and round(r.share_abs_log2_gt_0_5, 3) == lr[j]["share_gt_0_5"], k

# ------------------------------------------------------------------ outputs
HDR = [f"% GENERATED by {SELF} -- do not edit by hand",
       "% Main Tab. 1 rows: every row scored against A2 (the textile-trained reference annotator) alone, n = 76 scenes;",
       "% columns method & MAPE [95% CI] & % of counts <=5% & % of counts <=10% & scenes with both counts within 5% (of the",
       "% scenes scored). score150.py metrics() (scene bootstrap 2,000, seed 0; label_day.py ci_format, integer bounds).",
       "% $^{k/76}$ = scenes scored. Row 1 = A1's (the non-expert's) counts as a prediction of A2. $^\\ddagger$ = post hoc: the",
       "% two CNN rows were built after A1's counts were seen (trained on A1, out of fold); CountGD cells->lattice is a",
       "% post-hoc readout fixed before any human count. Every method was frozen before A2's counts were opened.",
       "% Unrounded values, sources, status and mean |log2(pred/ref)|: tab1_ref_rows.csv (same folder)."]
HDRB = [f"% GENERATED by {SELF} -- do not edit by hand",
        "% The paper's Tab. 1 rows (main150_rows_ref.tex with the final row names; best printed value per column in bold",
        "% among the methods, the two baseline rows excluded, ties included)."]
open(f"{OUT}/main150_rows_ref.tex", "w").write("\n".join(HDR + tex) + "\n")
open(f"{OUT}/main150_rows_refB.tex", "w").write("\n".join(HDRB + texB) + "\n")
T.to_csv(f"{OUT}/tab1_ref_rows.csv", index=False, float_format="%.6f")
json.dump(CHK, open(f"{OUT}/tab1_ref_checks.json", "w"), indent=1, default=str)
print("\n".join(texB))
print()
print(T[["label", "n_scored", "MAPE", "MAPE_lo", "MAPE_hi", "w5", "w10", "P5_n", "mean_abs_log2", "share_abs_log2_gt_0_5",
         "mean_abs_log2_lo_posthoc", "mean_abs_log2_hi_posthoc", "MAPE_vs_mean_reference_context"]].round(3).to_string(index=False))
print(json.dumps(CHK, indent=1, default=str))

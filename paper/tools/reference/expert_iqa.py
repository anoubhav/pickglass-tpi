#!/usr/bin/env python3
r"""expert_iqa.py -- main Sec. 5.4 ("Generic quality scores do not rank the error") and supplement Sec. S9: the IQA,
confidence and gate numbers with the textile-trained annotator A2 alone as the reference.

POST HOC.  The pre-specified test (paper/tools/iqa150.py, SHA-256-frozen before any human count of the 76 scenes
existed) scores against the MEAN of the two annotators (score150.gt_table); the paper's macros \FNgenericRhoMin,
\FNiqaRhoClipiqa, \FNconfRho(CI) and \FNgateMinusOursDiff(CI) were computed that way.  Re-scoring against A2 alone was
decided after both annotators' counts were seen, so every A2 number written here is post hoc.

Nothing in the repository is modified.  The frozen code is imported, never edited:
  paper/tools/iqa150.py   signals(), predictions(), errors(), analyse()  (statistics of iqa_generic.py)
  paper/tools/label_day.py paired(), per_scene_ape()                       (paired test and per-scene table)
and the frozen iqa150.py CLI is additionally run unmodified (subprocess, -B, --scene-gt) with --out under --out.

Step 1 (mean reference, the pre-specified analysis): the mean of the two annotators' counts as label_day.py and
  score150.py build it (each annotator's count to 2 decimals, as the tool export stores it; their mean) through
  iqa150.analyse(), and the gate-minus-PG-Lattice paired difference.  With --frozen-iqa-dir (the iqa150.csv /
  iqa150_summary.json of the frozen run) and --final-numbers-json/--final-numbers-tex (the rendered macros) the values
  are checked cell for cell and macro for macro, and the script stops if they differ.
Step 2 (post hoc): the same calls with A2's counts alone, and the paired difference on A2.

Definitions (all from the frozen tools):
  scenes      the 76 scene representatives (rep photo); the reference is the per-scene counts to 2 decimals (the
              annotation tool's export, read by score150.load_labels -> gt_table -> propagate_scene_labels), ascending
              scene order; scenes with h <= 0 or v <= 0 dropped (none are).
  error       per scene, mean APE of the two counts (100 |pred - ref| / ref, averaged over h and v); PG-Lattice's
              counts from methods/predictions_ours_150.csv (= label_day's 'Ours (lattice + fusion)').
  scores      paper/tables/iqa150_scores.csv (pyiqa 0.1.16), protocol 'work' (PRIMARY: the counter's working image,
              INTER_AREA to max side 1024, never up-sampled); 'fix1024' (every crop resized to 1024^2) is secondary.
              Orientation q = -BRISQUE, -NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR (higher = better), so a useful score has
              rho < 0 against error.  Confidence: PG-Lattice's own confidence (higher = more confident).
  Spearman    Pearson correlation of scipy mid-ranks; CI = plain scene bootstrap (each scene its own cluster),
              B = 5000 resamples as multiplicity weights, numpy default_rng(20261011) consumed in analyse()'s order
              (targets ours, log-Gabor, GPT P1, any; classes all / full / quarter), percentile 2.5 / 97.5.
  paired      label_day.py paired(): a - b of the per-scene mean APE on the scenes both answered; 10,000 scene
              resamples with numpy default_rng(0); percentile 2.5 / 97.5.  As for the paper's macros, the per-scene APE
              is the 3-decimal 'ape' column of label_day.py's per-scene table (main150_per_scene.csv), rebuilt here
              with label_day.per_scene_ape() and round(3) (or read from a run with --a2-run / --mean-run); label_day's
              paired() on unrounded predictions is the cross-check (|diff| <= 0.005).

    python paper/tools/reference/expert_iqa.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR] [optional checks]

Optional checks (inputs that are not part of the repository; each is skipped when not given):
  --frozen-iqa-dir DIR       iqa150.py output of the frozen mean-reference run (iqa150.csv, iqa150_summary.json)
  --final-numbers-json/-tex  the rendered macros of the paper (final_numbers.json / final_numbers.tex)
  --mean-run DIR             paper/tools/label_day.py run against the mean of both annotators (stage/, numbers.json)
  --a2-run DIR               paper/tools/label_day.py single-annotator run on A2's export (folder holding work/)
Outputs (--out): expert_iqa.json, expert_iqa_per_scene.csv, frozen_tool_runs/<ref>/{scene_gt.csv, iqa150.csv,
iqa150_summary.json, stdout.txt}.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # importing the frozen tools must not write __pycache__ into paper/tools

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TOOLS = os.path.join(ROOT, "paper", "tools")
sys.path.insert(0, TOOLS)
import iqa150 as I  # noqa: E402  (frozen; imported, not edited)
import iqa_generic as G  # noqa: E402
import label_day as LD  # noqa: E402  (frozen; paired() and per_scene_ape())

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
ap.add_argument("--frozen-iqa-dir", default=None, help="optional: iqa150 output of the frozen mean-reference run")
ap.add_argument("--final-numbers-json", default=None, help="optional: rendered macros (JSON)")
ap.add_argument("--final-numbers-tex", default=None, help="optional: rendered macros (TeX)")
ap.add_argument("--mean-run", default=None, help="optional: label_day.py run against the mean of both annotators")
ap.add_argument("--a2-run", default=None, help="optional: label_day.py single-annotator run on A2's export")
args = ap.parse_args()
OUT = os.path.abspath(args.out)   # absolute: the frozen iqa150.py CLI runs with cwd=paper/
FREEZE = {e["path"]: e for e in json.load(open(os.path.join(ROOT, "tools", "freeze_checksums.json")))}

OURS, GATE, GPT = "Ours (lattice + fusion)", "cascade-gate", "GPT-6.1-sol, P1"
T_OURS = "ours"                                                       # iqa150 target name of PG-Lattice
GENERIC4 = ["BRISQUE [work]", "NIQE [work]", "MUSIQ [work]", "TOPIQ-NR [work]"]   # the four of \FNgenericRhoMin
GENERIC5 = GENERIC4 + ["CLIP-IQA [work]"]
PRESPEC = {"BRISQUE [work]", "NIQE [work]"}
CONF = "ours confidence"
POST_HOC = ("post hoc: A2 (textile-trained) alone as the reference, chosen after both annotators' counts were seen; "
            "the pre-specified analysis scores against the mean of the two")


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha1(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


# --------------------------------------------------------------------- rendering exactly as the paper's macros
def num(x, d=2):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return r"\ensuremath{" + s + "}" if s.startswith("-") else s


def signed(x, d=1):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = "+" + s.lstrip("+-")
    return r"\ensuremath{" + s + "}"


def ci(lo, hi, d=1, sign=False):
    g = (lambda v: signed(v, d)) if sign else (lambda v: num(v, d))
    return f"[{g(lo)}, {g(hi)}]"


def plain(t):
    import re
    return re.sub(r"\\ensuremath\{([^{}]*)\}", r"\1", t)


# --------------------------------------------------------------------- references (as label_day.py / score150.py)
def scene_gt(path):
    """Per-scene counts -> the scored reference: <ax>_tpi to 2 decimals (the tool export), <= 0 or missing = missing,
    ascending scene order (index = the scene representative's id)."""
    L = pd.read_csv(path)
    L["scene_id"] = L.scene_id.astype(int)
    L = L.set_index("scene_id").sort_index()
    gs = pd.DataFrame(index=L.index)
    for ax in ("h", "v"):
        c = pd.to_numeric(L[f"{ax}_tpi"], errors="coerce").round(2)
        if f"{ax}_count" in L:   # score150.load_labels: <ax>_tpi when present, else <ax>_count
            c = c.fillna(pd.to_numeric(L[f"{ax}_count"], errors="coerce"))
        gs[ax] = c.where(c > 0)
    gs.index.name = None
    return gs


# --------------------------------------------------------------------- the frozen IQA analysis
def frozen_analysis(gt, label):
    """iqa150.main() between label loading and output, unchanged: filter, signals, reps check, predictions,
    default_rng(G.SEED), analyse()."""
    gt = gt[(gt.h > 0) & (gt.v > 0)]
    X = I.signals(I.P_SCORES)
    reps = set(X.index[X.is_scene_rep.astype(bool)])
    if not set(gt.index) <= reps:
        raise SystemExit(f"scene ids are not all scene representatives: {sorted(set(gt.index) - reps)[:10]}")
    P, _ = I.predictions()
    rng = np.random.default_rng(G.SEED)
    return I.analyse(gt, X, P, rng, f"{len(gt)} scenes (rep photo)"), gt, X, P


def run_frozen_cli(gt, out):
    """The unmodified iqa150.py CLI on the same reference table (--scene-gt id,h,v), output under --out only."""
    os.makedirs(out, exist_ok=True)
    sg = os.path.join(out, "scene_gt.csv")
    gt[["h", "v"]].rename_axis("id").reset_index().to_csv(sg, index=False)
    cmd = [sys.executable, "-B", os.path.join(TOOLS, "iqa150.py"), "--scene-gt", sg, "--out", out]
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "paper"), capture_output=True, text=True)
    open(os.path.join(out, "stdout.txt"), "w").write(r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else ""))
    if r.returncode:
        raise SystemExit(f"iqa150.py CLI failed ({r.returncode}); see {out}/stdout.txt")
    return pd.read_csv(os.path.join(out, "iqa150.csv"))


def same_table(A, B):
    """cell-for-cell equality of two iqa150.csv tables (numbers to 1e-9 after the tool's round(4))."""
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return False, "shape/columns differ"
    worst = 0.0
    for c in A.columns:
        a, b = A[c], B[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            fa, fb = a.to_numpy(float), b.to_numpy(float)
            if not np.array_equal(np.isnan(fa), np.isnan(fb)):
                return False, f"NaN pattern differs in {c}"
            m = ~np.isnan(fa)
            worst = max(worst, float(np.max(np.abs(fa[m] - fb[m]))) if m.any() else 0.0)
        elif not (a.fillna("").astype(str).to_numpy() == b.fillna("").astype(str).to_numpy()).all():   # '' == NaN after CSV
            return False, f"text differs in {c}"
    return worst <= 1e-9, f"max abs numeric diff {worst:.2e}"


def iqa_block(R):
    """Target PG-Lattice ('ours'), all scenes: every generic score at both protocols, the task signals."""
    A = R[(R.target_method == T_OURS) & (R.resolution == "all")].set_index("predictor")

    def row(p):
        r = A.loc[p]
        d = dict(n=int(r.n), rho=float(r.spearman), lo=float(r.spearman_lo), hi=float(r.spearman_hi),
                 ci_spans_0=bool(r.spearman_lo <= 0 <= r.spearman_hi))
        if p != CONF:
            d.update(d_rho_vs_conf=float(r.d_spearman_vs_conf), d_rho_vs_conf_lo=float(r.d_spearman_vs_conf_lo),
                     d_rho_vs_conf_hi=float(r.d_spearman_vs_conf_hi))
        if p.endswith("[work]") or p.endswith("[fix1024]"):
            d["status_metric"] = ("pre-registered metric" if p.split(" [")[0] in ("BRISQUE", "NIQE") else
                                  "exploratory metric") + ("; primary input size" if p.endswith("[work]") else
                                                           "; secondary input size")
        return d
    work = {p: row(p) for p in GENERIC5}
    fix = {p.replace("[work]", "[fix1024]"): row(p.replace("[work]", "[fix1024]")) for p in GENERIC5}
    r4 = [work[p]["rho"] for p in GENERIC4]
    r5 = [work[p]["rho"] for p in GENERIC5]
    argmin = lambda keys: min(keys, key=lambda k: work[k]["rho"])  # noqa: E731
    argmax = lambda keys: max(keys, key=lambda k: work[k]["rho"])  # noqa: E731
    return dict(
        n_scenes=int(A.loc[CONF].n), n_fail5=int(A.loc[CONF].n_fail5),
        generic_work=work, generic_fix1024_secondary=fix,
        summary=dict(
            four_as_macro_FNgenericRhoMin=dict(metrics=GENERIC4, rho_min=min(r4), rho_min_metric=argmin(GENERIC4),
                                              rho_max=max(r4), rho_max_metric=argmax(GENERIC4),
                                              every_ci_spans_0=all(work[p]["ci_spans_0"] for p in GENERIC4)),
            all_five=dict(metrics=GENERIC5, rho_min=min(r5), rho_min_metric=argmin(GENERIC5),
                          rho_max=max(r5), rho_max_metric=argmax(GENERIC5),
                          every_ci_spans_0=all(work[p]["ci_spans_0"] for p in GENERIC5)),
            pre_registered_two=dict(metrics=sorted(PRESPEC), every_ci_spans_0=all(work[p]["ci_spans_0"] for p in PRESPEC)),
            fix1024_every_ci_spans_0=all(v["ci_spans_0"] for v in fix.values())),
        confidence=row(CONF),
        other_task_signals={p: row(p) for p in ("resolvability eta", "cascade gate (frozen; 1 = kept offline)")})


def macros_from(blk):
    """The five Sec. 5.4 IQA macros, rendered as the paper's macro file renders them."""
    w, c = blk["generic_work"], blk["confidence"]
    s4 = blk["summary"]["four_as_macro_FNgenericRhoMin"]
    return {"FNgenericRhoMin": num(s4["rho_min"]), "FNgenericRhoMax": num(s4["rho_max"]),
            "FNiqaRhoClipiqa": num(w["CLIP-IQA [work]"]["rho"]), "FNconfRho": num(c["rho"]),
            "FNconfRhoCI": ci(c["lo"], c["hi"], 2)}


# --------------------------------------------------------------------- label_day.py per-scene tables and paired()
S150 = LD.load_score150()
PREDS = S150.load_predictions()
PREDS.update(S150.load_extra_wide([os.path.join(ROOT, "paper", "tables", "src", "extra150_wide.csv") + "::cascade-gate"]))
for _k in list(PREDS):
    PREDS[_k] = PREDS[_k].copy()
    PREDS[_k].index = PREDS[_k].index.astype(int)


def per_scene_table(gt, methods=(OURS, GATE, GPT)):
    """label_day.py step_score: per_scene_ape(gt, pr) for each method, written to main150_per_scene.csv with round(3)."""
    per = []
    for name in methods:
        e = LD.per_scene_ape(gt, PREDS[name])
        e.insert(0, "method", name)
        per.append(e.reset_index().rename(columns={"index": "scene_id"}))
    return pd.concat(per, ignore_index=True).round(3)


def paired_ld(ea, eb, n_boot=10000, seed=0):
    """label_day.py paired() core: a - b on the scenes both answered."""
    c = ea.index.intersection(eb.index)
    d = (ea[c] - eb[c]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(n=int(len(c)), mape_a=float(ea[c].mean()), mape_b=float(eb[c].mean()), delta=float(d.mean()),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                a_closer=int((d < 0).sum()), b_closer=int((d > 0).sum()), tied=int((d == 0).sum()))


GT2 = {}          # the scored reference per protocol ('mean', 'A2'), set in main()
PER_SCENE = {}    # label_day per-scene table per protocol (rebuilt, or read from a run)
PAIRED_CSV = {}   # label_day main150_paired.csv per protocol when a run is given


def stored_paired(ref, a, b):
    """label_day.py's stored paired row (main150_paired.csv, rounded to 3 decimals): read from the run when given,
    else recomputed with label_day.paired() on the unrounded predictions and rounded the same way."""
    if ref in PAIRED_CSV:
        st = pd.read_csv(PAIRED_CSV[ref]).set_index("comparison").loc[f"{a} vs {b}"]
        return {k: float(st[k]) for k in ("delta", "lo", "hi")}
    r = LD.paired(GT2[ref], PREDS[a], PREDS[b])
    return {k: float(np.round(r[k], 3)) for k in ("delta", "lo", "hi")}


def gate_minus_ours(ref):
    ps = PER_SCENE[ref].copy()
    ps["scene_id"] = ps.scene_id.astype(int)
    M = lambda k: ps[ps.method == k].set_index("scene_id")  # noqa: E731
    r = paired_ld(M(GATE).ape, M(OURS).ape)
    # cross-check 1: label_day.py paired() itself on the per-scene table's (3-decimal) reference and predictions
    gt = M(OURS)[["h", "v"]]
    pr = lambda k: M(k)[["h_p", "v_p"]].rename(columns={"h_p": "h", "v_p": "v"})  # noqa: E731
    ld = LD.paired(gt, pr(GATE), pr(OURS))
    # cross-check 2: label_day's own stored result (unrounded counts)
    st = stored_paired(ref, GATE, OURS)
    r["checks"] = {
        "label_day.paired() on the per-scene table": {k: ld[k] for k in ("delta", "lo", "hi")},
        "label_day main150_paired.csv (unrounded)": st,
        "max_abs_diff": max(abs(r[k] - ld[k]) for k in ("delta", "lo", "hi")),
        "max_abs_diff_vs_stored": max(abs(r[k] - st[k]) for k in ("delta", "lo", "hi")),
    }
    r["checks"]["ok"] = r["checks"]["max_abs_diff"] <= 0.005 and r["checks"]["max_abs_diff_vs_stored"] <= 0.005
    r["n_scenes_gate_changes_prediction"] = int(((M(GATE).h_p - M(OURS).h_p.reindex(M(GATE).index)).abs()
                                                 + (M(GATE).v_p - M(OURS).v_p.reindex(M(GATE).index)).abs()
                                                 > 0.01).sum())
    r["rendered"] = {"diff": signed(r["delta"]), "ci": ci(r["lo"], r["hi"], sign=True)}
    r["ci_spans_0"] = bool(r["lo"] <= 0 <= r["hi"])
    return r


def sha1_listed(FN, path, tail):
    """the file's sha1 equals the final-numbers record of the input whose path ends with `tail`."""
    want = [v for k, v in FN["input_sha1"].items() if k.replace(os.sep, "/").endswith(tail)]
    return bool(want) and sha1(path) in want


# --------------------------------------------------------------------- main
def main():
    log = []
    say = lambda s: (print(s), log.append(s))  # noqa: E731
    os.makedirs(OUT, exist_ok=True)

    # 0. integrity of the frozen inputs (tools/freeze_checksums.json)
    integ = {
        "iqa150.py sha256 == frozen": sha256(os.path.join(TOOLS, "iqa150.py")) == FREEZE["paper/tools/iqa150.py"]["sha256_original"],
        "iqa150_scores.csv sha256 == frozen": sha256(I.P_SCORES) == FREEZE["paper/tables/iqa150_scores.csv"]["sha256_original"],
        "cascade_150_decisions.csv md5 == frozen": hashlib.md5(open(I.P_GATE, "rb").read()).hexdigest() == I.GATE_MD5,
    }
    # iqa_generic.py is released with comment/docstring-only edits (statistics unchanged), so its released hash is
    # checked against tools/freeze_checksums.json (sha256_released)
    soft = {"iqa_generic.py sha256 == tools/freeze_checksums.json (released copy)":
            sha256(os.path.join(TOOLS, "iqa_generic.py")) == FREEZE["paper/tools/iqa_generic.py"]["sha256_released"]}
    fz = FN = None
    if args.frozen_iqa_dir:
        fz = json.load(open(os.path.join(args.frozen_iqa_dir, "iqa150_summary.json")))
        integ["iqa150.py sha256 == frozen run's record"] = sha256(os.path.join(TOOLS, "iqa150.py")) == fz["code_sha256"]["iqa150.py"]
        integ["iqa150_scores.csv sha256 == frozen run's record"] = sha256(I.P_SCORES) == fz["scores_sha256"]
        integ["iqa_generic.py: frozen run's record == tools/freeze_checksums.json original"] = (
            fz["code_sha256"]["iqa_generic.py"] == FREEZE["paper/tools/iqa_generic.py"]["sha256_original"])
    if args.final_numbers_json:
        FN = json.load(open(args.final_numbers_json))
        if args.mean_run:
            integ["mean-run main150_per_scene.csv sha1 == final_numbers.json input"] = sha1_listed(
                FN, os.path.join(args.mean_run, "stage", "main150_per_scene.csv"), "stage/main150_per_scene.csv")
        if args.frozen_iqa_dir:
            integ["mean-run iqa150_summary.json sha1 == final_numbers.json input"] = sha1_listed(
                FN, os.path.join(args.frozen_iqa_dir, "iqa150_summary.json"), "iqa150/iqa150_summary.json")
    for k, v in {**integ, **soft}.items():
        say(f"[integrity] {k}: {v}")
    if not all(integ.values()):
        raise SystemExit("integrity check failed")

    # references and label_day per-scene tables
    A1r, A2r = scene_gt(args.a1), scene_gt(args.a2)
    assert list(A1r.index) == list(A2r.index), "the two per-scene files must hold the same scenes"
    GT2["mean"] = (A1r[["h", "v"]] + A2r[["h", "v"]]) / 2   # score150.gt_table: mean over the two annotators
    GT2["A2"] = A2r[["h", "v"]]
    for ref, run in (("mean", args.mean_run and os.path.join(args.mean_run, "stage")),
                     ("A2", args.a2_run and os.path.join(args.a2_run, "work", "stage"))):
        if run:
            PER_SCENE[ref] = pd.read_csv(os.path.join(run, "main150_per_scene.csv"))
            PAIRED_CSV[ref] = os.path.join(run, "main150_paired.csv")
        else:
            PER_SCENE[ref] = per_scene_table(GT2[ref])

    # 1. the frozen mean-reference values
    gt_mean = GT2["mean"]
    R_mean, gt_mean, X, P = frozen_analysis(gt_mean, "mean")
    cli_mean = run_frozen_cli(gt_mean, os.path.join(OUT, "frozen_tool_runs", "mean_reference"))
    ok_cli_mean_inproc, why_cli_mean_inproc = same_table(cli_mean, R_mean.round(4).reset_index(drop=True))
    say(f"[mean] unmodified iqa150.py CLI (--scene-gt) == in-process: {ok_cli_mean_inproc} ({why_cli_mean_inproc})")
    if args.frozen_iqa_dir:
        frozen_csv = os.path.join(args.frozen_iqa_dir, "iqa150.csv")
        ok_tab, why = same_table(R_mean.round(4).reset_index(drop=True), pd.read_csv(frozen_csv))
        say(f"[reproduce] in-process iqa150.analyse() vs frozen iqa150.csv: {ok_tab} ({why})")
        ok_cli, why_cli = same_table(cli_mean, pd.read_csv(frozen_csv))
        say(f"[reproduce] unmodified iqa150.py CLI vs frozen iqa150.csv: {ok_cli} ({why_cli})")
    else:
        ok_tab = ok_cli = None
        why = why_cli = "not checked (no --frozen-iqa-dir)"
    blk_mean = iqa_block(R_mean)
    mac_mean = macros_from(blk_mean)
    gate_mean = gate_minus_ours("mean")
    mac_mean["FNgateMinusOursDiff"] = gate_mean["rendered"]["diff"]
    mac_mean["FNgateMinusOursDiffCI"] = gate_mean["rendered"]["ci"]
    rep, exact_vals = {}, None
    if FN is not None:
        for k, v in mac_mean.items():
            frozen = FN["macros"][k]["tex"]
            rep[k] = dict(reproduced=v, frozen=frozen, match=(v == frozen))
            say(f"[reproduce] \\{k}: reproduced {plain(v)}  frozen {plain(frozen)}  match={v == frozen}")
        if args.final_numbers_tex:
            tex = open(args.final_numbers_tex).read()
            for k, v in mac_mean.items():
                rep[k]["in_final_numbers_tex"] = ("\\newcommand{\\" + k + "}{" + v + "}") in tex
        exact_vals = {"FNgateMinusOursDiff": abs(gate_mean["delta"] - FN["macros"]["FNgateMinusOursDiff"]["value"]) < 1e-12,
                      "FNgateMinusOursDiffCI": np.allclose([gate_mean["lo"], gate_mean["hi"]],
                                                           FN["macros"]["FNgateMinusOursDiffCI"]["value"], atol=1e-12, rtol=0),
                      "FNconfRho": abs(blk_mean["confidence"]["rho"] - FN["macros"]["FNconfRho"]["value"]) < 5e-5}
        say(f"[reproduce] unrounded values equal to final_numbers.json: {exact_vals}")
    else:
        for k, v in mac_mean.items():
            rep[k] = dict(reproduced=v)
            say(f"[mean] \\{k}: {plain(v)}")
    checked = ok_tab is not None and FN is not None
    reproduced = (ok_tab and ok_cli and all(r["match"] and r.get("in_final_numbers_tex", True) for r in rep.values())
                  and all(exact_vals.values()) and gate_mean["checks"]["ok"]) if checked else None
    if checked and not reproduced:
        raise SystemExit("could not reproduce the frozen mean-reference values; A2 numbers NOT computed")
    if not gate_mean["checks"]["ok"] or not ok_cli_mean_inproc:
        raise SystemExit("mean reference: cross-checks failed; A2 numbers NOT computed")

    # 2. A2 alone (post hoc)
    gt_ng = GT2["A2"]
    R_ng, gt_ng, _, _ = frozen_analysis(gt_ng, "A2")
    L = pd.read_csv(args.a2).set_index("scene_id")
    ps_ng = PER_SCENE["A2"]
    o_ng = ps_ng[ps_ng.method == OURS].set_index("scene_id")
    E_ng = I.errors(gt_ng, P)[T_OURS]
    chk = {
        "n_scenes": int(len(gt_ng)),
        "same scene set as A2's per-scene counts": set(gt_ng.index) == set(L.index),
        "max |h,v - A2 per-scene h_tpi,v_tpi| (TPI; the scored reference is rounded to 2 decimals)": float(max(
            (gt_ng.h - L.h_tpi.reindex(gt_ng.index)).abs().max(), (gt_ng.v - L.v_tpi.reindex(gt_ng.index)).abs().max())),
        "max |h,v - A2 per-scene table reference|": float(max(
            (gt_ng.h - o_ng.h.reindex(gt_ng.index)).abs().max(), (gt_ng.v - o_ng.v.reindex(gt_ng.index)).abs().max())),
        "max |iqa150 ours error - A2 per-scene table Ours APE| (pp)": float((E_ng.ape - o_ng.ape.reindex(E_ng.index)).abs().max()),
    }
    # with both runs: the A2 single-annotator run's reference = A2 alone inside the two-annotator run; label_day's own
    # reference_alone paired results must be identical in both
    if args.mean_run and args.a2_run:
        nj = json.load(open(os.path.join(args.mean_run, "numbers.json")))["score"]["reference_alone"]
        nj = {k.replace("FL_NG", "A2"): v for k, v in nj.items()}["A2"]["paired"]   # the run labels A2 by its annotator code
        st = pd.read_csv(PAIRED_CSV["A2"]).set_index("comparison")
        chk["dual run reference_alone A2 == A2 run main150_paired (gate vs GPT, ours vs GPT)"] = all(
            abs(nj[k]["delta"] - float(st.at[k, "delta"])) < 6e-4 and abs(nj[k]["lo"] - float(st.at[k, "lo"])) < 6e-4
            for k in (f"{GATE} vs {GPT}", f"{OURS} vs {GPT}"))
    for k, v in chk.items():
        say(f"[A2 check] {k}: {v}")
    if not (chk["same scene set as A2's per-scene counts"] and chk[
            "max |h,v - A2 per-scene h_tpi,v_tpi| (TPI; the scored reference is rounded to 2 decimals)"] < 0.005
            and chk["max |h,v - A2 per-scene table reference|"] < 6e-4
            and chk["max |iqa150 ours error - A2 per-scene table Ours APE| (pp)"] < 6e-4
            and chk.get("dual run reference_alone A2 == A2 run main150_paired (gate vs GPT, ours vs GPT)", True)):
        raise SystemExit("A2 reference is not consistent across its sources")
    cli_ng = run_frozen_cli(gt_ng, os.path.join(OUT, "frozen_tool_runs", "A2_posthoc"))
    ok_cli_ng, why_ng = same_table(cli_ng, R_ng.round(4).reset_index(drop=True))
    say(f"[A2] unmodified iqa150.py CLI (--scene-gt A2) == in-process: {ok_cli_ng} ({why_ng})")
    # sensitivity: the unrounded per-scene counts as the reference
    R_sg, _, _, _ = frozen_analysis(pd.DataFrame({"h": L.h_tpi, "v": L.v_tpi}).rename_axis("id"), "A2 scene-gt")
    sg = iqa_block(R_sg)
    blk_ng = iqa_block(R_ng)
    sens = max(abs(sg["generic_work"][p][k] - blk_ng["generic_work"][p][k]) for p in GENERIC5 for k in ("rho", "lo", "hi"))
    sens = max(sens, max(abs(sg["confidence"][k] - blk_ng["confidence"][k]) for k in ("rho", "lo", "hi")))
    say(f"[A2] unrounded per-scene counts as the reference: max |diff| in rho / CI bounds = {sens:.4f}")
    gate_ng = gate_minus_ours("A2")
    mac_ng = macros_from(blk_ng)
    mac_ng["FNgateMinusOursDiff"] = gate_ng["rendered"]["diff"]
    mac_ng["FNgateMinusOursDiffCI"] = gate_ng["rendered"]["ci"]

    # per-scene table (audit)
    J = E_ng.join(X, how="inner")
    gp = ps_ng[ps_ng.method == GATE].set_index("scene_id").ape
    pst = pd.DataFrame({"scene_id": J.index, "ref_h_A2": gt_ng.h.reindex(J.index).round(3).to_numpy(),
                        "ref_v_A2": gt_ng.v.reindex(J.index).round(3).to_numpy(),
                        "ours_ape_A2": J.ape.round(3).to_numpy(), "ours_fail5": J.fail5.to_numpy(),
                        "gate_ape_A2": gp.reindex(J.index).to_numpy(),
                        "ours_ape_mean_ref": I.errors(gt_mean, P)[T_OURS].ape.reindex(J.index).round(3).to_numpy(),
                        "ours_confidence": J.conf.to_numpy(), "gate_escalate": J.gate_escalate.to_numpy(), "S_px": J.S.to_numpy()})
    for m, (nm, sgn) in G.METRICS.items():
        pst[f"q_{m}_work"] = (sgn * J[f"{m}_work"]).to_numpy()
    pst.to_csv(os.path.join(OUT, "expert_iqa_per_scene.csv"), index=False)

    # 3. sentence
    s4n = blk_ng["summary"]["all_five"]
    c = blk_ng["confidence"]
    f2 = lambda x: plain(num(x))  # noqa: E731
    span_all = s4n["every_ci_spans_0"]
    gw = blk_ng["generic_work"]
    useful = [p for p in GENERIC5 if gw[p]["hi"] < 0]           # CI below 0 = higher quality, smaller error
    wrong = [p for p in GENERIC5 if gw[p]["lo"] > 0]            # CI above 0 = higher quality, LARGER error
    nm = lambda ps: " and ".join(p.split(" [")[0] for p in ps)  # noqa: E731
    if span_all:
        note = ", every CI spanning 0"
    elif not useful:
        note = (f"; only {nm(wrong)}'s CI excludes 0, with the wrong sign" if len(wrong) == 1 else
                f"; only the CIs of {nm(wrong)} exclude 0, with the wrong sign")
    else:
        note = f"; CI below 0 for {nm(useful)}"
    lead = ("no generic score ranks \\ours's per-scene error usefully" if not useful else
            "generic scores rank \\ours's per-scene error at most weakly")
    sentence = (
        f"Against the textile-trained annotator alone (post hoc), {lead} "
        f"($\\rho$ from ${f2(s4n['rho_min'])}$ to ${f2(s4n['rho_max'])}${note}), "
        f"its own confidence gives $\\rho={f2(c['rho'])}$ [${f2(c['lo'])}$, ${f2(c['hi'])}$], "
        f"and the gate changes its error by ${plain(signed(gate_ng['delta']))}$ points "
        f"[${plain(signed(gate_ng['lo']))}$, ${plain(signed(gate_ng['hi']))}$].")
    # the Sec. 5.4 wording, checked against A2 (range \FNgenericRhoMin..\FNiqaRhoClipiqa = five scores)
    claim_checks = {
        "Sec. 5.4 'no CI lies below 0': no generic score is useful (five scores, range \\FNgenericRhoMin..\\FNiqaRhoClipiqa)":
            not useful,
        "Sec. 5.4 'CLIP-IQA's [CI] lies above it' (above 0, the wrong sign)": "CLIP-IQA [work]" in wrong,
        "every CI spanning 0, all five scores (not claimed in Sec. 5.4, since CLIP-IQA's CI lies above 0)": span_all,
        "every CI spanning 0, the four of \\FNgenericRhoMin (BRISQUE, NIQE, MUSIQ, TOPIQ-NR)":
            blk_ng["summary"]["four_as_macro_FNgenericRhoMin"]["every_ci_spans_0"],
        "every CI spanning 0, the two pre-registered (BRISQUE, NIQE)": blk_ng["summary"]["pre_registered_two"]["every_ci_spans_0"],
        "scores whose CI excludes 0 with the wrong sign (exploratory, unadjusted)": wrong,
        "confidence: |rho| < 0.5 (claim 'weakly, if at all')": abs(c["rho"]) < 0.5,
        "confidence CI spans 0": c["ci_spans_0"],
        "gate barely changes the error (CI includes 0)": gate_ng["ci_spans_0"],
    }
    for k, v in claim_checks.items():
        say(f"[claim A2] {k}: {v}")

    inputs = [os.path.join(TOOLS, "iqa150.py"), os.path.join(TOOLS, "iqa_generic.py"), os.path.join(TOOLS, "label_day.py"),
              I.P_SCORES, I.P_OURS, I.P_VLM, I.P_GATE, os.path.join(ROOT, "paper", "tables", "src", "extra150_wide.csv")]
    res = {
        "script": os.path.relpath(os.path.abspath(__file__), ROOT),
        "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": {"mean_reference": "pre-specified (frozen tools" + (", reproduced)" if reproduced else "; frozen outputs not checked)"),
                   "A2": POST_HOC},
        "definitions": {
            "scenes": "76 scene representatives (rep photo); reference = per-scene counts to 2 decimals (the tool export "
                      "read by score150.load_labels -> gt_table -> propagate_scene_labels); h <= 0 or v <= 0 dropped (none)",
            "error": "per scene: mean of APE_h and APE_v, APE = 100 |pred - ref| / ref; PG-Lattice = "
                     "methods/predictions_ours_150.csv (label_day 'Ours (lattice + fusion)')",
            "score_orientation": "q = -BRISQUE, -NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR; higher = better quality; a useful "
                                 "score has rho < 0 against error. Confidence: higher = more confident (useful: rho < 0)",
            "input_size_protocol": "'work' (PRIMARY): the counter's working image, INTER_AREA to max side 1024, never "
                                   "up-sampled; 'fix1024' (secondary): every crop resized to 1024^2 (INTER_CUBIC when "
                                   "smaller). Scores: paper/tables/iqa150_scores.csv (pyiqa 0.1.16)",
            "pre_registration": "BRISQUE and NIQE at 'work' pre-registered; MUSIQ, CLIP-IQA, TOPIQ-NR exploratory",
            "spearman_ci": "Pearson on scipy mid-ranks; plain scene bootstrap (each scene its own cluster), B = 5000 "
                           "multiplicity-weight resamples, numpy default_rng(20261011) consumed in iqa150.analyse() order "
                           "(target ours / all scenes is the first draw), percentile 2.5 / 97.5",
            "paired": "label_day.py paired(): mean over scenes of (APE_gate - APE_ours) on scenes both answered; "
                      "10,000 scene resamples, numpy default_rng(0), percentile 2.5 / 97.5; per-scene APE = the 3-decimal "
                      "'ape' column of label_day.py's per-scene table, as used for the paper's macro",
            "min_max_of_FNgenericRhoMin": "over BRISQUE, NIQE, MUSIQ, TOPIQ-NR [work]; the Sec. 5.4 text's upper end is "
                                          "\\FNiqaRhoClipiqa, so the five-score range is also given",
        },
        "inputs": {os.path.relpath(p, ROOT): sha256(p) for p in inputs},
        "integrity": {**integ, **soft},
        "reproduction_mean_reference": {
            "ok": reproduced, "iqa150_csv_cell_for_cell": [ok_tab, why], "iqa150_cli_cell_for_cell": [ok_cli, why_cli],
            "iqa150_cli_equals_in_process": [ok_cli_mean_inproc, why_cli_mean_inproc],
            "macros": rep, "unrounded_equal_to_final_numbers_json": exact_vals,
            "iqa": blk_mean, "gate_minus_ours": gate_mean},
        "A2_posthoc": {
            "status": POST_HOC,
            "reference_checks": chk,
            "iqa150_cli_equals_in_process": [ok_cli_ng, why_ng],
            "scene_gt_route_max_abs_diff": sens,
            "iqa": blk_ng,
            "gate_minus_ours": gate_ng,
            "macro_style_values": {k: plain(v) for k, v in mac_ng.items()},
            "claim_checks": claim_checks,
        },
        "side_by_side": {k: {"mean_reference_frozen": plain(mac_mean[k]), "A2_posthoc": plain(mac_ng[k])}
                         for k in mac_mean},
        "latex_sentence_suggestion": sentence,
        "log": log,
    }
    json.dump(res, open(os.path.join(OUT, "expert_iqa.json"), "w"), indent=1, default=float)

    print("\nA2 ALONE (post hoc), target PG-Lattice, 76 scenes, protocol 'work':")
    for p in GENERIC5 + [CONF]:
        d = blk_ng["confidence"] if p == CONF else blk_ng["generic_work"][p]
        print(f"  {p:18s} rho {d['rho']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  spans 0: {d['ci_spans_0']}")
    print(f"  four (macro): min {blk_ng['summary']['four_as_macro_FNgenericRhoMin']['rho_min']:+.3f}  max "
          f"{blk_ng['summary']['four_as_macro_FNgenericRhoMin']['rho_max']:+.3f}; all five: min {s4n['rho_min']:+.3f}  "
          f"max {s4n['rho_max']:+.3f}; every CI spans 0: {span_all}")
    print(f"  cascade gate - PG-Lattice: {gate_ng['delta']:+.3f} [{gate_ng['lo']:+.3f}, {gate_ng['hi']:+.3f}] "
          f"(n {gate_ng['n']}; gate closer {gate_ng['a_closer']}, PG-Lattice closer {gate_ng['b_closer']}, tied {gate_ng['tied']})")
    print("\nside by side (mean frozen -> A2 post hoc):")
    for k, v in res["side_by_side"].items():
        print(f"  \\{k}: {v['mean_reference_frozen']} -> {v['A2_posthoc']}")
    print("\n" + sentence)
    print("wrote", os.path.join(OUT, "expert_iqa.json"))


if __name__ == "__main__":
    main()

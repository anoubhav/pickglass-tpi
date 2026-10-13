"""iqa150.py -- LABEL DAY: do generic no-reference IQA scores predict counting error on the 76 scenes?

The PickGlass-150 counterpart of tools/iqa_generic.py part A (legacy degradation set).  Written on
Oct 11 2026, before any human count of the 76 scenes existed, and NOT run on labels then; only its
plumbing was checked on clearly fake labels in a scratch directory.

PRE-SPECIFIED.  This is the pre-specified 76-scene analysis of the generic-IQA question.  Its code (this file and
the statistics it imports from tools/iqa_generic.py) and its label-free input tables/iqa150_scores.csv were
SHA-256-recorded on 2026-10-12 ~04:00 UTC (Oct 11 ~24:00 EDT, Oct 12 ~09:30 IST), before any human count of the
76 scenes existed (paper/CHANGELOG.md, "Job IQA"; handover/results/generic_iqa_summary.md).  Do not edit iqa150.py
or iqa_generic.py after that; copy them to a new file if a change is ever needed, and say so.  PRIMARY: protocol 'work', all scenes, the three targets below, every
signal (BRISQUE and NIQE are the pre-specified generic metrics; MUSIQ, CLIP-IQA and TOPIQ-NR are scored the same
way but were already seen on the legacy set).  SECONDARY: protocol 'fix1024' and the two resolution classes.
Any other cut of these data is exploratory.

Inputs (read-only):
  --labels     the label-day merged scoring CSVs that quality150.py reads, i.e. the HEADLINE label set
               (label_day.py: <run>/work/merged/scoring_headline/<annotator>.csv, one per annotator; the mean over
               annotators is the reference, as in score150.py), comma-separated, repeated or a quoted glob; the
               ground truth is built EXACTLY as in quality150.py --unit scene: score150.load_labels
               (done = 1, nonwoven = 0, newest record per image x annotator, h_tpi / v_tpi when present,
               cleared lines missing) -> score150.gt_table (mean over annotators) -> the annotation tool's
               propagate_scene_labels.propagate -> one row per scene, indexed by the representative's id.
  --scene-gt   alternatively a CSV id,h,v (one row per scene representative, rep frame), e.g. label_day's
               reference table; skips the label loading.
  tables/iqa150_scores.csv                  generic IQA of every rectified crop (label-free; iqa_scores.py)
  methods/predictions_ours_150.csv          ours (h_count, v_count, confidence) and the log-Gabor bank
  vlm/results/rect150_P1_predictions_wide.csv  GPT-6.1-sol P1 (<model>__v = vertical yarns = our h)
  tables/cascade_150_decisions.csv          frozen gate (escalate_gate, eta = 'rho'), md5 57d49a4d...

Per scene: error = mean APE of the two counts against the human count; "fail5" = a count off by > 5 %
("fail10" likewise).  Targets: ours, GPT-6.1-sol P1, and "any method within 5 %" (fails only when none of ours,
log-Gabor and GPT-6.1-sol P1 has both counts within 5 %; error = the best of the three), so that ours' confidence
is not judged only against ours' own error.  Predictors (higher = better, so a useful one has rho < 0 and AUROC > 0.5):
-BRISQUE, -NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR (protocol 'work' = the counter's working image, primary;
'fix1024' = every crop at 1024^2, secondary), ours' confidence, eta, the frozen gate (1 = kept offline).
Statistics as in iqa_generic.py: Spearman and AUROC with a scene bootstrap (B = 5000, seed 20261011),
paired differences to ours' confidence on the same resamples, failures caught when each score escalates
as many scenes as the frozen gate; overall and by resolution class (full >= 1024 px / quarter).

Outputs (--out, default paper/tables): iqa150.csv, iqa150_summary.json; prints the numbers for the
draft sentence in the 'Confidence as a task-referenced quality score' paragraph.

    <venv>/bin/python tools/iqa150.py --labels '<run>/work/merged/scoring_headline/*.csv' --out <scratch>/iqa150
    <venv>/bin/python tools/iqa150.py --scene-gt <scene_gt.csv> --out <scratch>     # plumbing checks
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # importing score150 must not write __pycache__ next to the frozen scorer

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iqa_generic as G  # noqa: E402  (same statistics as the legacy analysis)

PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TAB = os.path.join(PAPER, "tables")
P_SCORES = os.path.join(TAB, "iqa150_scores.csv")
P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_VLM = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")
P_GATE = os.path.join(TAB, "cascade_150_decisions.csv")
GATE_MD5 = "57d49a4d4304bc395ed608894009f683"
ANY = "any method (ours, log-Gabor, GPT P1)"


def scene_gt_from_labels(paths, manifest):
    """Identical to quality150.py --unit scene (reuses score150 + the tool's propagation)."""
    import score150 as S
    rec = S.load_labels(paths)
    gt = S.gt_table(rec)
    gt.index = gt.index.astype(int)
    PS = S.scene_module(manifest)
    _, gt_scene = PS.propagate(gt, PS.load_scenes(json.load(open(manifest))))
    gt = gt_scene.drop(columns=["label_src_kind"], errors="ignore")
    gt.index = gt.index.astype(int)
    print(f"labels: {len(rec)} records from {len(paths)} file(s) -> {len(gt)} scenes")
    return gt


def predictions():
    o = pd.read_csv(P_OURS).set_index("id")
    P = {"ours": pd.DataFrame({"h": o.h_count, "v": o.v_count}),
         "log-Gabor": pd.DataFrame({"h": o.h_gabor, "v": o.v_gabor})}
    w = pd.read_csv(P_VLM).set_index("image_id")
    P["GPT-6.1-sol P1"] = pd.DataFrame({"h": w["gpt-6.1-sol__v"], "v": w["gpt-6.1-sol__h"]})
    return P, o.confidence


def signals(scores=P_SCORES):
    if hashlib.md5(open(P_GATE, "rb").read()).hexdigest() != GATE_MD5:
        raise SystemExit("tables/cascade_150_decisions.csv is not the frozen file (md5 57d49a4d...)")
    sc = pd.read_csv(scores).set_index("id")
    gate = pd.read_csv(P_GATE).set_index("image")
    _, conf = predictions()
    X = sc.copy()
    X["conf"] = conf.reindex(X.index).fillna(0.0)
    X["eta"] = gate.rho.reindex(X.index)
    X["gate_escalate"] = gate.escalate_gate.reindex(X.index).astype(str).str.lower().isin(["true", "1"])
    return X


def errors(gt, P):
    """Per-scene error of each method, plus 'any method within 5 %' (G.any_target: best method's mean APE; fails
    only if no method has both counts within 5 %).  A method without a valid count on a scene is skipped there."""
    E = {}
    for tname, pr in P.items():
        J = gt[["h", "v"]].join(pr, rsuffix="_p", how="inner")
        J = J[np.isfinite(J.h_p) & np.isfinite(J.v_p) & (J.h > 0) & (J.v > 0)]
        ah, av = 100 * (J.h_p - J.h).abs() / J.h, 100 * (J.v_p - J.v).abs() / J.v
        E[tname] = pd.DataFrame({"ape": 0.5 * (ah + av), "fail5": np.fmax(ah, av) > 5,
                                 "fail10": np.fmax(ah, av) > 10}).rename_axis("id")
    E[ANY] = G.any_target(E, list(P))
    return E


def analyse(gt, X, P, rng, unit_label):
    rows = []
    preds = G.predictors()
    for tname, Et in errors(gt, P).items():
        J = Et.join(X, how="inner")
        J.index.name = "id"
        for cls, sel in (("all", np.ones(len(J), bool)), ("full-res (>= 1024 px)", (J.S >= 1024).to_numpy()),
                         ("quarter-res (< 1024 px)", (J.S < 1024).to_numpy())):
            K = J[sel].reset_index()
            if len(K) < 8:
                continue
            Wb, ncl = G.cluster_weights(K["id"].to_numpy(), rng)     # one scene per row: plain scene bootstrap
            err, f5, f10 = K.ape.to_numpy(float), K.fail5.to_numpy(bool), K.fail10.to_numpy(bool)
            qs = {name: G.pred_q(K, kind, prot) for name, kind, prot in preds}
            boot = G.boot_all(qs, err, f5, Wb)
            k_gate = int(K.gate_escalate.sum())
            for name, kind, prot in preds:
                q = qs[name]
                primary = cls == "all" and prot in (None, "work")
                r = dict(unit=unit_label, resolution=cls, target_method=tname, predictor=name, protocol=prot or "",
                         status="pre-specified, " + ("primary" if primary else "secondary"),
                         n=len(K), n_fail5=int(f5.sum()), n_fail10=int(f10.sum()), spearman=G.spearman(q, err),
                         auroc_fail5=G.auroc(-q, f5), auroc_fail10=G.auroc(-q, f10))
                r["spearman_lo"], r["spearman_hi"] = G.ci(boot[name][0])
                r["auroc_fail5_lo"], r["auroc_fail5_hi"] = G.ci(boot[name][1])
                if name != "ours confidence":
                    base = boot["ours confidence"]
                    r["d_spearman_vs_conf"] = r["spearman"] - G.spearman(qs["ours confidence"], err)
                    r["d_spearman_vs_conf_lo"], r["d_spearman_vs_conf_hi"] = G.ci(boot[name][0] - base[0])
                    r["d_auroc5_vs_conf"] = r["auroc_fail5"] - G.auroc(-qs["ours confidence"], f5)
                    r["d_auroc5_vs_conf_lo"], r["d_auroc5_vs_conf_hi"] = G.ci(boot[name][1] - base[1])
                if kind == "gate":
                    esc = K.gate_escalate.to_numpy(bool)
                else:
                    esc = np.zeros(len(K), bool)
                    esc[np.argsort(q, kind="stable")[:k_gate]] = True
                r.update(k_escalated=k_gate, fail5_caught=int((esc & f5).sum()), fail10_caught=int((esc & f10).sum()))
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--scene-gt", default=None, help="CSV id,h,v per scene representative (instead of --labels)")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "annotation_tool", "data", "manifest.json"))
    ap.add_argument("--out", default=TAB)
    ap.add_argument("--scores", default=P_SCORES, help="per-crop IQA table (default tables/iqa150_scores.csv)")
    a = ap.parse_args()
    if a.scene_gt:
        gt = pd.read_csv(a.scene_gt).set_index("id")
        gt.index = gt.index.astype(int)
    else:
        paths = [p for arg in a.labels for s in arg.split(",") for p in sorted(glob.glob(s.strip()))]
        if not paths:
            raise SystemExit("give --labels (merged scoring CSVs) or --scene-gt")
        gt = scene_gt_from_labels(paths, a.manifest)
    gt = gt[(gt.h > 0) & (gt.v > 0)]
    X = signals(a.scores)
    reps = set(X.index[X.is_scene_rep.astype(bool)])
    if not set(gt.index) <= reps:
        raise SystemExit(f"scene table ids are not all scene representatives: {sorted(set(gt.index) - reps)[:10]}")
    P, _ = predictions()
    rng = np.random.default_rng(G.SEED)
    R = analyse(gt, X, P, rng, f"{len(gt)} scenes (rep photo)")
    os.makedirs(a.out, exist_ok=True)
    R.round(4).to_csv(os.path.join(a.out, "iqa150.csv"), index=False)

    def pick(tm, pred, cls="all"):
        r = R[(R.target_method == tm) & (R.predictor == pred) & (R.resolution == cls)]
        if not len(r):
            return None
        r = r.iloc[0]
        return {k: (round(float(r[k]), 4) if pd.notna(r.get(k)) else None)
                for k in ("n", "n_fail5", "spearman", "spearman_lo", "spearman_hi", "auroc_fail5", "auroc_fail5_lo",
                          "auroc_fail5_hi", "d_spearman_vs_conf_lo", "d_spearman_vs_conf_hi", "d_auroc5_vs_conf_lo",
                          "d_auroc5_vs_conf_hi", "k_escalated", "fail5_caught")}
    summ = {"generated_by": "paper/tools/iqa150.py", "n_scenes": int(len(gt)),
            "labels": a.scene_gt or a.labels, "results": {tm: {p[0]: pick(tm, p[0]) for p in G.predictors()}
                                                          for tm in list(P) + [ANY]},
            "code_sha256": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                            for f in ("iqa150.py", "iqa_generic.py")},
            "scores_sha256": hashlib.sha256(open(a.scores, "rb").read()).hexdigest()}
    json.dump(summ, open(os.path.join(a.out, "iqa150_summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    x = R[R.resolution == "all"][["target_method", "predictor", "n", "n_fail5", "spearman", "spearman_lo",
                                  "spearman_hi", "auroc_fail5", "auroc_fail5_lo", "auroc_fail5_hi",
                                  "d_spearman_vs_conf_lo", "d_spearman_vs_conf_hi", "k_escalated", "fail5_caught"]]
    print(x.round(2).to_string(index=False))
    f = lambda d, k: "--" if d is None or d.get(k) is None else f"{d[k]:.2f}"            # noqa: E731
    def one(tm, p):
        d = summ["results"][tm][p]
        return (f"{p}: rho {f(d, 'spearman')} [{f(d, 'spearman_lo')}, {f(d, 'spearman_hi')}], AUROC(>5%) "
                f"{f(d, 'auroc_fail5')} [{f(d, 'auroc_fail5_lo')}, {f(d, 'auroc_fail5_hi')}]"
                + ("" if p == "ours confidence" else f", paired diff. to confidence: rho [{f(d, 'd_spearman_vs_conf_lo')}, "
                   f"{f(d, 'd_spearman_vs_conf_hi')}], AUROC [{f(d, 'd_auroc5_vs_conf_lo')}, {f(d, 'd_auroc5_vs_conf_hi')}]"))
    print("\nPRE-SPECIFIED PRIMARY (76 scenes, protocol 'work'; never call a difference 'matches' without its CI):")
    for tm in list(P) + [ANY]:
        n = summ["results"][tm]["ours confidence"]
        print(f"  target {tm} (n = {int(n['n'])}, failures {int(n['n_fail5'])}):")
        for p in ("ours confidence", "resolvability eta", "cascade gate (frozen; 1 = kept offline)", "BRISQUE [work]",
                  "NIQE [work]", "MUSIQ [work]", "CLIP-IQA [work]", "TOPIQ-NR [work]"):
            print("    " + one(tm, p))
    print("wrote", os.path.join(a.out, "iqa150.csv"))


if __name__ == "__main__":
    main()

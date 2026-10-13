"""octsel_v1.py -- learned OCTAVE SELECTOR for the frozen spectral counter (octsel-v1).  POST HOC.

Per file and axis it predicts k in {-1, 0, +1} and outputs count = frozen counter's count * 2^k.
Inputs are LABEL-FREE only (features_octsel_150.csv from octsel_features.py: stage log-ratios, the counter's
alternative count, whitened spectral evidence g at c, 2c, c/2, confidence, px per yarn, log2(GPT/counter)).

Training target (FL_SA only): k* = clip(round(log2(human / counter)), -1, 1) on the 76 scene representatives x
2 axes (label h_tpi <-> prediction h_count, as score150.py / eval_octave_v1.py).
Model: multinomial logistic regression, standardised features, L2 on the weights (objective
0.5 ||W||^2 + C * sum_i w_i CE_i, w_i class-balanced on the training rows), intercepts lightly penalised (1e-4).
C from the grid C_GRID chosen by NESTED grouped CV inside the training folds only (inner folds = the remaining
fabric-grouped folds; criterion: class-balanced mean log-loss of the inner out-of-fold predictions; ties -> the
smaller C).
Decision (PRIMARY, rule 'cost'): undo the class balancing with the training class frequencies (p_j ~ p_bal_j * prior_j),
then choose the k with the smallest expected APE, COST[k, j] = 100 |2^(k-j) - 1| (choose k when the true octave is j).
SECONDARY (rule 'argmax', the design as first specified): argmax of the class-balanced probabilities.  The cost rule
was adopted AFTER the argmax rule scored 48.3% out of fold against FL_SA (raw counter 41.7%); both are frozen and
reported (predictions_octsel_oof_FLSA_150.csv = cost; predictions_octsel_argmax_oof_FLSA_150.csv = argmax).
Probability columns p-1/p+0/p+1 in the prediction files are the prior-corrected probabilities.
Folds: handover/results/labelday_diag_FLSA_20261012/cnn_cv_FLSA/fabric_grouped/folds.csv (5 folds, every one of
the 44 fabric groups of paper/tables/fabric_groups_v2.csv inside one fold; the same folds as the fabric-grouped
CNN and ens-v1).  Out-of-fold predictions for ALL 150 files: each file uses the model of its scene's fold
(trained without that fabric).  The final model (for new, unlabelled files) is fitted on all 76 scenes with C
chosen by the same grouped 5-fold CV.

    .tools/venv/bin/python -I -B methods/octave_learned/octsel_v1.py            # train, OOF, final model, eval
    .tools/venv/bin/python -I -B methods/octave_learned/octsel_v1.py --predict <features.csv> --out <preds.csv>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIAG = os.path.join(ROOT, "handover", "results", "labelday_diag_FLSA_20261012")
P_FEAT = os.path.join(HERE, "features_octsel_150.csv")
P_FOLDS = os.path.join(DIAG, "cnn_cv_FLSA", "fabric_grouped", "folds.csv")
P_LABELS = os.path.join(DIAG, "payment_audit", "labels_per_scene_FL_SA.csv")
P_GROUPS = os.path.join(ROOT, "paper", "tables", "fabric_groups_v2.csv")
P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_ENS = os.path.join(ROOT, "methods", "ensemble", "predictions_ens_v1_150.csv")
P_VLM = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")
P_EVAL = os.path.join(ROOT, "methods", "octave", "eval_octave_v1.py")
FEATURES = ["lr_lattice", "lr_fft", "lr_profile", "lr_tiles", "lr_gabor", "lr_alt",
            "g_c", "g_2c", "g_half", "conf", "lpx", "lr_gpt"]
CLASSES = np.array([-1, 0, 1])
COST = np.array([[100.0 * abs(2.0 ** (k - j) - 1) for j in (-1, 0, 1)] for k in (-1, 0, 1)])   # APE, choose k | true j
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
G_CLIP = (-5.0, 25.0)      # g features clipped (outliers); log-ratio features clipped to +-3 octaves
N_BOOT = 10000
SEED = 20261012


# ----------------------------------------------------------------------------------------------- model
def design(df):
    X = df[FEATURES].astype(float).to_numpy().copy()
    for j, f in enumerate(FEATURES):
        if f.startswith("g_"):
            X[:, j] = np.clip(X[:, j], *G_CLIP)
        elif f.startswith("lr_"):
            X[:, j] = np.clip(X[:, j], -3, 3)
    X[~np.isfinite(X)] = 0.0
    return X


def class_weights(y):
    n, K = len(y), len(CLASSES)
    w = np.zeros(n)
    for c in CLASSES:
        m = y == c
        if m.any():
            w[m] = n / (K * m.sum())
    return w


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit(X, y, C):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    K = len(CLASSES)
    Y = (y[:, None] == CLASSES[None, :]).astype(float)
    w = class_weights(y)

    def f(theta):
        W = theta[:d * K].reshape(d, K)
        b = theta[d * K:]
        P = _softmax(Xs @ W + b)
        ce = -np.sum(Y * np.log(P + 1e-300), axis=1)
        loss = 0.5 * np.sum(W ** 2) + 0.5e-4 * np.sum(b ** 2) + C * np.sum(w * ce)
        G = C * (w[:, None] * (P - Y))
        gW = Xs.T @ G + W
        gb = G.sum(axis=0) + 1e-4 * b
        return loss, np.r_[gW.ravel(), gb]

    r = minimize(f, np.zeros(d * K + K), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-8})
    return {"mu": mu, "sd": sd, "W": r.x[:d * K].reshape(d, K), "b": r.x[d * K:], "C": C, "converged": bool(r.success)}


def predict_proba(m, X):
    Xs = (X - np.asarray(m["mu"])) / np.asarray(m["sd"])
    return _softmax(Xs @ np.asarray(m["W"]) + np.asarray(m["b"]))


def balanced_logloss(P, y):
    w = class_weights(y)
    idx = np.searchsorted(CLASSES, y)
    return float(np.sum(w * -np.log(P[np.arange(len(y)), idx] + 1e-12)) / np.sum(w))


def choose_C(X, y, fold):
    """grouped CV over the distinct values of `fold`: returns (best C, {C: balanced log-loss})."""
    res = {}
    for C in C_GRID:
        P = np.zeros((len(y), len(CLASSES)))
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            P[te] = predict_proba(fit(X[tr], y[tr], C), X[te])
        res[C] = balanced_logloss(P, y)
    best = min(C_GRID, key=lambda c: (round(res[c], 10), c))
    return best, res


# ----------------------------------------------------------------------------------------------- data
def load_training():
    feat = pd.read_csv(P_FEAT)
    folds = pd.read_csv(P_FOLDS)
    folds["scene_id"] = folds.scene.str.replace("scene", "").astype(int)
    lab = pd.read_csv(P_LABELS)
    lab["scene_id"] = lab.scene_id.astype(int)
    grp = pd.read_csv(P_GROUPS)[["scene_id", "fabric_group"]]
    ff = folds[["image", "fold", "scene_id", "is_rep", "rot_k"]].rename(columns={"image": "id"})
    feat = feat.merge(ff, on="id", how="left").merge(grp, on="scene_id", how="left")
    if feat.fold.isna().any():
        raise SystemExit("files without a fold")
    # every fabric group inside one fold
    if feat.groupby("fabric_group").fold.nunique().max() != 1:
        raise SystemExit("a fabric group spans several folds")
    tr = feat[feat.is_rep.astype(bool)].copy()
    hum = lab.set_index("scene_id")
    tr["human"] = [float(hum.loc[s, f"{a}_tpi"]) for s, a in zip(tr.scene_id, tr.axis)]
    tr = tr[np.isfinite(tr.human) & (tr.human > 0) & (tr["count"] > 0)]
    tr["k_star"] = np.clip(np.rint(np.log2(tr.human / tr["count"])), -1, 1).astype(int)
    return feat, tr.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- train / OOF
def priors(y):
    return np.array([(y == c).mean() for c in CLASSES])


def prior_correct(P, prior):
    """undo the class balancing: p_j proportional to p_bal_j * prior_j (training class frequencies)"""
    Q = P * np.asarray(prior)[None, :]
    return Q / Q.sum(axis=1, keepdims=True)


def decide(P_bal, prior, rule):
    """rule 'cost' (PRIMARY): k minimising the expected APE under the prior-corrected probabilities, with
    COST[k, j] = 100 |2^(k - j) - 1| (APE of choosing k when the true octave is j).
    rule 'argmax' (secondary, the design as first specified): argmax of the class-balanced probabilities."""
    if rule == "argmax":
        return CLASSES[np.argmax(P_bal, axis=1)]
    return CLASSES[np.argmin(prior_correct(P_bal, prior) @ COST.T, axis=1)]


def run_cv(feat, tr):
    X, y, fold = design(tr), tr.k_star.to_numpy(), tr.fold.to_numpy()
    Xall = design(feat)
    oof = np.full((len(feat), len(CLASSES)), np.nan)
    oof_prior = np.full((len(feat), len(CLASSES)), np.nan)
    per_fold = []
    for f in sorted(np.unique(fold)):
        trn = fold != f
        Cf, res = choose_C(X[trn], y[trn], fold[trn])        # nested: inner folds = the other training folds
        m = fit(X[trn], y[trn], Cf)
        sel = (feat.fold == f).to_numpy()
        oof[sel] = predict_proba(m, Xall[sel])
        oof_prior[sel] = priors(y[trn])
        per_fold.append({"fold": int(f), "C": Cf, "inner_balanced_logloss": {str(k): round(v, 5) for k, v in res.items()},
                         "train_priors": priors(y[trn]).round(4).tolist(),
                         "n_train_axes": int(trn.sum()), "n_test_files": int(sel.sum()), "converged": m["converged"]})
    Cfull, resfull = choose_C(X, y, fold)
    full = fit(X, y, Cfull)
    full["prior"] = priors(y)
    Pfull = predict_proba(full, Xall)
    return oof, oof_prior, per_fold, full, Pfull, resfull


def to_predictions(feat, P, prior, rule="cost"):
    prior = np.broadcast_to(np.asarray(prior, float), P.shape)
    Q = np.vstack([prior_correct(P[i:i + 1], prior[i]) for i in range(len(P))])
    k = np.array([decide(P[i:i + 1], prior[i], rule)[0] for i in range(len(P))])
    P = Q
    d = feat[["id", "axis", "count"]].copy()
    d["k"] = k
    d["pred"] = d["count"] * np.power(2.0, d.k)
    for j, c in enumerate(CLASSES):
        d[f"p{c:+d}"] = P[:, j]
    w = d.pivot(index="id", columns="axis")
    out = pd.DataFrame({"id": w.index.astype(int), "h_count": w[("pred", "h")].to_numpy(),
                        "v_count": w[("pred", "v")].to_numpy(), "k_h": w[("k", "h")].astype(int).to_numpy(),
                        "k_v": w[("k", "v")].astype(int).to_numpy()})
    for c in CLASSES:
        for ax in ("h", "v"):
            out[f"p{c:+d}_{ax}"] = w[(f"p{c:+d}", ax)].round(5).to_numpy()
    return out.sort_values("id").reset_index(drop=True), d


def model_json(m, resfull):
    d = len(FEATURES)
    W = np.asarray(m["W"])
    return {"name": "octsel-v1", "classes_k": CLASSES.tolist(), "features": FEATURES, "g_clip": list(G_CLIP),
            "lr_clip": [-3, 3], "C": m["C"], "C_grid": C_GRID,
            "cv_balanced_logloss_by_C": {str(k): round(v, 5) for k, v in resfull.items()},
            "mu": np.asarray(m["mu"]).tolist(), "sd": np.asarray(m["sd"]).tolist(),
            "W": W.tolist(), "b": np.asarray(m["b"]).tolist(),
            "contrast_vs_k0_standardised": {f"k={c:+d}": {FEATURES[j]: round(float(W[j, i] - W[j, 1]), 4) for j in range(d)}
                                            for i, c in enumerate(CLASSES) if c != 0},
            "prior": np.asarray(m["prior"]).tolist(), "cost_ape_choose_k_given_true_j": COST.tolist(),
            "decision": "p_bal = softmax((x - mu) / sd @ W + b); p = p_bal * prior / sum; PRIMARY k = argmin_k sum_j p_j "
                        "COST[k, j]; secondary k = argmax p_bal; count = frozen counter count * 2^k; features from "
                        "octsel_features.py (missing g -> 0, missing stage/GPT -> log-ratio 0; g clipped to g_clip, "
                        "log-ratios to lr_clip)"}


# ----------------------------------------------------------------------------------------------- evaluation
def load_eval():
    spec = importlib.util.spec_from_file_location("eval_octave_v1_ro", P_EVAL)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


def evaluate_all(oof_pred, full_pred, tr, outdir, argmax_pred=None):
    E = load_eval()
    reps = set(json.load(open(E.MANIFEST))["scenes"]["reps"])
    ref = E.load_reference(P_LABELS, reps).dropna()
    base = E.load_preds(E.P_OCT, P_OURS, P_VLM)
    ours, gpt = base["ours-frozen"], base["gpt-6.1-sol P1"]
    ens = pd.read_csv(P_ENS).set_index("id")[["h_count", "v_count"]].rename(columns={"h_count": "h", "v_count": "v"})
    oo = oof_pred.set_index("id")[["h_count", "v_count"]].rename(columns={"h_count": "h", "v_count": "v"})
    fu = full_pred.set_index("id")[["h_count", "v_count"]].rename(columns={"h_count": "h", "v_count": "v"})

    def geo(a, b):
        b2 = b.reindex(a.index)
        g = np.sqrt(a * b2)
        return g.where((b2 > 0) & np.isfinite(b2), a)        # partner missing -> octsel alone
    orc = ours.copy()
    for ax in ("h", "v"):
        r = ref[ax].reindex(orc.index)
        k = np.clip(np.rint(np.log2(r / orc[ax])), -1, 1)
        orc[ax] = np.where(np.isfinite(k), orc[ax] * np.power(2.0, k.fillna(0)), orc[ax])
    snap = ours.copy()      # no-training reference: the counter snapped to GPT's octave
    for ax in ("h", "v"):
        k = np.clip(np.rint(np.log2(gpt[ax].reindex(snap.index) / snap[ax])), -1, 1)
        snap[ax] = snap[ax] * np.power(2.0, k.fillna(0))
    am = argmax_pred.set_index("id")[["h_count", "v_count"]].rename(columns={"h_count": "h", "v_count": "v"})
    methods = {"octsel-v1 OOF": oo, "spectral counter (frozen)": ours, "GPT-6.1-sol P1": gpt, "ens-v1": ens,
               "geo(octsel OOF, GPT)": geo(oo, gpt), "geo(octsel OOF, ens-v1)": geo(oo, ens),
               "octsel-v1 all-data model (in-sample)": fu, "octsel argmax-balanced OOF (as first specified)": am,
               "counter snapped to GPT octave (no training)": snap, "oracle octave (labels; bound)": orc}
    sc = {m: E.scene_scores(ref, p) for m, p in methods.items()}
    n = len(ref)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    # fabric-clustered bootstrap (44 groups) as a secondary interval
    grp = pd.read_csv(P_GROUPS).set_index("scene_id").fabric_group.reindex(ref.index).to_numpy()
    ug = np.unique(grp)
    gidx = [np.where(grp == g)[0] for g in ug]
    rg = np.random.default_rng(SEED)
    gb = rg.integers(0, len(ug), size=(N_BOOT, len(ug)))

    def ci(x):
        x = np.asarray(x, float)
        b = x[idx].mean(axis=1)
        fb = np.array([np.concatenate([x[gidx[j]] for j in row]).mean() for row in gb])
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], \
               [float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))]
    res = {"n_scenes": n, "methods": {}, "paired_octsel_oof_minus": {}, "paired_geo_minus": {}}
    for m, s in sc.items():
        c1, c2 = ci(s.mape.to_numpy())
        b1, _ = ci(100 * s.both5.astype(float).to_numpy())
        res["methods"][m] = {"MAPE": float(s.mape.mean()), "MAPE_ci95_scene": c1, "MAPE_ci95_fabric": c2,
                             "both5_n": int(s.both5.sum()), "both5_pct": float(100 * s.both5.mean()), "both5_ci95_scene": b1,
                             "MdAPE_axis": float(np.median(np.r_[s.ape_h, s.ape_v]))}
    for m in ("spectral counter (frozen)", "GPT-6.1-sol P1", "ens-v1"):
        p = E.paired(sc["octsel-v1 OOF"], sc[m], N_BOOT, SEED)
        d = (sc["octsel-v1 OOF"].mape - sc[m].mape).to_numpy()
        p["mape_diff_ci95_fabric"] = ci(d)[1]
        res["paired_octsel_oof_minus"][m] = p
    for g, other in (("geo(octsel OOF, GPT)", "GPT-6.1-sol P1"), ("geo(octsel OOF, ens-v1)", "ens-v1"),
                     ("geo(octsel OOF, GPT)", "ens-v1")):
        res["paired_geo_minus"][f"{g} - {other}"] = E.paired(sc[g], sc[other], N_BOOT, SEED)
    # k accuracy (scene representatives, 152 axes) from the OOF file
    oofk = oof_pred.set_index("id")
    kh = [int(oofk.loc[i, f"k_{a}"]) for i, a in zip(tr.id, tr.axis)]
    t = tr.k_star.to_numpy()
    kh = np.array(kh)
    cm = pd.crosstab(pd.Series(t, name="k_star (FL_SA)"), pd.Series(kh, name="k_pred (OOF)")).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    rec = [float((kh[t == c] == c).mean()) if (t == c).any() else float("nan") for c in CLASSES]
    res["k"] = {"n_axes": int(len(t)), "accuracy": float((kh == t).mean()), "balanced_accuracy": float(np.nanmean(rec)),
                "recall_by_class": dict(zip([f"{c:+d}" for c in CLASSES], rec)),
                "always_k0_accuracy": float((t == 0).mean()), "confusion": cm.to_dict(),
                "class_counts": {f"{c:+d}": int((t == c).sum()) for c in CLASSES},
                "n_axes_changed": int((kh != 0).sum()), "changed_correct": int(((kh != 0) & (kh == t)).sum())}
    fullk = full_pred.set_index("id")
    khf = np.array([int(fullk.loc[i, f"k_{a}"]) for i, a in zip(tr.id, tr.axis)])
    res["k_full_in_sample_accuracy"] = float((khf == t).mean())
    amk = argmax_pred.set_index("id")
    kha = np.array([int(amk.loc[i, f"k_{a}"]) for i, a in zip(tr.id, tr.axis)])
    res["k_argmax_oof"] = {"accuracy": float((kha == t).mean()), "n_axes_changed": int((kha != 0).sum()),
                           "confusion": pd.crosstab(pd.Series(t, name="k_star"), pd.Series(kha, name="k_pred")).reindex(
                               index=CLASSES, columns=CLASSES, fill_value=0).to_dict()}
    per = pd.concat({m: s[["ape_h", "ape_v", "mape", "both5"]] for m, s in sc.items()}, axis=1)
    per.to_csv(os.path.join(outdir, "eval_FLSA_per_scene.csv"))
    return res, cm


def text_report(res, cm, per_fold, full):
    L = [f"octsel-v1 vs FL_SA (POST HOC; n = {res['n_scenes']} scene reps; scene bootstrap {N_BOOT}, seed {SEED}; "
         "[scene CI] {fabric CI})", ""]
    k = res["k"]
    L.append(f"k (152 axes, OOF): accuracy {100*k['accuracy']:.1f}% (always k=0: {100*k['always_k0_accuracy']:.1f}%), "
             f"balanced {100*k['balanced_accuracy']:.1f}%; recall by class {k['recall_by_class']}; "
             f"axes changed {k['n_axes_changed']}, correctly {k['changed_correct']}; class counts {k['class_counts']}")
    L.append("confusion (rows k*, cols predicted):")
    L.append(cm.to_string())
    L.append(f"all-data model, in-sample k accuracy {100*res['k_full_in_sample_accuracy']:.1f}%")
    L.append("")
    L.append(f"{'method':40s} {'MAPE':>6s} {'[scene CI]':>16s} {'{fabric CI}':>16s} {'both5':>6s}")
    for m, r in res["methods"].items():
        L.append(f"{m:40s} {r['MAPE']:6.1f} [{r['MAPE_ci95_scene'][0]:5.1f}, {r['MAPE_ci95_scene'][1]:5.1f}] "
                 f"{{{r['MAPE_ci95_fabric'][0]:5.1f}, {r['MAPE_ci95_fabric'][1]:5.1f}}} {r['both5_n']:3d}/{res['n_scenes']}")
    L.append("")
    for m, p in res["paired_octsel_oof_minus"].items():
        L.append(f"octsel OOF - {m}: dMAPE {p['mape_diff']:+.1f} [{p['mape_diff_ci95'][0]:+.1f}, {p['mape_diff_ci95'][1]:+.1f}] "
                 f"{{{p['mape_diff_ci95_fabric'][0]:+.1f}, {p['mape_diff_ci95_fabric'][1]:+.1f}}}, Wilcoxon p {p['wilcoxon_p']:.3g}, "
                 f"better/worse/tied {p['n_better']}/{p['n_worse']}/{p['n_tied']}; d both5 {p['both5_diff_pts']:+.1f} pts "
                 f"[{p['both5_diff_ci95_pts'][0]:+.1f}, {p['both5_diff_ci95_pts'][1]:+.1f}], McNemar p {p['both5_mcnemar_p']:.3g}")
    for m, p in res["paired_geo_minus"].items():
        L.append(f"{m}: dMAPE {p['mape_diff']:+.1f} [{p['mape_diff_ci95'][0]:+.1f}, {p['mape_diff_ci95'][1]:+.1f}], "
                 f"d both5 {p['both5_diff_pts']:+.1f} pts (reported, not tuned)")
    L.append("")
    L.append("C per outer fold (nested): " + ", ".join(f"fold {r['fold']}: {r['C']}" for r in per_fold)
             + f"; all-data model C = {full['C']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", default=None, help="features csv (octsel_features.py format) -> predictions")
    ap.add_argument("--model", default=os.path.join(HERE, "model_octsel_v1.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args(argv)
    if a.predict:
        m = json.load(open(a.model))
        feat = pd.read_csv(a.predict)
        pred, _ = to_predictions(feat, predict_proba(m, design(feat)), m["prior"], "cost")
        pred.to_csv(a.out or os.path.join(a.outdir, "predictions_octsel_new.csv"), index=False)
        print(f"wrote {len(pred)} files")
        return
    feat, tr = load_training()
    oof, oof_prior, per_fold, full, Pfull, resfull = run_cv(feat, tr)
    fold_of = feat.drop_duplicates("id").set_index("id").fold
    preds = {}
    for rule, tag in (("cost", ""), ("argmax", "_argmax")):
        oo, _ = to_predictions(feat, oof, oof_prior, rule)
        oo.insert(5, "fold", fold_of.reindex(oo.id).astype(int).to_numpy())
        fu, _ = to_predictions(feat, Pfull, full["prior"], rule)
        preds[rule] = (oo, fu)
        os.makedirs(a.outdir, exist_ok=True)
        # full precision, so an unchanged axis equals the frozen counter's count exactly (ties in paired tests)
        oo.to_csv(os.path.join(a.outdir, f"predictions_octsel{tag}_oof_FLSA_150.csv"), index=False)
        fu.to_csv(os.path.join(a.outdir, f"predictions_octsel{tag}_full_150.csv"), index=False)
    oof_pred, full_pred = preds["cost"]
    # exploratory (reported, not tuned): per file and axis geometric mean of octsel-v1 OOF with GPT-6.1-sol P1 and
    # with ens-v1 (partner missing or non-positive -> octsel-v1 alone)
    w = pd.read_csv(P_VLM).set_index("image_id")
    partners = {"gpt": pd.DataFrame({"h_count": pd.to_numeric(w["gpt-6.1-sol__v"], errors="coerce"),
                                     "v_count": pd.to_numeric(w["gpt-6.1-sol__h"], errors="coerce")}),
                "ens_v1": pd.read_csv(P_ENS).set_index("id")[["h_count", "v_count"]]}
    for nm, q in partners.items():
        g = oof_pred[["id", "h_count", "v_count"]].copy()
        for c in ("h_count", "v_count"):
            pv = q[c].reindex(g.id).to_numpy(dtype=float)
            ok = np.isfinite(pv) & (pv > 0)
            g[c] = np.where(ok, np.sqrt(g[c].to_numpy() * np.where(ok, pv, 1.0)), g[c].to_numpy())
        g.to_csv(os.path.join(a.outdir, f"predictions_octsel_geo_{nm}_oof_FLSA_150.csv"), index=False)
    mj = model_json(full, resfull)
    mj["trained_on"] = "FL_SA, 76 scene representatives x 2 axes (labels_per_scene_FL_SA.csv)"
    mj["training_class_counts"] = {f"{c:+d}": int((tr.k_star == c).sum()) for c in CLASSES}
    with open(os.path.join(a.outdir, "model_octsel_v1.json"), "w") as f:
        json.dump(mj, f, indent=1)
    tr[["id", "axis", "scene_id", "fabric_group", "fold", "count", "human", "k_star"] + FEATURES].to_csv(
        os.path.join(a.outdir, "training_table_FLSA.csv"), index=False, float_format="%.6f")
    res, cm = evaluate_all(oof_pred, full_pred, tr, a.outdir, preds["argmax"][0])
    res["per_fold"] = per_fold
    res["final_model_C"] = full["C"]
    res["final_model_coefficients_vs_k0"] = mj["contrast_vs_k0_standardised"]
    with open(os.path.join(a.outdir, "cv_results_FLSA.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    txt = text_report(res, cm, per_fold, full)
    txt += "\n\nfinal model, standardised coefficients (class k minus class 0):\n" + \
        pd.DataFrame(mj["contrast_vs_k0_standardised"]).round(3).to_string()
    with open(os.path.join(a.outdir, "cv_results_FLSA.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

"""paired_ref.py -- paired comparisons against A2 alone, the textile-trained reference annotator whose counts were the
pre-specified blind test set (opened only after every method was frozen). A1 (the non-expert) is treated as one more
"method" (a human baseline). Main Secs. 5.1 and 5.2; supplement Sec. S4 (patch-clustered CIs).

Read-only on the repository. Writes only paired_ref.json in --out.

    python paper/tools/reference/paired_ref.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR] [--a2-run DIR] [--eval-dir DIR]

Scheme (frozen paper/tools/label_day.py, imported, never edited):
  per_scene_ape(): per-scene APE = 0.5 * (APE_h + APE_v), scenes both methods answered (inner join, dropna);
  paired(): mean difference a - b, 10,000 scene resamples, numpy default_rng(0), percentile 95% CI, Wilcoxon p.
Secondary (item 6 of the plan for the reference annotator, not in this repository, SHA-256 9ae1d9ca..., see
CHECKSUMS.md, asks for fabric-level CIs on headline numbers): fabric-clustered bootstrap over the 44 groups (patches) of
paper/tables/fabric_groups_v2.csv (10,000, default_rng(20261012)).
Symmetric error: per scene, mean over the two counts of |log2(prediction / reference)|; same paired scheme.
Half / double band (paper Sec. 5.2): | |log2(pred / A2)| - 1 | <= 0.3; double if log2 > 0.
Octave upper bound (the paper's octave oracle): per axis k = round(log2(ref / pred)), corrected = pred * 2^k; MAPE over
scenes before -> after.

Labels: the reference is A2's per-scene counts (--a2), unrounded; label_day.py rounds them to 2 decimals (the tool
export): the same comparisons on that rounded reference are reported as a check and reproduce label_day's stored paired
rows. The validation block scores the mean of both annotators as label_day.py stores it in its per-scene table (each
annotator's count to 2 decimals, their mean and the predictions to 3 decimals).
Optional checks (inputs that are not part of the repository; skipped when not given):
  --a2-run DIR    a label_day.py single-annotator run on A2's export (folder holding work/): the predictions here equal
                  the ones it scored
  --eval-dir DIR  methods/octave/eval_octave_v1.py outputs against A2 (cnn_fabric.json, ens_v1.json): per-scene APE check
POST HOC: everything here except where 'status' says otherwise (see STATUS below).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
ap.add_argument("--a2-run", default=None, help="optional: label_day.py single-annotator run on A2's export")
ap.add_argument("--eval-dir", default=None, help="optional: eval_octave_v1.py outputs against A2")
args = ap.parse_args()
OUT = args.out
TOOLS = f"{ROOT}/paper/tools"


def sha(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


FREEZE = {e["path"]: e for e in json.load(open(f"{ROOT}/tools/freeze_checksums.json"))}
FROZEN_BEFORE = {"label_day.py_sha256": sha(f"{TOOLS}/label_day.py"), "score150.py_md5": sha(f"{TOOLS}/score150.py", "md5")}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.dont_write_bytecode = True
LD = load_module("label_day_frozen", f"{TOOLS}/label_day.py")
S = LD.load_score150()
paired, per_scene_ape = LD.paired, LD.per_scene_ape


# ------------------------------------------------------------------ reference and predictions
def counts(path):
    L = pd.read_csv(path).set_index("scene_id")
    L.index = L.index.astype(int)
    return pd.DataFrame({"h": L.h_tpi.astype(float), "v": L.v_tpi.astype(float)})


ORDER = sorted(counts(args.a2).index)   # label_day's gt order (ascending scene id; it fixes the bootstrap)


def labels(path):
    return counts(path).loc[ORDER]


A2 = labels(args.a2)
A1 = labels(args.a1)
A2_RUN = A2.round(2)   # label_day's (2-decimal) A2 reference: the tool export stores <ax>_tpi to 2 decimals

w = pd.read_csv(f"{ROOT}/vlm/results/rect150_P1_predictions_wide.csv").set_index("image_id")


def vlm(key):   # score150 axis convention: <key>__v = vertical yarns = our h
    return pd.DataFrame({"h": pd.to_numeric(w[f"{key}__v"], errors="coerce"),
                         "v": pd.to_numeric(w[f"{key}__h"], errors="coerce")}).reindex(ORDER)


def hv(path):
    d = pd.read_csv(f"{ROOT}/{path}").set_index("id")
    return pd.DataFrame({"h": d.h_count.astype(float), "v": d.v_count.astype(float)}).reindex(ORDER)


molmo = S.load_extra_wide([f"{ROOT}/ft_results/results/rect150_predictions_wide.csv::molmo2_4b_synthft@P1"])
P = {
    "GPT-6.1-sol P1": vlm("gpt-6.1-sol"),
    "A1 counts": A1,
    "Leave-one-out mean": S.loo_mean(A2),
    "Molmo2-4B + LoRA P1": molmo["molmo2_4b_synthft@P1"][["h", "v"]].astype(float).reindex(ORDER),
    "CNN trained on A1 (fabric-grouped)": hv("methods/ensemble/cnn_fabric_oof_FLSA_150.csv"),
    "CNN-VLM ensemble (ens-v1)": hv("methods/ensemble/predictions_ens_v1_150.csv"),
    "Gemini 3.8 Flash P1": vlm("gemini-3.8-flash"),
    "Qwen3.8-Max-Prime P1": vlm("qwen3.8-max-prime"),
    "PG-Lattice": hv("methods/predictions_ours_150.csv"),
}
SRC = {
    "GPT-6.1-sol P1": "vlm/results/rect150_P1_predictions_wide.csv gpt-6.1-sol (h=__v, v=__h)",
    "A1 counts": "A1's per-scene counts (--a1)",
    "Leave-one-out mean": "score150.loo_mean() of A2's counts",
    "Molmo2-4B + LoRA P1": "ft_results/results/rect150_predictions_wide.csv molmo2_4b_synthft@P1 (score150.load_extra_wide)",
    "CNN trained on A1 (fabric-grouped)": "methods/ensemble/cnn_fabric_oof_FLSA_150.csv",
    "CNN-VLM ensemble (ens-v1)": "methods/ensemble/predictions_ens_v1_150.csv",
    "Gemini 3.8 Flash P1": "vlm/results/rect150_P1_predictions_wide.csv gemini-3.8-flash",
    "Qwen3.8-Max-Prime P1": "vlm/results/rect150_P1_predictions_wide.csv qwen3.8-max-prime",
    "PG-Lattice": "methods/predictions_ours_150.csv (h_count, v_count)",
}
RUN_KEY = {"GPT-6.1-sol P1": "GPT-6.1-sol, P1", "Leave-one-out mean": "Leave-one-out mean (reference)",
           "Molmo2-4B + LoRA P1": "molmo2_4b_synthft@P1", "Gemini 3.8 Flash P1": "Gemini 3.8 Flash, P1",
           "Qwen3.8-Max-Prime P1": "Qwen3.8-Max-Prime, P1", "PG-Lattice": "Ours (lattice + fusion)"}

# ------------------------------------------------------------------ checks on the inputs
chk = {"frozen_hashes_before": FROZEN_BEFORE,
       "frozen_expected": {"label_day.py_sha256_prefix": FREEZE["paper/tools/label_day.py"]["sha256_released"][:8],
                           "score150.py_md5_prefix": FREEZE["paper/tools/score150.py"]["md5_original"][:8]},
       "A2_labels_vs_labelday_run_ref_maxabsdiff": float((A2 - A2_RUN).abs().to_numpy().max()),
       "n_scenes": len(ORDER)}
ps_ng = None
if args.a2_run:
    ps_ng = pd.read_csv(f"{args.a2_run}/work/stage/main150_per_scene.csv")
    ps_ng["scene_id"] = ps_ng.scene_id.astype(int)
    assert list(ps_ng[ps_ng.method == "Ours (lattice + fusion)"].scene_id) == ORDER
for k, rk in RUN_KEY.items():   # predictions = the ones label_day scored in the A2-alone run
    if ps_ng is not None:
        g = ps_ng[ps_ng.method == rk].set_index("scene_id")
        p = P[k].loc[g.index] if k != "Leave-one-out mean" else S.loo_mean(A2_RUN).loc[g.index]
        chk[f"pred_vs_run_{k}_maxabsdiff"] = float(np.nanmax(np.abs(np.c_[p.h - g.h_p, p.v - g.v_p])))
    chk[f"pred_answered_{k}"] = int(P[k].dropna().shape[0])
if args.eval_dir:
    for k, f in (("CNN trained on A1 (fabric-grouped)", "cnn_fabric"), ("CNN-VLM ensemble (ens-v1)", "ens_v1")):
        ev = json.load(open(f"{args.eval_dir}/{f}.json"))["blocks"][0]["per_scene"]["octave-v1"]
        ev = pd.DataFrame(ev).set_index("scene_id")
        mine = per_scene_ape(A2, P[k]).ape.loc[ev.index]
        chk[f"scene_ape_vs_eval_A2_{f}_maxabsdiff"] = float((mine - ev.mape).abs().max())
for k, p in P.items():
    chk[f"nonpositive_predictions_{k}"] = int(((p.h <= 0) | (p.v <= 0)).sum())

# ------------------------------------------------------------------ helpers
FG = pd.read_csv(f"{ROOT}/paper/tables/fabric_groups_v2.csv").set_index("scene_id")["fabric_group"]


def boot_clustered(cl, stat, n_boot=10000, seed=20261012):   # fabric-clustered bootstrap
    cl = np.asarray(cl)
    ug = np.unique(cl)
    members = [np.where(cl == u)[0] for u in ug]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = stat(np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], int(len(ug))


def scene_log(gt, pr):
    j = gt[["h", "v"]].join(pr[["h", "v"]], rsuffix="_p", how="inner").dropna()
    lh, lv = np.abs(np.log2(j.h_p / j.h)), np.abs(np.log2(j.v_p / j.v))
    return 0.5 * (lh + lv)


def paired_log(gt, pa, pb, n_boot=10000, seed=0):
    """label_day paired() on the per-scene symmetric error mean |log2(pred / ref)| instead of APE."""
    ea, eb = scene_log(gt, pa), scene_log(gt, pb)
    c = ea.index.intersection(eb.index)
    d = (ea[c] - eb[c]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    out = dict(n=int(len(c)), log_a=float(ea[c].mean()), log_b=float(eb[c].mean()), delta=float(d.mean()),
               lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_wilcoxon=p,
               a_closer=int((d < 0).sum()), b_closer=int((d > 0).sum()))
    ci, ng = boot_clustered(FG.reindex(c).to_numpy(), lambda ix: d[ix].mean())
    out.update(fabric_ci=ci, n_fabric_groups=ng)
    return out


def full_paired(gt, a, b):
    r = paired(gt, P[a], P[b])
    ea, eb = per_scene_ape(gt, P[a]).ape, per_scene_ape(gt, P[b]).ape
    c = ea.index.intersection(eb.index)
    d = (ea[c] - eb[c]).to_numpy()
    r["ties"] = int((d == 0).sum())
    ci, ng = boot_clustered(FG.reindex(c).to_numpy(), lambda ix: d[ix].mean())
    r.update(fabric_ci=ci, n_fabric_groups=ng)
    return r


def rnd(d, k=3):
    if isinstance(d, dict):
        return {kk: rnd(v, k) for kk, v in d.items()}
    if isinstance(d, list):
        return [rnd(v, k) for v in d]
    if isinstance(d, float):
        return round(d, k)
    return d


# ------------------------------------------------------------------ status of each comparison
STATUS = {
    "A1 counts": "comparison pre-specified (plan item 3 / addendum rule 4: method-vs-A2 error against "
                 "A1-vs-A2 error); no test was specified, so this paired() test is post hoc",
    "Leave-one-out mean": "post hoc",
    "Molmo2-4B + LoRA P1": "post hoc",
    "CNN trained on A1 (fabric-grouped)": "pre-specified secondary of PREREG_ens_v1.json (CNN alone vs GPT, eval_octave_v1.py, "
                                          "seed 20261012); this paired() recomputation (default_rng(0)) is post hoc",
    "CNN-VLM ensemble (ens-v1)": "PRE-SPECIFIED PRIMARY of PREREG_ens_v1.json (ens-v1 minus GPT vs A2, eval_octave_v1.py, "
                                 "seed 20261012): -1.75 [-4.15, +0.66], FAIL; this paired() recomputation is post hoc",
    "Gemini 3.8 Flash P1": "post hoc",
    "Qwen3.8-Max-Prime P1": "post hoc (paired() scores the 72 scenes both answered)",
    "PG-Lattice": "label_day's frozen paired row 'Ours vs GPT', run on A2 alone (single-annotator run, plan item 3); "
                  "sign flipped here",
}
EXISTING = {
    "GPT-6.1-sol P1 minus A1 counts": {
        "value": "-10.2 [-17.9, -2.7] (method 27.1, human 37.3)", "scheme": "loop bootstrap 10,000, default_rng(20261012)",
        "source": "paper/tools/reference/human_floor_paired.py (human_floor_paired.json)"},
    "GPT-6.1-sol P1 minus Leave-one-out mean": {
        "value": "LOO minus GPT +6.056 [+0.563, +11.659], Wilcoxon p 0.035, fabric CI [-0.98, +13.01]",
        "scheme": "label_day paired() on label_day's 2-decimal A2 reference",
        "source": "label_day paired() on the 2-decimal A2 reference (checks.run_reference_variant_reproduces_stored)"},
    "GPT-6.1-sol P1 minus CNN trained on A1 (fabric-grouped)": {
        "value": "CNN minus GPT +0.96 [-4.06, +6.29], Wilcoxon p 0.967", "scheme": "eval_octave_v1.py, default_rng(20261012)",
        "source": "methods/octave/eval_octave_v1.py against A2 (cnn_fabric.json)"},
    "GPT-6.1-sol P1 minus CNN-VLM ensemble (ens-v1)": {
        "value": "ens-v1 minus GPT -1.75 [-4.15, +0.66], Wilcoxon p 0.088 (pre-specified primary: FAIL)",
        "scheme": "eval_octave_v1.py, default_rng(20261012)", "source": "methods/octave/eval_octave_v1.py against A2 (ens_v1.json)"},
    "GPT-6.1-sol P1 minus PG-Lattice": {
        "value": "Ours minus GPT +7.685 [+1.884, +13.622], Wilcoxon p 0.054", "scheme": "label_day paired() (frozen run)",
        "source": "paper/tools/label_day.py single-annotator run on A2's export (work/stage/main150_paired.csv)"},
    "PG-Lattice minus Leave-one-out mean": {
        "value": "+1.629 [-6.093, +9.386], Wilcoxon p 0.709", "scheme": "label_day paired() (frozen run)",
        "source": "paper/tools/label_day.py single-annotator run on A2's export (work/stage/main150_paired.csv)"},
}

# ------------------------------------------------------------------ 1. paired MAPE differences against A2
G = "GPT-6.1-sol P1"
COMPS = [(G, b) for b in P if b != G] + [("PG-Lattice", "Leave-one-out mean"), ("Molmo2-4B + LoRA P1", "A1 counts")]
res, res_run = {}, {}
for a, b in COMPS:
    key = f"{a} minus {b}"
    r = full_paired(A2, a, b)
    if a == G:
        r["status"] = STATUS[b]
    elif a == "PG-Lattice":
        r["status"] = ("label_day's frozen paired row 'Ours vs Leave-one-out mean', run on A2 alone "
                       "(single-annotator run, plan item 3)")
    else:
        r["status"] = ("comparison of the kind pre-specified by plan item 3 (method-vs-A2 error against A1-vs-A2 error); "
                       "paired() test post hoc")
    if key in EXISTING:
        r["existing"] = EXISTING[key]
    res[key] = r
    res_run[key] = paired(A2_RUN, P[a] if a != "Leave-one-out mean" else S.loo_mean(A2_RUN),
                          P[b] if b != "Leave-one-out mean" else S.loo_mean(A2_RUN))
# Holm over the 8 GPT-minus-X Wilcoxon p values (post hoc family)
fam = [k for k in res if k.startswith(G + " minus")]
ps_ = np.array([res[k]["p_wilcoxon"] for k in fam])
o = np.argsort(ps_)
adj = np.empty(len(ps_))
run = 0.0
for i, j in enumerate(o):
    run = max(run, min(1.0, (len(ps_) - i) * ps_[j]))
    adj[j] = run
for k, a_ in zip(fam, adj):
    res[k]["p_wilcoxon_holm_gpt_family8"] = float(a_)

chk["run_reference_variant_max_abs_diff"] = {
    k: float(max(abs(res[k][f] - res_run[k][f]) for f in ("delta", "lo", "hi"))) for k in res}
chk["run_reference_variant_reproduces_stored"] = {
    "GPT minus PG-Lattice (stored Ours-GPT +7.685 [+1.884, +13.622])":
        [round(res_run["GPT-6.1-sol P1 minus PG-Lattice"][f], 3) for f in ("delta", "lo", "hi")],
    "PG-Lattice minus LOO (stored +1.629 [-6.093, +9.386])":
        [round(res_run["PG-Lattice minus Leave-one-out mean"][f], 3) for f in ("delta", "lo", "hi")],
    "GPT minus LOO (stored LOO-GPT +6.056 [+0.563, +11.659])":
        [round(res_run["GPT-6.1-sol P1 minus Leave-one-out mean"][f], 3) for f in ("delta", "lo", "hi")],
}

# extras (post hoc): every method minus A1's counts against A2 = the noise-floor table under the A2 reference
extra_floor = {}
for a in P:
    if a in ("A1 counts",):
        continue
    extra_floor[f"{a} minus A1 counts"] = full_paired(A2, a, "A1 counts")

# ------------------------------------------------------------------ 2. symmetric error |log2 ratio|
log_req = {
    "GPT-6.1-sol P1 minus A1 counts": paired_log(A2, P[G], P["A1 counts"]),
    "Molmo2-4B + LoRA P1 minus GPT-6.1-sol P1": paired_log(A2, P["Molmo2-4B + LoRA P1"], P[G]),
}
log_req["GPT-6.1-sol P1 minus A1 counts"]["existing"] = {
    "value": "-0.069 [-0.161, +0.019] (BNlogGptMinusHum)", "scheme": "loop bootstrap 10,000, default_rng(20261012), shared rng stream",
    "source": "paper/tools/reference/logratio_endpoint.py (logratio_endpoint.json)"}
for k in log_req:
    log_req[k]["status"] = ("metric pre-specified (plan item 5: mean |log2| against A2 alone); this paired test is post hoc"
                            if "A1" in k else "post hoc")
log_extra = {f"{G} minus {b}": paired_log(A2, P[G], P[b]) for b in P if b != G}
log_extra["PG-Lattice minus Leave-one-out mean"] = paired_log(A2, P["PG-Lattice"], P["Leave-one-out mean"])
log_extra["Molmo2-4B + LoRA P1 minus A1 counts"] = paired_log(A2, P["Molmo2-4B + LoRA P1"], P["A1 counts"])
log_per_method = {k: float(scene_log(A2, p).mean()) for k, p in P.items()}
# extras (post hoc): does each method beat the label-free mean-density guess against A2, on MAPE and on |log2|?
extra_vs_loo = {}
for a in P:
    if a == "Leave-one-out mean":
        continue
    extra_vs_loo[f"{a} minus Leave-one-out mean"] = {"mape": full_paired(A2, a, "Leave-one-out mean"),
                                                     "abs_log2": paired_log(A2, P[a], P["Leave-one-out mean"])}


# ------------------------------------------------------------------ 3a. half / double lines against A2
def lines_ratio(p, ref):
    return pd.concat([np.log2(p.h / ref.h), np.log2(p.v / ref.v)], keys=["h", "v"])


band = {}
for k in list(P):
    r = lines_ratio(P[k], A2)
    dbl, hlf = (np.abs(r - 1) <= 0.3), (np.abs(r + 1) <= 0.3)
    sc = (dbl | hlf).groupby(level=1).any()
    band[k] = dict(lines_answered=int(r.notna().sum()), half=int(hlf.sum()), double=int(dbl.sum()),
                   total=int((hlf | dbl).sum()), scenes_any=int(sc.sum()),
                   scenes_half=int(hlf.groupby(level=1).any().sum()), scenes_double=int(dbl.groupby(level=1).any().sum()),
                   lines_abs_log2_gt_0_5=int((r.abs() > 0.5).sum()),
                   requested=k in ("PG-Lattice", G, "A1 counts"))
band_vs_A1_context = {}   # post hoc context: the same band against A1 (does a method share A2's or A1's octave?)
for k, p in (("PG-Lattice", P["PG-Lattice"]), (G, P[G]), ("Molmo2-4B + LoRA P1", P["Molmo2-4B + LoRA P1"]),
             ("CNN trained on A1 (fabric-grouped)", P["CNN trained on A1 (fabric-grouped)"]),
             ("CNN-VLM ensemble (ens-v1)", P["CNN-VLM ensemble (ens-v1)"]), ("A2 counts", A2)):
    r = lines_ratio(p, A1)
    dbl, hlf = (np.abs(r - 1) <= 0.3), (np.abs(r + 1) <= 0.3)
    band_vs_A1_context[k] = dict(half=int(hlf.sum()), double=int(dbl.sum()), total=int((hlf | dbl).sum()),
                                 lines_abs_log2_gt_0_5=int((r.abs() > 0.5).sum()), mean_abs_log2=float(r.abs().mean()))
band_existing = {"source": "an earlier independent tally with the same band (not part of this folder)",
                 "A1": "30 double / 18 half", "GPT": "3 double / 34 half", "PG-Lattice": "12 double / 24 half (36 lines, 28 scenes)",
                 "ens-v1": "2 double / 13 half"}


# ------------------------------------------------------------------ 3b. octave chosen with the reference's counts (upper bound)
def oracle(p, ref):
    j = ref[["h", "v"]].join(p[["h", "v"]], rsuffix="_p", how="inner").dropna()
    before = 0.5 * (100 * (j.h_p - j.h).abs() / j.h + 100 * (j.v_p - j.v).abs() / j.v)
    kh, kv = np.round(np.log2(j.h / j.h_p)), np.round(np.log2(j.v / j.v_p))
    ah = 100 * (j.h_p * 2.0 ** kh - j.h).abs() / j.h
    av = 100 * (j.v_p * 2.0 ** kv - j.v).abs() / j.v
    after = 0.5 * (ah + av)
    both5 = int(((ah <= 5) & (av <= 5)).sum())
    return dict(n=int(len(j)), mape_before=float(before.mean()), mape_after=float(after.mean()),
                axes_changed=int((kh != 0).sum() + (kv != 0).sum()),
                axes_halved_by_oracle=int((kh < 0).sum() + (kv < 0).sum()),
                axes_doubled_by_oracle=int((kh > 0).sum() + (kv > 0).sum()),
                axes_k_abs_ge2=int((kh.abs() >= 2).sum() + (kv.abs() >= 2).sum()),
                both5_after=both5, both5_before=int((((j.h_p - j.h).abs() / j.h <= .05) & ((j.v_p - j.v).abs() / j.v <= .05)).sum()),
                after_scene_ci_scorer=None)


def oracle_band(p, ref):
    """post hoc variant: correct only the axes in the paper's half/double band (x2 or x0.5), nothing else."""
    j = ref[["h", "v"]].join(p[["h", "v"]], rsuffix="_p", how="inner").dropna()
    out = {}
    for ax in "hv":
        r = np.log2(j[f"{ax}_p"] / j[ax])
        k = np.where(np.abs(r - 1) <= 0.3, -1.0, np.where(np.abs(r + 1) <= 0.3, 1.0, 0.0))
        out[ax] = 100 * (j[f"{ax}_p"] * 2.0 ** k - j[ax]).abs() / j[ax]
        out[f"n{ax}"] = int((k != 0).sum())
    after = 0.5 * (out["h"] + out["v"])
    return dict(n=int(len(j)), mape_after_band_only=float(after.mean()), axes_changed=out["nh"] + out["nv"],
                both5_after=int(((out["h"] <= 5) & (out["v"] <= 5)).sum()))


def mape_ci(e, n_boot=2000, seed=0):   # score150 metrics(): scene bootstrap 2,000, default_rng(0)
    e = np.asarray(e)
    rng = np.random.default_rng(seed)
    bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


orc = {}
for k in ("PG-Lattice", G, "A1 counts", "Molmo2-4B + LoRA P1", "CNN-VLM ensemble (ens-v1)"):
    orc[k] = oracle(P[k], A2)
    orc[k]["requested"] = k == "PG-Lattice"
    orc[k].pop("after_scene_ci_scorer")
    orc[k]["post_hoc_band_only_variant"] = oracle_band(P[k], A2)
# validation: the same code reproduces the paper's A1 numbers and the mean-reference numbers. The mean reference as
# label_day.py stores it in its per-scene table (main150_per_scene.csv): each annotator's count to 2 decimals (the tool
# export), their mean, then reference and predictions rounded to 3 decimals.
GT_MEAN = (A1.round(2) + A2.round(2)) / 2
PRED_MEAN_RUN = {"Ours (lattice + fusion)": S.load_predictions()["Ours (lattice + fusion)"],
                 "GPT-6.1-sol, P1": S.load_predictions()["GPT-6.1-sol, P1"]}


def ps_table(m):   # label_day.py step_score: per_scene_ape(gt, pr), written with round(3)
    pr = PRED_MEAN_RUN[m].copy()
    pr.index = pr.index.astype(int)
    return per_scene_ape(GT_MEAN, pr).round(3)


def from_ps(m):
    g = ps_table(m)
    return pd.DataFrame({"h": g.h_p, "v": g.v_p})


ref_mean = ps_table("Ours (lattice + fusion)")[["h", "v"]].sort_index()
orc_valid = {
    "PG-Lattice vs A1 (paper BNoursSA 41.7 -> BNoracleOursSA 13.3)": oracle(P["PG-Lattice"], A1),
    "GPT vs A1 (paper BNoracleGptSA 13.6)": oracle(P[G], A1),
    "PG-Lattice vs mean, label_day run (final_numbers.tex FNoursOracleMape 18.3)": oracle(from_ps("Ours (lattice + fusion)"), ref_mean),
    "GPT vs mean, label_day run (FNgptOracleMape 17.8)": oracle(from_ps("GPT-6.1-sol, P1"), ref_mean),
}
for v in orc_valid.values():
    v.pop("after_scene_ci_scorer")
orc_valid["PG-Lattice vs A1 (paper BNoursSA 41.7 -> BNoracleOursSA 13.3)"]["post_hoc_band_only_variant"] = oracle_band(P["PG-Lattice"], A1)

# ------------------------------------------------------------------ write
R = {
    "description": __doc__.strip().splitlines()[0],
    "reference": "A2 alone, per-scene counts (76 scenes)",
    "scheme": {"paired": "paper/tools/label_day.py paired() imported: per-scene MAPE = mean(APE_h, APE_v); a minus b; "
                         "10,000 scene resamples, default_rng(0), percentile 95% CI; Wilcoxon signed-rank (scipy default)",
               "fabric_ci": "secondary: 44 fabric groups (paper/tables/fabric_groups_v2.csv), 10,000, default_rng(20261012)",
               "log2": "per scene 0.5*(|log2(h_p/h)| + |log2(v_p/v)|); same paired scheme",
               "band": "| |log2(pred/A2)| - 1 | <= 0.3 (paper Sec. 5.2 band)",
               "oracle": "octave upper bound: k = round(log2(ref/pred)) per axis, pred*2^k"},
    "post_hoc": "Everything in this file is post hoc (computed after every count was opened) unless its "
                "'status' says a comparison or metric was pre-specified; even then the exact test here (seed 0) is a "
                "recomputation and the pre-specified result is the 'existing' one.",
    "sources": SRC,
    "per_method_mape_vs_A2_all_scenes": {k: dict(n=int(per_scene_ape(A2, p).shape[0]), mape=float(per_scene_ape(A2, p).ape.mean()),
                                                 mape_ci_score150=mape_ci(per_scene_ape(A2, p).ape))
                                         for k, p in P.items()},
    "per_method_mean_abs_log2_vs_A2": log_per_method,
    "paired_mape_vs_A2": res,
    "paired_mape_vs_A2_extra_every_method_minus_A1_post_hoc": extra_floor,
    "paired_abs_log2_vs_A2_requested": log_req,
    "paired_abs_log2_vs_A2_extra_post_hoc": log_extra,
    "paired_vs_leave_one_out_mean_extra_post_hoc": extra_vs_loo,
    "half_double_lines_vs_A2": {"of_152_lines": band, "existing": band_existing,
                                "context_same_band_vs_A1_post_hoc": band_vs_A1_context,
                                "status": "band pre-defined in the paper (Sec. 5.2) and used against A1; these tallies against A2 are post hoc"},
    "octave_upper_bound_vs_A2": {"results": orc, "validation_same_code": orc_valid,
                                 "status": "post hoc upper bound (labels choose the octave), as the paper's A1 number"},
    "checks": chk,
}
FROZEN_AFTER = {"label_day.py_sha256": sha(f"{TOOLS}/label_day.py"), "score150.py_md5": sha(f"{TOOLS}/score150.py", "md5")}
R["checks"]["frozen_unchanged"] = FROZEN_AFTER == FROZEN_BEFORE
R["checks"]["label_day.py_matches_tools_freeze_checksums"] = FROZEN_BEFORE["label_day.py_sha256"] == FREEZE["paper/tools/label_day.py"]["sha256_released"]
R = rnd(R, 4)
os.makedirs(OUT, exist_ok=True)
json.dump(R, open(f"{OUT}/paired_ref.json", "w"), indent=1)

# ------------------------------------------------------------------ console summary
f = lambda r: f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p_wilcoxon']:.3g} fabric[{r['fabric_ci'][0]:+.2f}, {r['fabric_ci'][1]:+.2f}] n={r['n']} (a {r.get('mape_a', r.get('log_a')):.2f}, b {r.get('mape_b', r.get('log_b')):.2f}; a closer {r['a_closer']}/{r['n']})"
print("== paired MAPE vs A2")
for k, r in res.items():
    print(f"{k:70s} {f(r)}" + (f" holm={r['p_wilcoxon_holm_gpt_family8']:.3g}" if 'p_wilcoxon_holm_gpt_family8' in r else ""))
print("== extra: X minus A1 vs A2")
for k, r in extra_floor.items():
    print(f"{k:70s} {f(r)}")
print("== |log2| requested")
for k, r in log_req.items():
    print(f"{k:70s} {f(r)}")
print("== |log2| extra")
for k, r in log_extra.items():
    print(f"{k:70s} {f(r)}")
print("== extra: X minus LOO vs A2 (MAPE | |log2|)")
for k, r in extra_vs_loo.items():
    print(f"{k:60s} {f(r['mape'])}\n{'':60s} {f(r['abs_log2'])}")
print("== band lines vs A2")
for k, r in band.items():
    print(f"{k:40s} {r}")
print("== band lines vs A1 (context)")
for k, r in band_vs_A1_context.items():
    print(f"{k:40s} {r}")
print("== oracle vs A2")
for k, r in orc.items():
    print(f"{k:40s} {r}")
for k, r in orc_valid.items():
    print(f"{k:70s} {r['mape_before']:.2f} -> {r['mape_after']:.2f} ({r['axes_changed']} axes)")
print("== checks")
print(json.dumps(rnd(chk, 4), indent=1))

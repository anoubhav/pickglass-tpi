"""legacy_unified.py -- the legacy-13 table under ONE crop rule (supplement Tab. S3).

Every row is scored on the CANONICAL legacy input: vlm/crops/legacy/{name}.png
(vlm/data_vlm.detect_legacy_window trims the dark pick-glass frame -- per side, axis-aligned,
absolute dark level 40 -- and the remainder is resized to a square of side round((h+w)/2) with
INTER_AREA; that square is taken as the 1-inch window).  These are the exact pixels every VLM,
Molmo2 (Mac 4-bit MLX and GPU bf16) and CountGD saw: the PNGs reproduce bit-exactly from
data_vlm.legacy_window_crop; the GPU copies used by Molmo2 bf16 have the same file md5; CountGD's
crops, recomputed on the server from md5-identical originals, are pixel-identical (20/20 decoded
arrays).  Per-image boxes/trims and visual check: tables/src/legacy_crop_rule*.  The classical counters,
the CNN and the legacy baselines were re-run on the same PNGs for this table.

Sources (all read-only):
  GT            methods/legacy_2019/run_pipeline.py load_gt (legacy harness parser, extracted by ast)
  classical     methods/results/legacy13_vlmcrops_estimators.csv        (methods/run_legacy_vlmcrops.py,
                frozen counter methods/results/counting_frozen_final.py) + ..._latency.csv
  legacy doc3   methods/legacy_2019/run_vlmcrops/results.csv  (+ seed_sweep_precropped.csv)
  legacy test.py  testpy_rgb_2019 rows of legacy13_vlmcrops_estimators.csv (weft only)
  CNN           cnn/runs/synth_r18_v2mix_aniso/eval_vlmcrops/legacy_predictions_vlm.csv (+ latency json)
  VLMs (API, Molmo2 4-bit MLX)  vlm/results/legacy_predictions.csv
  Molmo2 bf16   gpu_results/molmo2_bf16/results/legacy_predictions.csv if synced, else the read-only
                snapshot paper/tables/src/molmo2_bf16_legacy_predictions.csv (md5 23018a88...)
  CountGD       gpu_results/countgd/outputs/results/legacy_predictions.csv (+ stageA_legacy_metrics.csv)

Scoring: 13 images x 2 counts = 26.  'axis' (PRIMARY, as in the paper): vertical yarns
(h_count) <-> warp, horizontal yarns (v_count) <-> weft.  'sorted': larger prediction <->
larger GT (the VLM scorer's 'best assignment').  Parse failures are not imputed (n shown).
95 % CIs: percentile bootstrap, 10 000 resamples of the IMAGES (both counts of an image stay
together).  Paired tests (ours vs others): paired bootstrap over images on the difference of
MAE / MAPE, Wilcoxon signed-rank on the per-count absolute (and absolute %) error differences
(n=26) and on the per-image mean absolute % errors (n=13), exact McNemar on within-10 %.

Outputs: tables/legacy13_unified.csv, tables/legacy13_unified.tex (full, table*),
         tables/legacy13_unified_main.tex (compact drop-in for tab:legacy),
         tables/legacy13_paired.csv, tables/legacy13_crop_sensitivity.csv,
         tables/legacy13_unified_per_count.csv (every prediction that entered the table).

    <venv2>/bin/python tools/legacy_unified.py
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import math
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TAB = os.path.join(PAPER, "tables")
B = 10_000
SEED = 20261008

P_RUNPIPE = os.path.join(ROOT, "methods", "legacy_2019", "run_pipeline.py")
GT_XLSX = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "legacy2019")), "Warp-Weft count.xlsx")
P_EST = os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_estimators.csv")
P_EST_LAT = os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_estimators_latency.csv")
P_EST_P1 = os.path.join(ROOT, "methods", "results", "legacy13_estimators.csv")
P_WIN = os.path.join(ROOT, "methods", "results", "legacy13_window_scale.csv")
P_2019 = os.path.join(ROOT, "methods", "legacy_2019", "run_vlmcrops", "results.csv")
P_2019_SWEEP = os.path.join(ROOT, "methods", "legacy_2019", "run_vlmcrops", "seed_sweep_precropped.csv")
P_2019_ORIG = os.path.join(ROOT, "methods", "legacy_2019", "run1", "results.csv")
P_2019_LAT = os.path.join(ROOT, "methods", "legacy_2019", "run_vlmcrops", "latency.json")
CNN_RUN = os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso")
P_CNN = os.path.join(CNN_RUN, "eval_vlmcrops", "legacy_predictions_vlm.csv")
P_CNN_LAT = os.path.join(CNN_RUN, "eval_vlmcrops", "latency.json")
P_VLM = os.path.join(ROOT, "vlm", "results", "legacy_predictions.csv")
P_MOLMO_SYNC = os.path.join(ROOT, "gpu_results", "molmo2_bf16", "results", "legacy_predictions.csv")
# GPU-side results synced to gpu_results/results/ (md5 23018a88..., identical to the snapshot)
P_MOLMO_SYNC2 = os.path.join(ROOT, "gpu_results", "results", "legacy_predictions.csv")
P_MOLMO_SNAP = os.path.join(TAB, "src", "molmo2_bf16_legacy_predictions.csv")
P_CG = os.path.join(ROOT, "gpu_results", "countgd", "outputs", "results", "legacy_predictions.csv")
# Molmo2-4B base and LoRA-tuned on synthetic pointing data only (ft_results/code/summarize_ft.py)
P_FT = os.path.join(ROOT, "ft_results", "results", "legacy_predictions.csv")
# CNN augmentation / pseudo-label variants of cnn/RESULTS_AUG.md (scene-split selection), scored once
# each on legacy-13 by cnn/train_aug.py final after selection (cnn/runs_aug/selection.json).
AUG_FINAL = os.path.join(ROOT, "cnn", "runs_aug", "final")
P_CG_MET = os.path.join(ROOT, "gpu_results", "countgd", "outputs", "results", "stageA_legacy_metrics.csv")


# ============================================================================ ground truth
def load_gt():
    src = open(P_RUNPIPE).read()
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
            and n.name in ("_parse_count", "_parse_weave", "load_gt")]
    assert len(keep) == 3
    import openpyxl
    ns = {"re": re, "os": os, "openpyxl": openpyxl, "_dt": _dt}
    exec(compile(ast.Module(body=keep, type_ignores=[]), P_RUNPIPE, "exec"), ns)
    gt, _ = ns["load_gt"](GT_XLSX)
    rows = []
    for k, g in gt.items():
        if not g["usable"]:
            continue
        rows.append(dict(image=k, warp=g["warp"], weft=g["weft"], weave=g["weave"], unclear=g["unclear"],
                         warp_raw=g["warp_raw"], weft_raw=g["weft_raw"],
                         warp_formula="=" in str(g["warp_raw"]), weft_formula="=" in str(g["weft_raw"]),
                         is_range=(g["warp_lo"] != g["warp_hi"]) or (g["weft_lo"] != g["weft_hi"])))
    df = pd.DataFrame(rows).set_index("image")
    assert len(df) == 13, len(df)
    return df


# ============================================================================ predictions
def P(group, method, prompt, h, v, *, key=None, model=None, input_px="native (1626-1941)", usd=np.nan,
      sec=np.nan, sec_hw="", offline="", note="", in_tex=True, repeat=0, weft_only=False, parsed_of=13):
    """One method = dict with per-image predictions h (vertical yarns <-> warp), v (<-> weft)."""
    return dict(group=group, method=method, prompt=prompt, key=key or f"{method}|{prompt}", model=model,
                h=dict(h), v=dict(v), input_px=input_px, usd_per_img=usd, s_per_img=sec, latency_hw=sec_hw,
                offline=offline, note=note, in_tex=in_tex, repeat=repeat, weft_only=weft_only)


EST_NAME = {"ours": "Ours (PG-Lattice: lattice + fusion)", "lattice": "Lattice only", "gabor": "Log-Gabor bank",
            "fft": "FFT max near-axis peak", "fft_harm": "FFT harmonic sum", "profile": "Deskewed profile + YIN",
            "peaks": "Profile peak count", "tiles": "FFT tile voting", "fusion3": "FFT/profile/tile fusion (v0)"}
EST_ORDER = ["ours", "lattice", "gabor", "fft", "fft_harm", "tiles", "profile", "peaks", "fusion3"]
VLM_NAME = {"gpt-6.1-sol": "GPT-6.1-sol", "gpt-6.1-sol@reasoning-high": "GPT-6.1-sol (high reasoning)",
            "gemini-3.8-flash": "Gemini 3.8 Flash", "qwen3.8-max-prime": "Qwen3.8-Max-Prime",
            "gemini-3.1-pro": "Gemini 3.1 Pro", "claude-fable-5.1": "Claude Fable 5.1",
            "qwen3.8-27b": "Qwen3.8-27B (open)", "gemma-4-26b-a4b": "Gemma-4-26B-A4B (open)",
            "molmo2_8b": "Molmo2-8B 4-bit MLX", "molmo2_8b_bf16": "Molmo2-8B bf16"}
VLM_ORDER = ["gpt-6.1-sol", "gpt-6.1-sol@reasoning-high", "gemini-3.8-flash", "qwen3.8-max-prime",
             "gemini-3.1-pro", "claude-fable-5.1", "qwen3.8-27b", "gemma-4-26b-a4b"]
PROMPT_ORDER = ["P1", "P2", "P3", "P3s", "P4", "P4q"]


def classical():
    d = pd.read_csv(P_EST)
    d = d[d.source == "vlmcrop"]
    lat = pd.read_csv(P_EST_LAT)
    out = []
    for est in EST_ORDER:
        g = d[d.estimator == est].set_index("image")
        if est in lat.estimator.values:
            sec = float(lat[lat.estimator == est].latency_s.median())
            hw = "M2 Max CPU, median"
        else:  # estimate_many shares the pre-processing; runtime_s_shared = own time + pre-processing
            sec = float(g.runtime_s_shared.median())
            hw = "M2 Max CPU (1 thread, shared pre-proc), median"
        out.append(P("classical", EST_NAME[est], "--", g.h, g.v, key=f"est:{est}", usd=0.0, sec=sec, sec_hw=hw,
                     offline="yes", in_tex=True,
                     note="frozen counter (counting_frozen_final.py)" if est == "ours" else ""))
    t = d[d.estimator == "testpy_rgb_2019"].set_index("image")
    out.append(P("2019", "2019 RGB irregularity (weft only)", "--", {}, t.v, key="2019:testpy", usd=0.0,
                 sec=np.nan, offline="yes", weft_only=True,
                 note="FR-final test.py re-run on the canonical crops, no-blur variant that reproduces "
                      "the 2019 xlsx column (14/20 exact)"))
    return out


def doc3():
    r = pd.read_csv(P_2019)
    r = r.set_index("key")
    lat = json.load(open(P_2019_LAT)) if os.path.exists(P_2019_LAT) else {}
    return [P("2019", "2019 dot map + transition count", "--", r.pred_warp_per_inch, r.pred_weft_per_inch,
              key="2019:doc3", usd=0.0, sec=lat.get("median_s", np.nan), sec_hw=lat.get("hw", ""), offline="yes",
              note="faithful doc3.py+count.py, seed 0 (k-means unseeded in 2019); seeds 1-4 in "
                   "legacy13_crop_sensitivity.csv")]


def cnn():
    c = pd.read_csv(P_CNN).set_index("name")
    lat = json.load(open(P_CNN_LAT)) if os.path.exists(P_CNN_LAT) else {}
    return [P("learned", "CNN, synthetic only (zero-shot)", "--", c.pred_h, c.pred_v, key="cnn:synth_r18",
              input_px="cap 1024, TTA 672/768/864 px/in", usd=0.0, sec=lat.get("median_s", np.nan),
              sec_hw=lat.get("hw", ""), offline="yes",
              note="ResNet-18, runs/synth_r18_v2mix_aniso/model.pt via cnn/evaluate.py --legacy-modes vlm")]


def _vlm_rows(df, model, prompt, repeat, label, *, input_px, hw, offline, note="", in_tex=True):
    g = df[(df.model == model) & (df.method == prompt) & (df.repeat == repeat) & (df.set == "legacy")]
    g = g[g.gt_warp.notna()]
    ok = g[g.ok.astype(str) == "True"].set_index("image")
    usd = float(g.usd.astype(float).mean()) if "usd" in g else np.nan
    sec = float(g.seconds.astype(float).mean())
    n_fail = int((g.ok.astype(str) != "True").sum())
    nt = note + (f"; {n_fail}/13 GT images unparsed (not imputed)" if n_fail else "")
    return P("vlm", label, prompt, ok.vertical.astype(float), ok.horizontal.astype(float),
             key=f"vlm:{model}|{prompt}|r{repeat}", model=model, input_px=input_px, usd=usd, sec=sec, sec_hw=hw,
             offline=offline, note=nt.strip("; "), in_tex=in_tex, repeat=repeat)


def vlms():
    v = pd.read_csv(P_VLM)
    out = []
    for m in VLM_ORDER:
        open_w = m in ("qwen3.8-27b", "gemma-4-26b-a4b")
        off = "no (API; open weights, not run locally)" if open_w else "no (API)"
        for p in PROMPT_ORDER:
            for rep in (0, 1, 2):
                if not len(v[(v.model == m) & (v.method == p) & (v.repeat == rep) & (v.set == "legacy")]):
                    continue
                out.append(_vlm_rows(v, m, p, rep, VLM_NAME[m] + ("" if rep == 0 else f" (repeat {rep})"),
                                     input_px="1536 (JPEG q92)", hw="OpenRouter API, mean", offline=off,
                                     in_tex=(rep == 0)))
    # Molmo2 4-bit MLX on the Mac (P1 complete; other prompts only 3-4/13 -> CSV only)
    for p in PROMPT_ORDER:
        g = v[(v.model == "molmo2_8b") & (v.method == p) & (v.repeat == 0) & (v.set == "legacy") & v.gt_warp.notna()]
        if len(g):
            r = _vlm_rows(v, "molmo2_8b", p, 0, VLM_NAME["molmo2_8b"], input_px="644",
                          hw="M2 Max (MLX, shared GPU), mean", offline="yes", in_tex=False,
                          note="" if len(g) == 13 else f"partial run: {len(g)}/13 GT images")
            out.append(r)
    # Molmo2 bf16 on the GPU server
    src = next((q for q in (P_MOLMO_SYNC, P_MOLMO_SYNC2) if os.path.exists(q)), P_MOLMO_SNAP)
    mb = pd.read_csv(src)
    for p in PROMPT_ORDER:
        if len(mb[(mb.method == p) & (mb.repeat == 0)]):
            out.append(_vlm_rows(mb, "molmo2_8b_bf16", p, 0, VLM_NAME["molmo2_8b_bf16"], input_px="644",
                                 hw="RTX 4090 (transformers bf16), mean",
                                 offline="GPU (open weights, RTX 4090; P1 also run 4-bit on the M2 Max)",
                                 note=f"source {os.path.relpath(src, ROOT)}"))
    return out


def countgd():
    c = pd.read_csv(P_CG)
    met = pd.read_csv(P_CG_MET).set_index("method")
    out = []
    for m, label, intex, note in (
            ("CG-B-yarn-ex3", "CountGD (pre-declared primary)", True,
             "official adaptive-crop strips, 'yarn', text + 3 automatic exemplars"),
            ("CG-W-ex3-lat", "CountGD, whole-crop cells + lattice", True,
             "variant (not the pre-declared primary): cell detections -> lattice spacing")):
        g = c[(c.method == m) & (c.repeat == 0) & c.gt_warp.notna()]
        ok = g[g.ok.astype(str) == "True"].set_index("image")
        out.append(P("learned", label, "--", ok.vertical.astype(float), ok.horizontal.astype(float),
                     key=f"countgd:{m}", input_px="canonical crop (strips/tiles)", usd=0.0,
                     sec=float(met.loc[m, "sec_per_img"]), sec_hw="RTX 4090, mean",
                     offline="GPU (open weights, RTX 4090; 3 crops also on the M2 Max GPU)", note=note, in_tex=intex))
    return out


def molmo4b_ft():
    """Molmo2-4B, base and LoRA-tuned on 12k synthetic crops with exact crossing points. The primary
    readout of the tuned model (P1) was fixed on synthetic validation (ft_results/results/selection.json)."""
    if not os.path.exists(P_FT):
        return []
    d = pd.read_csv(P_FT, dtype={"image": str})
    out = []
    for model, label in (("molmo2_4b_base", "Molmo2-4B"), ("molmo2_4b_synthft", "Molmo2-4B + synthetic LoRA")):
        for meth in ("P1", "P4", "P4q", "P4s"):
            g = d[(d.model == model) & (d.method == meth)]
            if not len(g):
                continue
            ok = g[g.ok.astype(str) == "True"].set_index("image")
            out.append(P("vlm", label, meth, ok.vertical.astype(float), ok.horizontal.astype(float),
                         key=f"ft:{model}|{meth}", model=model, input_px="644", usd=0.0,
                         sec=float(g.seconds.astype(float).mean()),
                         sec_hw="RTX 4090 (transformers bf16, batched greedy), mean",
                         offline="GPU (open weights, RTX 4090)",
                         note=("LoRA rank 32 on synthetic pointing data only; P1 = primary readout fixed on "
                               "synthetic validation" if model.endswith("synthft") else "zero-shot base model"),
                         in_tex=meth in ("P1", "P4s")))
    return out


def cnn_aug():
    """The tile CNN after label-exact augmentation (V2, synthetic only) and after synthetic
    pre-training + fine-tuning on our counter's confident counts for 28 of the 76 scenes (V4; V5 = V4 + TTA).
    Variants and checkpoints were chosen on synthetic + held-out pseudo-labelled scenes only
    (cnn/runs_aug/PREREG.json, selection.json); legacy-13 was predicted once per variant."""
    spec = [("V2_r18", "CNN, + label-exact augmentation (R18, V2)", False),
            ("V4_b0", "CNN, syn. + pseudo-label FT (B0, V4)", True),
            ("V5_b0", "CNN, syn. + pseudo-label FT + TTA (B0, V5)", False)]
    out = []
    for run, label, intex in spec:
        p = os.path.join(AUG_FINAL, run, "legacy_predictions_vlm.csv")
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p).set_index("name")
        out.append(P("learned", label, "--", c.pred_h, c.pred_v, key=f"aug:{run}",
                     input_px="cap 1024, TTA 672/768/864 px/in", usd=0.0, offline="yes",
                     note=f"cnn/runs_aug/final/{run} (cnn/RESULTS_AUG.md)", in_tex=intex))
    return out


def loo_mean(gt):
    n = len(gt)
    lw = (gt.warp.sum() - gt.warp) / (n - 1)
    lf = (gt.weft.sum() - gt.weft) / (n - 1)
    return P("reference", "Leave-one-out mean", "--", lw, lf, key="ref:loo_mean", input_px="--", usd=0.0,
             sec=np.nan, offline="yes", note="mean of the other 12 hand counts, per axis")


# ============================================================================ scoring
def per_count(m, gt, assign="axis"):
    """-> DataFrame, one row per scored count: image, axis, pred, gt, flags."""
    rows = []
    for img, g in gt.iterrows():
        h, v = m["h"].get(img, np.nan), m["v"].get(img, np.nan)
        if m["weft_only"]:
            if np.isfinite(v):
                rows.append(dict(image=img, axis="weft", pred=v, true=g.weft, plain=g.weave == "1/1",
                                 formula=g.weft_formula))
            continue
        if not (np.isfinite(h) and np.isfinite(v)):
            continue
        if assign == "axis":
            pairs = [("warp", h, g.warp, g.warp_formula), ("weft", v, g.weft, g.weft_formula)]
        else:
            (pl, ph), gs = sorted([h, v]), sorted([(g.warp, "warp", g.warp_formula), (g.weft, "weft", g.weft_formula)])
            pairs = [(gs[0][1], pl, gs[0][0], gs[0][2]), (gs[1][1], ph, gs[1][0], gs[1][2])]
        for ax, p, t, f in pairs:
            rows.append(dict(image=img, axis=ax, pred=float(p), true=float(t), plain=g.weave == "1/1", formula=bool(f)))
    d = pd.DataFrame(rows)
    if len(d):
        d["err"] = d.pred - d.true
        d["ae"] = d.err.abs()
        d["ape"] = 100 * d.ae / d.true
    return d


def boot_idx(n, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def image_sums(d, imgs):
    g = d.groupby("image")
    return (g.ae.sum().reindex(imgs).to_numpy(), g.ape.sum().reindex(imgs).to_numpy(),
            g.size().reindex(imgs).to_numpy().astype(float))


def boot_ci(d, seed):
    imgs = sorted(d.image.unique())
    ae, ape, cnt = image_sums(d, imgs)
    idx = boot_idx(len(imgs), seed)
    n = cnt[idx].sum(1)
    mae = ae[idx].sum(1) / n
    mape = ape[idx].sum(1) / n
    q = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
    return q(mae), q(mape)


def score(m, gt, assign):
    d = per_count(m, gt, assign)
    if not len(d):
        return None
    (mae_lo, mae_hi), (mape_lo, mape_hi) = boot_ci(d, SEED)
    pl, nf = d[d.plain], d[~d.formula]
    r = dict(n_images=d.image.nunique(), n_counts=len(d), MAE=d.ae.mean(), MAE_lo=mae_lo, MAE_hi=mae_hi,
             MAPE=d.ape.mean(), MAPE_lo=mape_lo, MAPE_hi=mape_hi, medAE=d.ae.median(), bias=d.err.mean(),
             n_w5=int((d.ape <= 5).sum()), n_w10=int((d.ape <= 10).sum()),
             n_gross25=int((d.ape > 25).sum()),
             MAE_plain=pl.ae.mean(), MAPE_plain=pl.ape.mean(), n_counts_plain=len(pl),
             n_w10_plain=int((pl.ape <= 10).sum()),
             MAE_noformula=nf.ae.mean(), MAPE_noformula=nf.ape.mean(), n_counts_noformula=len(nf))
    if assign == "axis":
        r["MAE_warp"] = d[d.axis == "warp"].ae.mean() if (d.axis == "warp").any() else np.nan
        r["MAE_weft"] = d[d.axis == "weft"].ae.mean()
    return r, d


# ============================================================================ paired tests
def paired(ma, mb, gt, assign, label, subset="all"):
    da, db = per_count(ma, gt, assign), per_count(mb, gt, assign)
    if subset == "noformula":  # drop the 3 formula-derived twill warp labels (a1, a6, r8)
        da, db = da[~da.formula], db[~db.formula]
    elif subset == "plain":
        da, db = da[da.plain], db[db.plain]
    k = ["image", "axis"]
    j = da.merge(db, on=k, suffixes=("_a", "_b"))
    imgs = sorted(j.image.unique())
    n = len(imgs)
    g = j.groupby("image")
    ae_a, ae_b = g.ae_a.sum().reindex(imgs).to_numpy(), g.ae_b.sum().reindex(imgs).to_numpy()
    pe_a, pe_b = g.ape_a.sum().reindex(imgs).to_numpy(), g.ape_b.sum().reindex(imgs).to_numpy()
    cnt = g.size().reindex(imgs).to_numpy().astype(float)
    idx = boot_idx(n, SEED + 1)
    N = cnt[idx].sum(1)
    dmae = (ae_a[idx].sum(1) - ae_b[idx].sum(1)) / N
    dmape = (pe_a[idx].sum(1) - pe_b[idx].sum(1)) / N

    def p2(x):  # two-sided bootstrap p for H0: difference = 0
        return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))

    def wil(x):
        x = np.asarray(x, float)
        if np.all(x == 0):
            return np.nan
        return float(stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)

    w10a, w10b = j.ape_a <= 10, j.ape_b <= 10
    b_, c_ = int((w10a & ~w10b).sum()), int((~w10a & w10b).sum())
    mcn = float(stats.binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0
    img_mape_a, img_mape_b = pe_a / cnt, pe_b / cnt
    return dict(comparison=label, assign=assign, subset=subset, A=ma["key"], B=mb["key"], n_images=n, n_counts=len(j),
                MAE_A=j.ae_a.mean(), MAE_B=j.ae_b.mean(), dMAE=j.ae_a.mean() - j.ae_b.mean(),
                dMAE_lo=np.percentile(dmae, 2.5), dMAE_hi=np.percentile(dmae, 97.5), p_boot_MAE=p2(dmae),
                MAPE_A=j.ape_a.mean(), MAPE_B=j.ape_b.mean(), dMAPE=j.ape_a.mean() - j.ape_b.mean(),
                dMAPE_lo=np.percentile(dmape, 2.5), dMAPE_hi=np.percentile(dmape, 97.5), p_boot_MAPE=p2(dmape),
                p_wilcoxon_abs_err_26=wil(j.ae_a - j.ae_b), p_wilcoxon_abs_pct_err_26=wil(j.ape_a - j.ape_b),
                p_wilcoxon_image_mean_pct_err_13=wil(img_mape_a - img_mape_b),
                counts_A_better=int((j.ae_a < j.ae_b).sum()), counts_B_better=int((j.ae_a > j.ae_b).sum()),
                counts_tied=int((j.ae_a == j.ae_b).sum()),
                n_w10_A=int(w10a.sum()), n_w10_B=int(w10b.sum()), w10_only_A=b_, w10_only_B=c_, p_mcnemar_w10=mcn)


def mean_of_repeats(ms, key, label):
    imgs = set.intersection(*[set(k for k in m["h"] if np.isfinite(m["h"][k]) and np.isfinite(m["v"][k])) for m in ms])
    h = {i: float(np.mean([m["h"][i] for m in ms])) for i in imgs}
    v = {i: float(np.mean([m["v"][i] for m in ms])) for i in imgs}
    return P("vlm", label, "P1", h, v, key=key, in_tex=False)


# ============================================================================ crop sensitivity
def crop_sensitivity(gt, cur):
    """ours / gabor / CNN / legacy pipeline under the earlier crop rules vs the canonical one (axis-matched)."""
    rows = []

    def add(method, rule, h, v, note=""):
        m = P("x", method, "--", h, v)
        r = score(m, gt, "axis")[0]
        rows.append(dict(method=method, crop_rule=rule, note=note, **{k: r[k] for k in (
            "n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10", "MAE_plain", "bias")}))
    e = pd.read_csv(P_EST_P1)
    e = e[(e.crop == "gt") & (e.preproc == "lcn") & (e.glare == "mask") & (e.kind == "clean")]
    ws = pd.read_csv(P_WIN).set_index("image")
    for est in ("ours", "gabor", "lattice", "fft"):
        g = e[e.estimator == est].set_index("image")
        add(EST_NAME[est], "P1: whole 2019 hand crop = 1 inch", g.h, g.v, "legacy13_estimators.csv")
        add(EST_NAME[est], "P2: methods/legacy.detect_window scale", g.h * ws.sx.reindex(g.index),
            g.v * ws.sy.reindex(g.index), "count_P1 x window/image side")
        c = cur[f"est:{est}"]
        add(EST_NAME[est], "canonical: vlm/crops/legacy PNG", c["h"], c["v"], "this table")
    for mode, rule in (("raw", "P1: whole 2019 hand crop = 1 inch"), ("detect", "det.: methods/legacy.detect_window homography"),
                       ("trim", "trim: fabric_data.legacy_inner_window")):
        p = os.path.join(CNN_RUN, "eval_final", f"legacy_predictions_{mode}.csv")
        if os.path.exists(p):
            c = pd.read_csv(p).set_index("name")
            add("CNN, synthetic only", rule, c.pred_h, c.pred_v, f"eval_final/legacy_predictions_{mode}.csv")
    c = cur["cnn:synth_r18"]
    add("CNN, synthetic only", "canonical: vlm/crops/legacy PNG", c["h"], c["v"], "this table")
    r1 = pd.read_csv(P_2019_ORIG)
    r1 = r1[r1.variant == "precropped"].set_index("key")
    add("2019 dot map + transition count", "P1: whole 2019 hand crop (seed 0)", r1.pred_warp_per_inch,
        r1.pred_weft_per_inch, "legacy_2019/run1")
    c = cur["2019:doc3"]
    add("2019 dot map + transition count", "canonical (seed 0)", c["h"], c["v"], "this table")
    if os.path.exists(P_2019_SWEEP):
        sw = pd.read_csv(P_2019_SWEEP)
        for s, g in sw.groupby("seed"):
            g = g.set_index("key")
            add("2019 dot map + transition count", f"canonical (seed {s})", g.pred_warp_per_inch,
                g.pred_weft_per_inch, "k-means seed sweep")
    # legacy test.py: xlsx column (legacy crop) vs re-run on the canonical crop
    x = r1.xlsx_rgb_weft_pred
    m = P("x", "t", "--", {}, x, weft_only=True)
    r = score(m, gt, "axis")[0]
    rows.append(dict(method="2019 RGB irregularity (weft only)", crop_rule="P1: 2019 xlsx column", note="n=13 weft",
                     **{k: r[k] for k in ("n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10",
                                          "MAE_plain", "bias")}))
    import sys
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "methods"))
    from run_legacy_vlmcrops import testpy_weft, load as _load
    xo = {k: float(testpy_weft(_load(k, "orig_P1"), blur=False)) for k in gt.index}
    r = score(P("x", "t", "--", {}, xo, weft_only=True), gt, "axis")[0]
    rows.append(dict(method="2019 RGB irregularity (weft only)", crop_rule="P1: whole 2019 hand crop, no-blur re-run",
                     note="n=13 weft; reproduces 14/20 xlsx values exactly", **{k: r[k] for k in (
                         "n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10", "MAE_plain", "bias")}))
    r = score(cur["2019:testpy"], gt, "axis")[0]
    rows.append(dict(method="2019 RGB irregularity (weft only)", crop_rule="canonical: vlm/crops/legacy PNG",
                     note="n=13 weft, no-blur variant", **{k: r[k] for k in (
                         "n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10", "MAE_plain", "bias")}))
    return pd.DataFrame(rows)


# ============================================================================ LaTeX
def f1(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fusd(x):
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if x < 0.0001:
        return r"$<$0.0001"
    return f"{x:.4f}" if x < 0.001 else f"{x:.3f}"


def fsec(x):
    if not np.isfinite(x):
        return "--"
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


def foff(s):
    s = str(s)
    if s.startswith("yes"):
        return r"\checkmark"
    if s.startswith("GPU"):
        return "GPU"
    if "open weights" in s:
        return r"API$^{o}$"
    return "API"


def tex_row(r, show_prompt=True):
    n = int(r.n_counts)
    mape = f"{f1(r.MAPE)} [{f1(r.MAPE_lo)}, {f1(r.MAPE_hi)}]"
    w5, w10 = f"{int(r.n_w5)}/{n}", f"{int(r.n_w10)}/{n}"
    name = r.method_tex
    prompt = r.prompt if show_prompt else ""
    cells = [name, prompt, f1(r.MAE), mape, w5, w10, f1(r.MAE_plain), fusd(r.usd_per_img), fsec(r.s_per_img),
             foff(r.offline)]
    return " & ".join(cells) + r" \\"


HEAD = (r"Method & Prompt & MAE & MAPE [95\% CI] & $\le$5\% & $\le$10\% & Plain MAE & \$/img & s/img & Offline \\")


def write_tex(T, paired_df, path_full, path_main):
    ax = T[T["assign"] == "axis"].copy()
    ax["method_tex"] = ax.method.str.replace("&", r"\&").str.replace(
        "Ours (PG-Lattice: lattice + fusion)", r"\ours (lattice + fusion)", regex=False).str.replace(
        "CountGD, whole-crop cells + lattice", r"CountGD, cells $\to$ lattice (post hoc)", regex=False)
    groups = [("reference", "References"), ("2019", "2019 pipeline (re-run on the same crops)"),
              ("classical", "Classical spectral counters (frozen; CPU)"), ("learned", "Learned counters and counting specialists"),
              ("vlm", "Vision-language models (API unless marked)")]
    pr = paired_df[(paired_df["comparison"] == "ours vs GPT-6.1-sol P1") & (paired_df["assign"] == "axis")
                  & (paired_df["subset"] == "all")].iloc[0]
    lines = [r"% Generated by paper/tools/legacy_unified.py -- do not edit by hand.",
             r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{@{}llrcccrrrc@{}}", r"\toprule", HEAD, r"\midrule"]
    first = True
    for gkey, gname in groups:
        g = ax[(ax.group == gkey) & ax.in_tex]
        if gkey == "2019":
            g = g.sort_values("method", ascending=False)
        if not len(g):
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(r"\multicolumn{10}{@{}l}{\emph{" + gname + r"}} \\")
        for _, r in g.iterrows():
            lines.append(tex_row(r, show_prompt=(gkey == "vlm")))
    lines += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"\caption{Held-out legacy test\LEG under one crop rule: every row is scored on the same pixels, "
           r"the 2019 hand crops with the dark pick-glass frame trimmed by a per-side dark-band detector and "
           r"resized to a square taken as the 1-inch window (the VLM inputs; \texttt{vlm/crops/legacy}). "
           r"13 images, 26 counts, axis-matched (vertical yarns $\leftrightarrow$ ends); MAE in threads per window side; MAPE in \% "
           r"with a percentile-bootstrap 95\% CI over images (10k resamples); $\le$5\%/$\le$10\%: counts within "
           r"that relative error; Plain MAE: the 10 plain-weave images (20 counts). Three twill warp labels "
           r"(a1, a6, r8) were derived by formula ($a+2b$) and are the least certain GT. "
           r"Parse failures are not imputed (denominator shown). s/img: classical rows median on an M2 Max CPU "
           r"under shared load, VLM rows mean API wall-clock, Molmo2 bf16 and CountGD on an RTX 4090. "
           r"Offline: \checkmark\ runs without a network on a laptop; GPU: open weights, run on an RTX 4090 "
           r"(Molmo2-8B P1 also ran 4-bit on a laptop); "
           r"API$^{o}$: open weights, run via an API here. Prompts: P1 direct count, P2 step by step, P3 central "
           r"quarter-inch patch $\times$4, P3s sum of four diagonal quarter-inch patches, P4/P4q pointing along a "
           r"guide line (whole inch / quarter inch). "
           r"\ours\ vs GPT-6.1-sol P1, paired over the same 26 counts: "
           + f"$\\Delta$MAPE ${pr.dMAPE:+.1f}$ points (95\\% CI ${pr.dMAPE_lo:+.1f}$ to ${pr.dMAPE_hi:+.1f}$), "
           + f"Wilcoxon $p={pr.p_wilcoxon_abs_pct_err_26:.2f}$ (per count) / ${pr.p_wilcoxon_image_mean_pct_err_13:.2f}$ (per image): "
           + (r"not significant at $n{=}13$.}" if pr.p_wilcoxon_abs_pct_err_26 > 0.05 and pr.dMAPE_lo < 0 < pr.dMAPE_hi
              else r"see \texttt{tables/legacy13\_paired.csv}.}"))
    lines += [cap, r"\label{tab:legacy_unified}", r"\end{table*}"]
    open(path_full, "w").write("\n".join(lines) + "\n")

    # compact drop-in (one column), key rows only
    # FFT max-peak and lattice-only rows are left out of the compact table for space (both stay in the full table, Tab. S3)
    keep = ["ref:loo_mean", "2019:doc3", "est:gabor", "est:ours",
            "cnn:synth_r18", "countgd:CG-B-yarn-ex3", "countgd:CG-W-ex3-lat",
            "vlm:molmo2_8b_bf16|P1|r0", "vlm:molmo2_8b_bf16|P4q|r0", "ft:molmo2_4b_synthft|P1",
            "vlm:gpt-6.1-sol|P1|r0", "vlm:gemini-3.8-flash|P1|r0",
            "vlm:qwen3.8-27b|P1|r0", "vlm:gemma-4-26b-a4b|P1|r0"]
    # Molmo2-8B P1 and Gemma-4 rows are also kept only in the full table (Tab. S3), for space
    keep = [k for k in keep if k not in ("vlm:molmo2_8b_bf16|P1|r0", "vlm:gemma-4-26b-a4b|P1|r0")]
    # the legacy algorithm is not in the compact table (full table only)
    keep = [k for k in keep if k != "2019:doc3"]
    # Qwen3.8-Max-Prime and Claude Fable rows stay in the full table (Tab. S3) for space; the pseudo-label CNN is added
    keep.insert(keep.index("cnn:synth_r18") + 1, "aug:V4_b0")
    a = ax.set_index("key")
    L = [r"% Generated by paper/tools/legacy_unified.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Method & MAE & MAPE [95\% CI] & $\le$5\% & $\le$10\% & Off. \\", r"\midrule"]
    seps = {"est:gabor", "cnn:synth_r18", "vlm:gpt-6.1-sol|P1|r0"}
    special = {"countgd:CG-B-yarn-ex3": "CountGD, strips (primary)",
               "countgd:CG-W-ex3-lat": r"CountGD, cells $\to$ lattice$^b$",
               "vlm:molmo2_8b_bf16|P1|r0": "Molmo2-8B, P1",
               "vlm:molmo2_8b_bf16|P4q|r0": r"Molmo2-8B, P4q (pointing)",
               "ft:molmo2_4b_synthft|P1": r"Molmo2-4B + LoRA, P1$^c$",  # footnotes a, b, c in order
               "2019:testpy": r"2019 RGB irregularity$^a$",
               "2019:doc3": "2019 dot map + transitions",
               "cnn:synth_r18": "CNN, synthetic only",
               "est:ours": r"\ours (lattice + fusion)",
               "vlm:gemma-4-26b-a4b|P1|r0": "Gemma-4-26B (open), P1",
               "vlm:qwen3.8-max-prime|P1|r0": "Qwen3.8-Max-Prime, P1$^c$",
               "aug:V4_b0": r"CNN, + pseudo-label FT$^a$"}
    for k in keep:
        if k not in a.index:
            continue
        r = a.loc[k]
        if k in seps:
            L.append(r"\midrule")
        name = special.get(k, r.method_tex + ("" if r.group != "vlm" else f", {r.prompt}"))
        n = int(r.n_counts)
        cells = [name, f1(r.MAE), f"{f1(r.MAPE)} [{f1(r.MAPE_lo, 0)}, {f1(r.MAPE_hi, 0)}]",
                 f"{int(r.n_w5)}" + ("" if n == 26 else f"/{n}"), f"{int(r.n_w10)}" + ("" if n == 26 else f"/{n}"),
                 foff(r.offline)]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Held-out legacy test\LEG: 13 hand-counted windows, 26 counts, one crop rule, axis-matched. "
          r"MAE in threads per window side; MAPE in \% with a bootstrap 95\% CI over images; $\le$5/10\%: counts out of 26. "
          r"Off.: offline on a laptop; GPU: RTX 4090; API$^{o}$: open weights via an API. "
          r"$^a$EfficientNet-B0, synthetic pre-training, fine-tuned on \ours's confident counts of 28 scenes. "
          r"$^b$Post hoc: best of 18 configurations on these images. $^c$LoRA-tuned on synthetic crops. "
          r"CountGD strips: detections counted in a 1-inch strip two yarn spacings tall (rule fixed before the run); "
          r"cells $\to$ lattice: the yarn pitch from detections over the whole crop. P4q: Molmo2-8B points at the yarns crossing a guide line on the central "
          r"quarter inch, counted $\times$4. All methods, cost and latency: Tab.~S2.}",
          r"\label{tab:legacy}", r"\end{table}"]
    open(path_main, "w").write("\n".join(L) + "\n")


# ============================================================================ main
def main():
    gt = load_gt()
    methods = [loo_mean(gt)] + doc3() + classical() + cnn() + countgd() + vlms() + molmo4b_ft() + cnn_aug()
    by_key = {m["key"]: m for m in methods}
    rows, counts = [], []
    for m in methods:
        for assign in ("axis", "sorted"):
            if m["weft_only"] and assign == "sorted":
                continue
            s = score(m, gt, assign)
            if s is None:
                continue
            r, d = s
            rows.append(dict(group=m["group"], method=m["method"], prompt=m["prompt"], key=m["key"],
                             repeat=m["repeat"], assign=assign, crop_rule="canonical vlm/crops/legacy PNG",
                             input_px=m["input_px"], **r, usd_per_img=m["usd_per_img"], s_per_img=m["s_per_img"],
                             latency_hw=m["latency_hw"], offline=m["offline"], in_tex=m["in_tex"], note=m["note"]))
            d = d.assign(key=m["key"], method=m["method"], prompt=m["prompt"], assign=assign)
            counts.append(d)
    T = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    T.round({c: 3 for c in T.columns if c != "usd_per_img"} | {"usd_per_img": 5}).to_csv(
        os.path.join(TAB, "legacy13_unified.csv"), index=False)
    pd.concat(counts).round(3).to_csv(os.path.join(TAB, "legacy13_unified_per_count.csv"), index=False)

    # paired comparisons
    ours = by_key["est:ours"]
    g_reps = [by_key[f"vlm:gpt-6.1-sol|P1|r{r}"] for r in (0, 1, 2) if f"vlm:gpt-6.1-sol|P1|r{r}" in by_key]
    gmean = mean_of_repeats(g_reps, "vlm:gpt-6.1-sol|P1|mean3", "GPT-6.1-sol P1 mean of 3 repeats")
    comps = [("ours vs GPT-6.1-sol P1", ours, g_reps[0])]
    comps += [(f"ours vs GPT-6.1-sol P1 repeat {i}", ours, g) for i, g in enumerate(g_reps) if i > 0]
    comps += [("ours vs GPT-6.1-sol P1 mean of 3 repeats", ours, gmean),
              ("ours vs GPT-6.1-sol P3s", ours, by_key["vlm:gpt-6.1-sol|P3s|r0"]),
              ("ours vs Gemini 3.8 Flash P3", ours, by_key["vlm:gemini-3.8-flash|P3|r0"]),
              ("ours vs log-Gabor", ours, by_key["est:gabor"]),
              ("ours vs lattice only", ours, by_key["est:lattice"]),
              ("ours vs CNN synthetic", ours, by_key["cnn:synth_r18"]),
              ("ours vs leave-one-out mean", ours, by_key["ref:loo_mean"]),
              ("log-Gabor vs GPT-6.1-sol P1", by_key["est:gabor"], g_reps[0])]
    # best specialist variants vs ours and the synthetic-LoRA Molmo2-4B
    comps += [("ours vs Molmo2-8B bf16 P4q", ours, by_key["vlm:molmo2_8b_bf16|P4q|r0"]),
              ("ours vs CountGD cells->lattice (post hoc)", ours, by_key["countgd:CG-W-ex3-lat"])]
    if "ft:molmo2_4b_synthft|P1" in by_key:
        ft1 = by_key["ft:molmo2_4b_synthft|P1"]
        comps += [("ours vs Molmo2-4B synthetic LoRA P1", ours, ft1),
                  ("Molmo2-4B synthetic LoRA P1 vs base P1", ft1, by_key["ft:molmo2_4b_base|P1"]),
                  ("Molmo2-4B synthetic LoRA P1 vs Molmo2-8B bf16 P4q", ft1, by_key["vlm:molmo2_8b_bf16|P4q|r0"]),
                  ("Molmo2-4B synthetic LoRA P1 vs CNN synthetic", ft1, by_key["cnn:synth_r18"]),
                  ("Molmo2-4B synthetic LoRA P1 vs GPT-6.1-sol P1", ft1, g_reps[0]),
                  ("Molmo2-4B synthetic LoRA P1 vs leave-one-out mean", ft1, by_key["ref:loo_mean"])]
    # pseudo-label fine-tuned CNN (cnn/RESULTS_AUG.md), appended so earlier rows keep their order
    if "aug:V4_b0" in by_key:
        comps += [("ours vs CNN pseudo-label FT B0 (V4)", ours, by_key["aug:V4_b0"]),
                  ("CNN pseudo-label FT B0 (V4) vs CNN synthetic", by_key["aug:V4_b0"], by_key["cnn:synth_r18"]),
                  ("CNN pseudo-label FT B0 (V4) vs GPT-6.1-sol P1", by_key["aug:V4_b0"], g_reps[0])]
    PR = [paired(a, b, gt, assign, lab) for lab, a, b in comps for assign in ("axis", "sorted")]
    for sub in ("noformula", "plain"):
        for assign in ("axis", "sorted"):
            PR.append(paired(ours, g_reps[0], gt, assign, "ours vs GPT-6.1-sol P1", subset=sub))
            PR.append(paired(ours, gmean, gt, assign, "ours vs GPT-6.1-sol P1 mean of 3 repeats", subset=sub))
    PR = pd.DataFrame(PR)
    PR.round(4).to_csv(os.path.join(TAB, "legacy13_paired.csv"), index=False)

    CS = crop_sensitivity(gt, by_key)
    CS.round(3).to_csv(os.path.join(TAB, "legacy13_crop_sensitivity.csv"), index=False)

    write_tex(T, PR, os.path.join(TAB, "legacy13_unified.tex"), os.path.join(TAB, "legacy13_unified_main.tex"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 40)
    show = T[(T["assign"] == "axis")][["group", "method", "prompt", "repeat", "n_counts", "MAE", "MAPE", "MAPE_lo",
                                    "MAPE_hi", "n_w5", "n_w10", "MAE_plain", "MAE_noformula", "usd_per_img",
                                    "s_per_img"]]
    print(show.round(2).to_string(index=False))
    print()
    print(T[(T["assign"] == "sorted") & T.key.isin(["est:ours", "est:gabor", "vlm:gpt-6.1-sol|P1|r0", "cnn:synth_r18"])][
        ["method", "prompt", "n_counts", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "n_w5", "n_w10"]].round(2).to_string(index=False))
    print()
    print(PR[["comparison", "assign", "subset", "n_counts", "MAE_A", "MAE_B", "dMAE", "dMAE_lo", "dMAE_hi", "MAPE_A", "MAPE_B",
              "dMAPE", "dMAPE_lo", "dMAPE_hi", "p_boot_MAPE", "p_wilcoxon_abs_err_26", "p_wilcoxon_abs_pct_err_26",
              "p_wilcoxon_image_mean_pct_err_13", "counts_A_better", "counts_B_better", "n_w10_A", "n_w10_B",
              "p_mcnemar_w10"]].round(3).to_string(index=False))
    print()
    print(CS.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

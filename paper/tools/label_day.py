#!/usr/bin/env python
"""label_day.py -- ONE-COMMAND LABEL ANALYSIS for the WVAQ 2027 paper (PickGlass: 150 files = 76 scenes).

Given the annotators' exports (tool JSON + CSV downloads, or the disk autosave in
annotation_tool/data/saved/), runs in order, logs every step and STOPS at the first error:

  1 validate      merge the exports per annotator (label_day_merge.js = the tool's own core.js merge), check
                  done records, annotator names, blind/IAA status, cleared lines (-> missing), ids, crop sizes
                  (another kit's export, e.g. the screening test, is refused), partly counted lines (policy below),
                  proposal-free records, plausibility; coverage report against the pre-declared go/no-go (THR)
  2 propagate     annotation_tool/propagate_scene_labels.py -> per-scene + per-file labels
  3 agreement     --protocol dual: inter-annotator agreement over every scene finished by both annotators (MAE,
                  Bland-Altman, ICC(2,1), weave kappa, median |diff|%; also badge / no-badge / without the guide
                  scenes) = the human noise floor + the post-hoc proposal check per annotator;
                  --protocol single: post-hoc proposal check (machine dots never shown vs the human dots, per line)
                  -> supp Tab. S4 (tables/supp_labelcheck_rows.tex);
                  --protocol iaa: annotation_tool/analyze.py (IAA + anchoring), human-human noise floor
  4 score         tools/score150.py for every method (scene unit primary, per-file secondary, fabric-group
                  sensitivity, --blind-only check in the iaa protocol only), tools/quality150.py (confidence vs error),
                  sensitivity subsets (guide / badge scenes, proposal-free records, partial lines rescaled), and the further
                  analyses: paired tests, resolution classes, cascade frozen gate, ruler, weave, the 13
                  "unknown"-weave scenes, the 39-scene pseudo-label row, distortion basis, worst scenes
  5 cnn           (optional, --cnn) real-label scene-grouped 5-fold cnn/train_cv.py on gpu-server (or locally), with
                  and without --extra-aug1016 and --augment whitelist; poll, pull back, score (re-runs step 4)
  6 tables        tables/main150_rows.tex (Tab. 1 rows, TAB1_SPEC selection) + every analysis CSV, Tab. S4 /
                  Tab. S6 / all-methods fragments for the supplement, figs/fig_coverage.pdf
  7 compile       main.pdf and supp.pdf with Tectonic; page/body check; tools/validate_tex.py
  8 report        LABEL_DAY_REPORT.md: go/no-go verdicts, numbers (and what changed since the last run), sentences
                  that need rewording, TBD/TODO markers still present

MODES (one is required)
  --out-root DIR   DRY RUN. Everything is written under DIR/runs/<stamp>/ (a fresh copy of the paper sources,
                   propagated labels, staging, CNN runs, PDFs, report). Nothing in the project is written.
  --real           Writes the real outputs: annotation_tool/propagated/, paper/tables/*, paper/figs/fig_coverage.*,
                   paper/main.pdf + supp.pdf, paper/LABEL_DAY_REPORT.md; logs and backups of every overwritten
                   file in <repo root>/label_day/<stamp>/. Refused for labels that look fake, and for a NO-GO scene
                   count unless --allow-partial.

PROTOCOLS (--protocol, default auto = single when exactly one annotator has finished records; dual when two or more
          have finished records and no finished record carries proposal dots; iaa otherwise, with a warning)
  dual    (two annotators, A1 + A2) both count every scene from scratch in the annotator
          kit (proposals hidden): everything single does (proposal-free k of N incl. --proposals-hidden-at, the
          partial-line policy, the post-hoc proposal check, now per annotator) plus inter-annotator agreement over
          every scene finished by both (MAE h/v, Bland-Altman bias + 95% LoA, ICC(2,1), within 5/10%, median |diff|%,
          weave kappa; also on badge / no-badge scenes, without the 4 guide scenes, proven proposal-free records);
          go/no-go = scene coverage + screening (both annotators) + human-human median |diff|% (GO <= 5%, REDUCED
          <= 10%); Tab. 1 row 'Human vs human' (annotator 2 scored against annotator 1, n = scenes counted by both);
          reference = the mean of the two (score150 gt_table), each annotator alone as a sensitivity reference;
          supp Tab. S4 = tables/supp_labelcheck_rows.tex with the agreement rows; no anchoring rule, no blind-only
          run, no blind NO-GO (the 20-scene IAA subset has no role)
  single  ONE external annotator counts every scene from scratch in the annotator kit (proposals
          hidden): go/no-go on scene coverage only, no human-vs-human Tab. 1 row, no --blind-only run, no
          anchoring; supp Tab. S4 = screening constants + the post-hoc proposal check + ours MAPE without the 4
          guide-example scenes + median minutes per photo; 'proposal-free records: k of N' from n_prop / dot
          origins / proposal_sig / creation time (core.js sets proposals_seen = true even when it inserts 0 dots)
  iaa     (the first protocol) two annotators, 20 blind IAA scenes, anchoring rule, --blind-only check (unchanged)

PARTLY COUNTED LINES (the tool's enforce-done rule: a line may cover < 75% of the crop if 'partially occluded' is ticked
or Notes say why; span = (last dot - first dot) / crop side, as the tool computes it)
  headline     only records whose two lines both span >= 0.75 (a record with a shorter line is dropped from the
               scoring files like a cleared line: the per-scene metrics need both axes)
  sensitivity  such a record is kept only if 'partially occluded' or Notes, with the short line scored as
               (n_dots - 1) / span_in (span_in = (last - first) / cos(angle) / crop side) in the SEPARATE columns
               <ax>_tpi_rescaled of merged/scoring_partial/*.csv; reported as the 'sensitivity: partial lines
               rescaled' rows of main150_subsets.csv (never in Tab. 1). Lines spanning 0.75-0.95 are only warned about.

EXAMPLES (from paper/; label_day.sh uses the durable python in <repo root>/.tools/venv)
  tools/label_day.sh --out-root <tmpdir>/ld                                   # plumbing check on the autosave
  tools/label_day.sh --out-root <tmpdir>/ld annotations_A1_*.json annotations_A1_*.csv
  tools/label_day.sh --out-root <tmpdir>/ld --protocol dual <A1 exports> <A2 exports>   # two annotators
  tools/label_day.sh --real --protocol dual <A1 exports> <A2 exports>   # real run (pass the files explicitly)
  tools/label_day.sh --real --cnn launch          # final labels; CNN CV launched on gpu-server, not awaited
  tools/label_day.sh --real --cnn-collect latest  # later: pull the CNN runs, re-score, rebuild, report
  tools/label_day.sh --real --cnn wait            # all in one go (polls gpu-server until the CV runs finish)
  tools/label_day.sh --out-root DIR --stop-after 4   # validate .. score only

Nothing is deleted. Inputs are only read. The frozen counter and the frozen cascade decisions are md5-checked.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # importing score150 / analyze / propagate must not write __pycache__ anywhere

import argparse
import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import traceback

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as _e:   # pragma: no cover
    sys.exit(f"label_day.py needs numpy<2, pandas and scipy (use <repo root>/.tools/venv/bin/python, e.g. via "
             f"tools/label_day.sh): {_e}")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TOOL = os.path.join(ROOT, "annotation_tool")
MANIFEST = os.path.join(TOOL, "data", "manifest.json")
SAVED = os.path.join(TOOL, "data", "saved")

SCRATCH = os.environ.get("LABEL_DAY_SCRATCH", os.environ.get("PICKGLASS_SCRATCH") or os.path.join(ROOT, "scratch"))
VENVS = os.environ.get("PICKGLASS_VENVS", SCRATCH)   # holds venv_cnn/ (or set LABEL_DAY_CNN_PY)
# durable toolchain outside /tmp (macOS empties idle /tmp dirs after ~3 days):
# .tools/venv (numpy 1.26.4, pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2, opencv-headless, pillow, pypdf 6.19.0),
# .tools/bin/tectonic (0.17.0) + .tools/tectonic_cache. The LABEL_DAY_* environment variables still override.
TOOLS = os.path.join(ROOT, ".tools")
DEF_TECTONIC = os.environ.get("LABEL_DAY_TECTONIC", os.path.join(TOOLS, "bin", "tectonic"))
DEF_TECTONIC_CACHE = os.environ.get("LABEL_DAY_TECTONIC_CACHE", os.path.join(TOOLS, "tectonic_cache"))
DEF_PDF_PY = os.environ.get("LABEL_DAY_PDF_PY", os.path.join(TOOLS, "venv", "bin", "python"))
DEF_CNN_PY = os.environ.get("LABEL_DAY_CNN_PY", os.path.join(VENVS, "venv_cnn", "bin", "python"))

# Frozen artefacts: checked, never written.
FROZEN = {
    "methods/results/counting_frozen_final.py": "3d0520d6bec8a016543ce7e5b353d784",
    "paper/tables/cascade_150_decisions.csv": "57d49a4d4304bc395ed608894009f683",
    # pre-registered combined-v1 predictions (PREREG_combined_v1.json: SHA-256 c3849108f2bf03a3...)
    "methods/results/combined_v1_150_wide.csv": "68c563f629330fc1943cc125f9ddf82c",
}
SCORER_MD5 = {"paper/tools/score150.py": "5879ef44698de7b2a090e8bbee973625"}   # disclosed in supp S10

# Pre-declared go/no-go after annotation, in scenes. Single protocol: scenes with a finished count of the one
# annotator GO >= 51, REDUCED 25-50, NO-GO < 25; that annotator vs the legacy hand counts (screening) GO <= 5%, REDUCED 5-10%.
# Dual protocol (two annotators): the same scene and screening rows (screening = max over both annotators) plus
# the human-human median |diff|% over every scene finished by both: GO <= hh_go, REDUCED <= hh_reduced, else NO-GO.
THR = dict(scenes_go=51, scenes_reduced=25, blind_go=15, blind_reduced=10, hh_go=5.0, hh_reduced=10.0,
           anchoring_gap=3.0, screen_go=5.0, screen_reduced=10.0)
# Screening on two legacy crops, per annotator: (crop, raw answer h x v, legacy hand count h x v) (supp Tab. S4 top rows,
# go/no-go screening row). Annotator order = annotator 1, 2 in the dual protocol (Tab. 1 human row: annotator 2 vs annotator 1).
SCREENING = {"FL_SA": [("r2", (97, 69), (99, 69)), ("a5", (46, 50), (47, 49))],
             "FL_NG": [("r2", (99, 68), (99, 69)), ("a5", (48, 49), (47, 49))]}
SCREENING_DEFAULT = "FL_SA"   # single protocol under another name (e.g. a FAKE test annotator): A1's constants
# Annotator kit: the time the machine proposals were removed from the deployed manifest, plus a 10 min upload margin
# (freelancer/web/annotate/data/manifest.json, proposals_hidden); a non-IAA record CREATED before this instant may have
# shown proposal dots even if its fields no longer say so (core.js refreshes untouched records).
PROPOSALS_HIDDEN_AT = "2026-10-11T17:00:00Z"
SPAN_MIN = 0.75    # tool enforce-done rule (app.js doneProblems): below it only with 'partially occluded' or Notes
SPAN_WARN = 0.95   # report lines spanning less than this (kept in the headline)

# Every method of the paper's comparison (score150.py already loads ours + spectral baselines, the 7 P1 VLMs and
# the synthetic CNN by itself). Keys None = every key of the file. All are required (a missing one stops step 4).
EXTRA_SOURCES = [
    ("gpu_results/countgd/outputs/results/rect150_predictions_wide.csv", ["countgd", "countgd@CG-W-ex3-lat"]),
    ("gpu_results/scored/molmo2_rect150_predictions_wide.csv", ["molmo2_8b_bf16", "molmo2_8b_bf16@P4"]),
    ("ft_results/results/rect150_predictions_wide.csv",
     ["molmo2_4b_base@P1", "molmo2_4b_synthft@P1", "molmo2_4b_synthft@P4s"]),
    ("vlm/scaffold/results/rect150_S3_predictions_wide.csv",
     ["gpt-6.1-sol@S3", "gemini-3.8-flash@S3", "qwen3.8-27b@S3", "s4-peaks"]),
    ("paper/tables/src/extra150_wide.csv", ["pg-lattice@ruler", "cascade-gate"]),
    ("cnn/runs_aug/report/real150_aug_wide_heldout.csv", None),   # V3-V5 blank on their training scenes
    # The pre-registered combined counter (methods/results/PREREG_combined_v1.json, frozen
    # before any label) passed its held-out synthetic test (P5 72.4% vs ours 70.3% / log-Gabor
    # 59.5%; dP5 +2.05 [+1.35, +2.78] and +12.89 [+10.24, +15.61]; methods/results/combined_v1_eval_synth.json), so its
    # primary variant enters Tab. 1 (PREREG_combined_v1.json key label_day_plan); combined-v1-mape stays a sensitivity analysis (not scored)
    ("methods/results/combined_v1_150_wide.csv", ["combined-v1"]),
]
P_VLM_LONG = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_long.csv")
P_HELDOUT = os.path.join(ROOT, "cnn", "runs_aug", "report", "real150_aug_wide_heldout.csv")
P_PSEUDO = os.path.join(ROOT, "cnn", "runs_aug", "pseudo_labels.csv")
P_DECISIONS = os.path.join(PAPER, "tables", "cascade_150_decisions.csv")
P_STRATA = os.path.join(PAPER, "tables", "strata150.csv")
P_GROUPS = os.path.join(PAPER, "tables", "fabric_groups_v2.csv")   # 44 groups, tools/fabric_groups_v2.py
PSEUDO_KEY = "cnn-aug-V4-b0"
RESHOOT_SCENES = (8, 9, 15, 38, 45, 52, 61, 103, 110)   # held-out scenes that re-shoot a training scene's fabric patch (tables/fabric_groups_v2.json cnn_split.affected_heldout) -> the 39-scene check
GUIDE_EXAMPLE_SCENES = (27, 65, 93, 110)   # the annotator guide's figures show the machine dots on these reps
GUIDE_SUBSET = "without the 4 guide-example scenes (27, 65, 93, 110)"

# Method names as score150.py writes them (main150.csv "method")
OURS = "Ours (lattice + fusion)"
GPT = "GPT-6.1-sol, P1"
FLASH = "Gemini 3.8 Flash, P1"
CNN0 = "CNN, synthetic only (zero-shot)"
LOO = "Leave-one-out mean (reference)"
CV_PRIMARY = "cnn-cv-base-resnet18-synth_ft"   # pre-declared "CNN, 5-fold FT" row

# Tab. 1: the 16 fallback rows, one method each, in the order of the fallback table;
# "FFT / log-Gabor" -> the log-Gabor bank, "CountGD strips / cells" and "Molmo2-8B / 4B+LoRA" -> two rows each
# (as in Tab. 2), "Qwen3.8-27B / Gemma-4" -> Qwen3.8-27B (Gemma-4 to the supplement). Override: --tab1-spec.
TAB1_SPEC = [
    ["@human", r"Human \vs human"],   # dual: annotator 2 vs annotator 1 on the scenes counted by both; iaa: + " (blind)"
    [LOO, "Leave-one-out mean"],
    "midrule",
    ["Log-Gabor bank", "Log-Gabor bank"],
    [OURS, r"\ours (ours)"],
    ["combined-v1", r"Combined (\ours + log-Gabor)"],   # pre-registered combined-v1 (held-out test passed)
    ["pg-lattice@ruler", r"\ours, ruler crop"],
    ["cascade-gate", r"Cascade (\ours/GPT)"],
    "midrule",
    [CNN0, "CNN, synthetic (zero-shot)"],
    [PSEUDO_KEY, r"CNN, + pseudo-labels$^\ast$"],
    [CV_PRIMARY, "CNN, 5-fold FT"],
    ["countgd", "CountGD, strips"],
    ["countgd@CG-W-ex3-lat", r"CountGD, cells $\to$ lattice"],
    ["molmo2_8b_bf16@P4", "Molmo2-8B, P4"],
    ["molmo2_4b_synthft@P1", "Molmo2-4B + LoRA, P1"],
    "midrule",
    [GPT, "GPT-6.1-sol"],
    [FLASH, "Gemini 3.8 Flash"],
    ["Qwen3.8-Max-Prime, P1", "Qwen3.8-Max-Prime"],
    ["Qwen3.8-27B (open), P1", "Qwen3.8-27B (open)"],
]
KEY_METHODS = [OURS, "pg-lattice@ruler", "cascade-gate", GPT, FLASH, CNN0, PSEUDO_KEY, CV_PRIMARY, "Log-Gabor bank",
               "combined-v1"]

# CNN cross-validation: scene-grouped 5-fold, real labels
CNN_CONFIGS = {"base": [], "aug1016": ["--extra-aug1016"], "wl": ["--augment", "whitelist"],
               "aug1016_wl": ["--extra-aug1016", "--augment", "whitelist"]}
CNN_SYNTH_LOCAL = os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso", "model.pt")
CNN_SYNTH_MD5 = "dad61b74874196b49bc4e32fab498438"
SSH_HOST = os.environ.get("PICKGLASS_GPU_HOST", "gpu-server")
REMOTE_FW = (os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass"))               # relative to $HOME on gpu-server; never write outside it
REMOTE_CA = REMOTE_FW + "/cnn_aug"                   # env.sh (all caches inside), venv
REMOTE_LD = REMOTE_CA + "/labelday"                  # this tool's own tree (code copy, labels, runs)
REMOTE_SYNTH = REMOTE_CA + "/wvaq/cnn/runs/synth_r18_v2mix_aniso/model.pt"   # existing, read-only, md5-checked
WEAVE_CLASSES = ["plain", "twill", "satin", "other"]   # cnn_common.WEAVE_CLASSES


class StepError(RuntimeError):
    pass


# ============================================================================ small utilities
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fnum(x, d=1, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s


def tex_signed(x, d=1):
    if x is None or not math.isfinite(x):
        return "--"
    return f"${x:+.{d}f}$"


def verdict(x, go, reduced, higher_is_better=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NO-GO"
    if higher_is_better:
        return "GO" if x >= go else ("REDUCED" if x >= reduced else "NO-GO")
    return "GO" if x <= go else ("REDUCED" if x <= reduced else "NO-GO")


def proposal_free(r):
    """True when the record itself shows that no machine dot was ever inserted. The single/dual-protocol kit has proposal = null for
    every image, but core.js insertProposals() still sets proposals_seen = true when a non-blind record is created
    (0 dots, method null), so the flag proves nothing: n_prop, the dot origins, proposal_method and proposal_sig do.
    (A record created before the proposals were hidden is caught separately: opened_before_hide().)"""
    if not r.get("proposals_seen"):
        return True
    sig = str(r.get("proposal_sig") or "")
    return (not r.get("proposal_method") and not (r.get("n_prop_h") or 0) and not (r.get("n_prop_v") or 0)
            and not any(r.get("h_orig") or []) and not any(r.get("v_orig") or [])
            and (not sig or sig.startswith("none:")))


def opened_before_hide(r, hidden_at, iaa):
    """Annotator kit: a non-IAA record that was not blind from the start and was CREATED before the proposals were
    hidden may have shown proposal dots, even when its fields were later refreshed (core.js refreshProposals
    re-initialises an untouched record from the new, empty proposal but keeps 'created')."""
    if int(r["id"]) in iaa or r.get("blind_initial"):
        return False
    try:
        return pd.Timestamp(r.get("created")) < pd.Timestamp(hidden_at)
    except (TypeError, ValueError):
        return False


def line_geom(r, line):
    """Per guide line: n dots, span = (last - first) / crop side (exactly as app.js doneProblems computes it), the
    median gap (px), and the density over the counted part, (n - 1) / span_in with span_in = (last - first) /
    cos(angle) / crop side (dot positions are line parameters; cos converts them to length along a turned line)."""
    d = sorted(float(x) for x in r.get(line + "_dots") or [])
    S = float(r.get("rect_size") or 0)
    if len(d) < 2 or S <= 0:
        return dict(n=len(d), span=0.0, gap=None, tpi_rescaled=None)
    a = float(r.get(line + "_angle_deg") or 0)
    span_in = (d[-1] - d[0]) / math.cos(math.radians(a)) / S
    gap = float(np.median(np.diff(d)))
    return dict(n=len(d), span=(d[-1] - d[0]) / S, gap=gap,
                tpi_rescaled=(len(d) - 1) / span_in if span_in > 0 else None)


def line_span(r, line):
    """Share of the crop side covered by a line's dots, first to last (the tool's enforce-done rule asks >= 75%)."""
    return line_geom(r, line)["span"]


def match_1d(a, b, tol):
    """Maximum one-to-one matching of two SORTED 1-D point sets with |a_i - b_j| <= tol; among the maximum matchings,
    the one with the smallest total distance. Order-preserving DP (in 1-D an optimal matching never crosses).
    Returns [(i, j), ...]."""
    n, m = len(a), len(b)
    cnt = np.zeros((n + 1, m + 1), int)
    cost = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            best_c, best_d = cnt[i - 1, j], cost[i - 1, j]
            if cnt[i, j - 1] > best_c or (cnt[i, j - 1] == best_c and cost[i, j - 1] < best_d):
                best_c, best_d = cnt[i, j - 1], cost[i, j - 1]
            d = abs(ai - b[j - 1])
            if d <= tol:
                c2, d2 = cnt[i - 1, j - 1] + 1, cost[i - 1, j - 1] + d
                if c2 > best_c or (c2 == best_c and d2 < best_d):
                    best_c, best_d = c2, d2
            cnt[i, j], cost[i, j] = best_c, best_d
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        d = abs(a[i - 1] - b[j - 1])
        if d <= tol and cnt[i, j] == cnt[i - 1, j - 1] + 1 and abs(cost[i, j] - (cost[i - 1, j - 1] + d)) < 1e-9:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif cnt[i, j] == cnt[i - 1, j] and abs(cost[i, j] - cost[i - 1, j]) < 1e-9:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def looks_fake(name):
    """Annotator names of test data (FAKE_A, dummy, smoke, test...): refused by --real and by the gpu-server upload."""
    return bool(re.search(r"fake|dummy|smoke|^test|test$", str(name), re.I))


def annotator_order(names):
    """Annotator 1, 2, ...: the SCREENING order (A1, A2) first, then any other name alphabetically."""
    names = [n for n in names if n != "anonymous"] or list(names)
    return [n for n in SCREENING if n in names] + sorted(n for n in names if n not in SCREENING)


def screening_rows(names):
    """[(annotator, crop, answer, legacy hand count), ...] for the annotators of this run that have screening constants;
    A1's (SCREENING_DEFAULT) when none has (single protocol under another name, as before the dual protocol)."""
    have = [n for n in SCREENING if n in names] or [SCREENING_DEFAULT]
    return [(n, crop, ans, ref) for n in have for crop, ans, ref in SCREENING[n]]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        return {prefix.rstrip("."): json.dumps(d)} if len(json.dumps(d)) < 200 else {}
    else:
        out[prefix.rstrip(".")] = d
    return out


# ============================================================================ context
class Ctx:
    def __init__(self, a):
        self.a = a
        self.stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = "real" if a.real else "dry"
        if self.mode == "dry":
            self.root_out = os.path.abspath(os.path.expanduser(a.out_root))
            self.run_dir = os.path.join(self.root_out, "runs", self.stamp)
            self.paper = os.path.join(self.run_dir, "paper")
            self.prop_dir = os.path.join(self.run_dir, "annotation_tool_propagated")
            self.cnn_local = os.path.join(self.run_dir, "cnn_runs")
            self.work = os.path.join(self.run_dir, "work")
            self.history_glob = os.path.join(self.root_out, "runs", "*", "work", "numbers.json")
            self.state_root = self.root_out
        else:
            self.root_out = ROOT
            self.run_dir = os.path.join(ROOT, "label_day", self.stamp)
            self.paper = PAPER
            self.prop_dir = os.path.join(TOOL, "propagated")
            self.cnn_local = os.path.join(ROOT, "cnn", "runs")
            self.work = self.run_dir
            self.history_glob = os.path.join(ROOT, "label_day", "*", "numbers.json")
            self.state_root = os.path.join(ROOT, "label_day")
        self.tables = os.path.join(self.paper, "tables")
        self.figs = os.path.join(self.paper, "figs")
        self.stage = os.path.join(self.work, "stage")
        self.backup = os.path.join(self.work, "backup")
        self.N = {"meta": {}, "validation": {}, "coverage": {}, "agreement": {}, "score": {}, "cnn": {}, "build": {}}
        self.warnings = []
        self.written = []
        self.inputs = []
        self.scoring_csv = []
        self.scoring_json = []
        self.full_json = []
        self.cv_wides = []
        self.extra_specs = []
        self.steps_done = []
        self.failed = None
        self.logf = None
        self.headline = "all"
        self.protocol = "iaa"   # 'single' | 'dual' | 'iaa' (set in step 1)
        self.single = False   # single protocol: one annotator, every scene from scratch, no IAA (set in step 1)
        self.dual = False     # dual protocol: two annotators (A1 + A2), every scene from scratch (step 1)
        self.oct11 = False    # single or dual: the annotator kit with the proposals hidden (no blind/assisted split)
        self.ann_order = []   # annotator 1, 2, ... (annotator_order())
        self.partial_csv = []   # sensitivity label files (partly counted lines rescaled), written in step 1
        self.hint_ids = set()   # reps on which the annotator kit showed the 'check carefully' / 'take care' badge
        self.not_free = set()   # (annotator, id) of finished records not proven proposal-free (single/dual protocol)
        self.human_metrics = None   # Tab. 1 'Human vs human' row (score150.metrics), iaa / dual protocol

    # ---------------------------------------------------------------- logging
    def open_log(self):
        os.makedirs(self.work, exist_ok=True)
        self.logf = open(os.path.join(self.work, "label_day.log"), "a")

    def log(self, msg=""):
        print(msg, flush=True)
        if self.logf:
            self.logf.write(msg + "\n")
            self.logf.flush()

    def warn(self, msg):
        self.warnings.append(msg)
        self.log(f"  WARNING: {msg}")

    def env(self):
        e = dict(os.environ)
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        e["MPLCONFIGDIR"] = os.path.join(self.work, "mpl")
        os.makedirs(e["MPLCONFIGDIR"], exist_ok=True)
        # a fresh MPLCONFIGDIR makes every subprocess rebuild matplotlib's font cache (1-3 min each on a busy
        # machine): seed it with the prebuilt cache of the durable toolchain (read only; nothing in .tools is written)
        for f in glob.glob(os.path.join(TOOLS, "mplconfig", "fontlist-*.json")):
            dst = os.path.join(e["MPLCONFIGDIR"], os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copyfile(f, dst)
        return e

    def run(self, cmd, cwd=None, env=None, check=True, quiet=False, capture=None, timeout=None):
        """Run a command, stream its output to the console and the log; returns (rc, text)."""
        shown = " ".join(shlex.quote(str(c)) for c in cmd)
        self.log(f"  $ {shown}" + (f"   (cwd {cwd})" if cwd else ""))
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env or self.env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        t0 = time.time()
        for line in p.stdout:
            lines.append(line)
            if not quiet:
                self.log("  | " + line.rstrip("\n"))
            elif self.logf:
                self.logf.write("  | " + line)
            if timeout and time.time() - t0 > timeout:
                break
        rc = p.wait()
        text = "".join(lines)
        if capture:
            os.makedirs(os.path.dirname(capture), exist_ok=True)
            with open(capture, "w") as f:
                f.write(text)
        if check and rc != 0:
            raise StepError(f"command failed (exit {rc}): {shown}\n" + "".join(lines[-15:]))
        return rc, text

    # ---------------------------------------------------------------- writing outputs (backup first, never delete)
    def put(self, dest, text=None, src=None):
        dest = os.path.abspath(dest)
        self._guard(dest)
        if self.mode == "real" and os.path.exists(dest):
            rel = os.path.relpath(dest, ROOT)
            b = os.path.join(self.backup, rel)
            if not os.path.exists(b):
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copy2(dest, b)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".label_day_tmp"
        if src is not None:
            shutil.copyfile(src, tmp)
        else:
            with open(tmp, "w") as f:
                f.write(text)
        os.replace(tmp, dest)
        self.written.append(dest)

    def backup_existing(self, paths):
        for p in paths:
            if self.mode == "real" and os.path.exists(p):
                b = os.path.join(self.backup, os.path.relpath(p, ROOT))
                if not os.path.exists(b):
                    os.makedirs(os.path.dirname(b), exist_ok=True)
                    shutil.copy2(p, b)

    def _guard(self, dest):
        """Dry run: every write must stay under the run dir."""
        if self.mode == "dry" and not dest.startswith(self.run_dir + os.sep):
            raise StepError(f"refusing to write outside the dry-run directory: {dest}")


# ============================================================================ step 0: setup
def setup(ctx):
    a = ctx.a
    if a.real == bool(a.out_root):
        raise StepError("choose exactly one of --out-root DIR (dry run into a scratch copy) or --real")
    if ctx.mode == "dry":
        ro = ctx.root_out
        for forbidden in (PAPER, TOOL, os.path.join(ROOT, "cnn"), os.path.join(ROOT, "methods")):
            if ro == forbidden or ro.startswith(forbidden + os.sep):
                raise StepError(f"--out-root must not be inside {forbidden}")
        if os.path.realpath(ro).startswith(os.path.realpath((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")))):
            raise StepError("--out-root must not be inside fabric_notebooks")
    # inputs first (nothing is created before they pass): explicit exports (globs) or the disk autosave;
    # JSON first so a CSV copy never beats its JSON
    pats = a.exports or [os.path.join(SAVED, "*.json"), os.path.join(SAVED, "*.csv")]
    files = []
    for p in pats:
        hits = sorted(glob.glob(os.path.expanduser(p)))
        if not hits and not any(ch in p for ch in "*?["):
            raise StepError(f"export not found: {p}")
        files += [os.path.abspath(h) for h in hits if os.path.isfile(h)]
    files = list(dict.fromkeys(files))
    files = [f for f in files if f.lower().endswith(".json")] + [f for f in files if f.lower().endswith(".csv")]
    other = [f for f in files if not f.lower().endswith((".json", ".csv"))]
    if other:
        raise StepError(f"exports must be .json or .csv: {other}")
    if not files:
        raise StepError(f"no annotation exports found ({pats})")
    if ctx.mode == "real":
        bad = [f for f in files if re.search(r"fake|dummy|smoke", os.path.basename(f), re.I) or
               os.path.realpath(f).startswith(os.path.realpath(SCRATCH))]
        if bad:
            raise StepError(f"--real refuses files that look like test labels: {bad}")
    os.makedirs(ctx.work, exist_ok=True)
    ctx.open_log()
    ctx.log(f"label_day.py {ctx.stamp}  mode={ctx.mode.upper()}  paper={ctx.paper}")
    ctx.log(f"  work dir: {ctx.work}")
    ctx.inputs = files
    ctx.N["meta"].update(stamp=ctx.stamp, mode=ctx.mode, inputs=[dict(path=f, md5=md5(f), bytes=os.path.getsize(f))
                                                                  for f in files])
    ctx.log(f"  {len(files)} export file(s):")
    for f in files:
        ctx.log(f"    {f}")
    # frozen artefacts and scorer
    md = {}
    for rel, want in {**FROZEN, **SCORER_MD5}.items():
        p = os.path.join(ROOT, rel)
        got = md5(p) if os.path.exists(p) else None
        md[rel] = got
        ok = got == want
        ctx.log(f"  md5 {rel}: {got} {'OK' if ok else '(expected ' + want + ')'}")
        if not ok:
            if rel in FROZEN:
                raise StepError(f"frozen artefact changed: {rel} md5 {got}, expected {want}")
            ctx.warn(f"{rel} md5 {got} differs from the disclosed {want} (supp S10 must then be updated)")
    for rel in ("annotation_tool/propagate_scene_labels.py", "annotation_tool/analyze.py", "paper/tools/quality150.py",
                "annotation_tool/core.js", "annotation_tool/data/manifest.json"):
        md[rel] = md5(os.path.join(ROOT, rel))
    ctx.N["meta"]["md5"] = md
    # tools
    ctx.node = shutil.which("node") or next((p for p in ("/usr/local/bin/node", "/opt/homebrew/bin/node")
                                              if os.path.exists(p)), None)
    if not ctx.node:
        raise StepError("node not found (needed for label_day_merge.js, the tool's own merge/export code)")
    # dry run: a fresh copy of the paper sources
    if ctx.mode == "dry":
        os.makedirs(ctx.paper, exist_ok=True)
        for f in ("main.tex", "supp.tex", "preamble.tex", "wacv.sty", "ieeenat_fullname.bst", "paper_refs.bib"):
            shutil.copy2(os.path.join(PAPER, f), os.path.join(ctx.paper, f))
        for d in ("sections", "tables", "figs"):
            shutil.copytree(os.path.join(PAPER, d), os.path.join(ctx.paper, d))
        os.makedirs(os.path.join(ctx.paper, "tools"), exist_ok=True)
        shutil.copy2(os.path.join(HERE, "validate_tex.py"), os.path.join(ctx.paper, "tools", "validate_tex.py"))
        ctx.log(f"  copied the paper sources to {ctx.paper}")
    os.makedirs(ctx.state_root, exist_ok=True)
    with open(os.path.join(ctx.state_root, "LATEST_RUN.txt"), "w") as f:   # pointer only (dry: in --out-root)
        f.write(ctx.run_dir + "\n")


# ============================================================================ step 1: validate + coverage
def _r2(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 2)


def write_policy_files(ctx, merge_dir, names, ann, partial):
    """Partly counted lines (module docstring). From label_day_merge.js's scoring files (cleared and nonwoven records
    already removed) write
      merged/scoring_headline/<stem>.json|csv  minus every record with a line spanning < SPAN_MIN (the headline label
                                               set: propagate, score150, quality150 and the proposal check read it)
      merged/scoring_partial/<stem>.csv        the headline rows plus the JUSTIFIED partial records ('partially
                                               occluded' or Notes), with the separate columns span_h/v, partial_h/v and
                                               h_tpi_rescaled / v_tpi_rescaled (= h_tpi / v_tpi on complete lines)
    and point ctx.scoring_json / ctx.scoring_csv at the headline files; ctx.partial_csv = the sensitivity files."""
    drop = {(r.annotator, int(r.id)) for _, r in partial.iterrows()}
    keep_partial = {(r.annotator, int(r.id)): r for _, r in partial[partial.sens_ok].iterrows()}
    hd, pdir = os.path.join(merge_dir, "scoring_headline"), os.path.join(merge_dir, "scoring_partial")
    os.makedirs(hd, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    sj, sc, pc = [], [], []
    for n in [x for x in names if not (x == "anonymous" and ann[x]["n_done"] == 0)]:
        stem = ann[n]["file_stem"]
        js = json.load(open(os.path.join(merge_dir, "scoring", stem + ".json")))
        recs = js["records"]
        js["records"] = [r for r in recs if (n, int(r["id"])) not in drop]
        js["n_records"] = len(js["records"])
        js["n_done"] = sum(1 for r in js["records"] if r.get("done"))
        js["label_day_policy"] = (f"records with a guide line spanning < {SPAN_MIN} of the crop removed "
                                  f"({len(recs) - len(js['records'])}); see label_day.py, PARTLY COUNTED LINES")
        with open(os.path.join(hd, stem + ".json"), "w") as f:
            json.dump(js, f, indent=1)
        df = pd.read_csv(os.path.join(merge_dir, "scoring", stem + ".csv"), dtype=str, keep_default_na=False)
        ids = pd.to_numeric(df["id"], errors="coerce")
        dmask = np.array([(n, int(i)) in drop if pd.notna(i) else False for i in ids])
        df[~dmask].to_csv(os.path.join(hd, stem + ".csv"), index=False)
        # sensitivity copy: same rows + the justified partial records, rescaled TPI in separate columns
        sens = df[~dmask | np.array([(n, int(i)) in keep_partial if pd.notna(i) else False for i in ids])].copy()
        for ax in ("h", "v"):
            sens[f"span_{ax}"], sens[f"partial_{ax}"], sens[f"{ax}_tpi_rescaled"] = "", "0", sens[f"{ax}_tpi"]
        for i, row in sens.iterrows():
            r = keep_partial.get((n, int(float(row["id"])))) if row["id"] != "" else None
            if r is None:
                continue
            for ax in ("h", "v"):
                sens.at[i, f"span_{ax}"] = f"{float(r[f'span_{ax}']):.4f}"
                if r[f"partial_{ax}"]:
                    sens.at[i, f"partial_{ax}"] = "1"
                    x = _r2(r[f"{ax}_tpi_rescaled"])
                    sens.at[i, f"{ax}_tpi_rescaled"] = "" if x is None else f"{x:.2f}"
        sens.to_csv(os.path.join(pdir, stem + ".csv"), index=False)
        sj.append(os.path.join(hd, stem + ".json"))
        sc.append(os.path.join(hd, stem + ".csv"))
        pc.append(os.path.join(pdir, stem + ".csv"))
    ctx.scoring_json, ctx.scoring_csv, ctx.partial_csv = sj, sc, pc


def step_validate(ctx):
    a = ctx.a
    merge_dir = os.path.join(ctx.work, "merged")
    cmd = [ctx.node, os.path.join(HERE, "label_day_merge.js"), "--out", merge_dir, "--manifest", MANIFEST]
    for r in a.rename or []:
        cmd += ["--rename", r]
    ctx.run(cmd + ctx.inputs)
    rep = json.load(open(os.path.join(merge_dir, "merge_report.json")))
    errors = []
    V = ctx.N["validation"]
    bad = [f for f in rep["files"] if f.get("error")]
    for f in bad:
        msg = f"unreadable export {f['path']}: {f['error']}"
        if a.skip_bad_files:
            ctx.warn(msg + " (skipped, --skip-bad-files)")
        else:
            errors.append(msg + "  (a half-written autosave? re-export, or pass --skip-bad-files)")
    ann = rep["annotators"]
    names = sorted(ann)
    if "anonymous" in ann and ann["anonymous"]["n_done"] > 0:
        errors.append(f"{ann['anonymous']['n_done']} finished record(s) have no annotator name: name them with "
                      "--rename anonymous=NAME (one person per export set!)")
    folded = {}
    for n in names:
        folded.setdefault(n.strip().casefold(), []).append(n)
    for k, v in folded.items():
        if len(v) > 1:
            errors.append(f"annotator names differ only in case/spaces: {v} -> merge them with --rename")
    if ctx.mode == "real":
        fake = [n for n in names if looks_fake(n)]
        if fake:
            errors.append(f"--real refuses annotators that look like test data: {fake}")
    stamps = {f.get("manifest_generated") for f in rep["files"] if f.get("manifest_generated")}
    if stamps and rep.get("manifest_generated") and stamps != {rep["manifest_generated"]}:
        ctx.warn(f"exports were written against manifest(s) {sorted(stamps)}; the current manifest is "
                 f"{rep['manifest_generated']} (scene fields and the IAA subset are taken from the current one)")
    # merged per-annotator files (scoring = without records whose line was cleared)
    for n in names:
        if n == "anonymous" and ann[n]["n_done"] == 0:
            continue
        stem = ann[n]["file_stem"]
        ctx.full_json.append(os.path.join(merge_dir, "full", stem + ".json"))
        ctx.scoring_json.append(os.path.join(merge_dir, "scoring", stem + ".json"))
        ctx.scoring_csv.append(os.path.join(merge_dir, "scoring", stem + ".csv"))
    man = json.load(open(MANIFEST))
    ctx.man = man
    PS = load_module("propagate_scene_labels", os.path.join(TOOL, "propagate_scene_labels.py"))
    AN = load_module("analyze_tool", os.path.join(TOOL, "analyze.py"))
    ctx.PS, ctx.AN = PS, AN
    scenes = PS.load_scenes(man)
    ctx.scenes = scenes
    iaa = set(man.get("iaa_subset") or [])
    ctx.iaa = sorted(iaa)
    attn = {int(e["id"]): (e.get("attention"), e.get("attention_score")) for e in man["images"] if e.get("attention")}
    ctx.attention = attn
    rect = {int(e["id"]): float(e["rect_size"][0]) for e in man["images"]}
    rows = []           # one row per (annotator, record)
    for n, path in zip([x for x in names if not (x == "anonymous" and ann[x]["n_done"] == 0)], ctx.full_json):
        recs = json.load(open(path))["records"]
        cleared = {c["id"]: c["lines"] for c in ann[n]["cleared"]}
        for r in recs:
            i = int(r["id"])
            if i not in scenes.index:
                errors.append(f"{n}: record id {i} is not in the manifest")
                continue
            if r.get("rect_size") is not None and abs(float(r["rect_size"]) - rect[i]) > 0.5:
                errors.append(f"{n}: record #{i} has rect_size {r['rect_size']} px but the manifest crop is {rect[i]:.0f} px "
                              "(an export of another kit? the screening test reuses ids 1-2)")
                continue
            ht, vt = r.get("h_tpi"), r.get("v_tpi")
            gh, gv = line_geom(r, "h"), line_geom(r, "v")
            rows.append(dict(annotator=n, id=i, scene=int(scenes.at[i, "scene_id"]), rep=bool(scenes.at[i, "is_scene_rep"]),
                             kind=scenes.at[i, "kind"], done=bool(r.get("done")), nonwoven=bool(r.get("nonwoven")),
                             iaa_scene=i in iaa, group=AN.blind_group(r), cleared=",".join(cleared.get(i, [])),
                             h_tpi=ht, v_tpi=vt, time_s=r.get("time_spent_s") or 0, weave=r.get("weave"),
                             csv_only=bool(r.get("imported_from_csv")), tool_version=str(r.get("tool_version") or ""),
                             attention=attn.get(i, (None, None))[0], prop_free_fields=proposal_free(r),
                             early=opened_before_hide(r, a.proposals_hidden_at, iaa), created=str(r.get("created") or ""),
                             n_h=gh["n"], n_v=gv["n"], span_h=gh["span"], span_v=gv["span"],
                             cov_h=gh["span"] + (gh["gap"] or 0) / float(r.get("rect_size") or 1),
                             cov_v=gv["span"] + (gv["gap"] or 0) / float(r.get("rect_size") or 1),
                             h_tpi_rescaled=gh["tpi_rescaled"], v_tpi_rescaled=gv["tpi_rescaled"],
                             occluded=bool((r.get("flags") or {}).get("occluded")), notes=str(r.get("notes") or "")[:80]))
    R = pd.DataFrame(rows, columns=["annotator", "id", "scene", "rep", "kind", "done", "nonwoven", "iaa_scene", "group",
                                    "cleared", "h_tpi", "v_tpi", "time_s", "weave", "csv_only", "tool_version",
                                    "attention", "prop_free_fields", "early", "created", "n_h", "n_v", "span_h", "span_v",
                                    "cov_h", "cov_v", "h_tpi_rescaled", "v_tpi_rescaled", "occluded", "notes"])
    # partly counted lines (module docstring): headline = both lines span >= SPAN_MIN; sensitivity keeps a shorter line
    # only with 'partially occluded' or Notes, scored over its counted part (<ax>_tpi_rescaled)
    R["partial_h"] = (R.n_h > 0) & (R.span_h < SPAN_MIN)
    R["partial_v"] = (R.n_v > 0) & (R.span_v < SPAN_MIN)
    R["partial"] = R.partial_h | R.partial_v
    R["justified"] = R.occluded | (R.notes.str.strip() != "")
    ok_h = pd.to_numeric(R.h_tpi_rescaled, errors="coerce").gt(0)
    ok_v = pd.to_numeric(R.v_tpi_rescaled, errors="coerce").gt(0)
    R["sens_ok"] = R.justified & (~R.partial_h | ok_h) & (~R.partial_v | ok_v)   # enters the sensitivity rows
    if errors:
        R.to_csv(os.path.join(ctx.work, "records.csv"), index=False)
        raise StepError("validation failed:\n    - " + "\n    - ".join(errors))
    done = R[R.done]
    if done.empty:
        R.to_csv(os.path.join(ctx.work, "records.csv"), index=False)
        raise StepError("no finished (done) record in the exports: nothing to score")
    # protocol (module docstring): auto = single with one finished annotator; dual with two or more when no finished
    # record carries proposal dots (the annotator kit with proposals hidden); iaa (the first protocol) otherwise
    n_ann = int(done.annotator.nunique())
    carries = done[~done.prop_free_fields]
    if a.protocol != "auto":
        proto = a.protocol
    elif n_ann == 1:
        proto = "single"
    elif carries.empty:
        proto = "dual"
    else:
        proto = "iaa"
        ctx.warn(f"--protocol auto picked 'iaa' (Oct 7 protocol) because {len(carries)} finished record(s) carry proposal "
                 f"dots: {sorted((r.annotator, int(r.id)) for _, r in carries.iterrows())[:20]}. If these exports come from "
                 "the Oct 11 freelancer kit (dots live for ~30 min before the hide), re-run with --protocol dual")
    ctx.protocol = proto
    ctx.single, ctx.dual = proto == "single", proto == "dual"
    ctx.oct11 = proto in ("single", "dual")
    ctx.ann_order = annotator_order(sorted(done.annotator.unique()))
    V["protocol"] = proto
    V["annotator_order"] = ctx.ann_order
    V["records_carrying_proposal_dots"] = int(len(carries))
    if ctx.dual and n_ann < 2:
        ctx.warn(f"--protocol dual but only {n_ann} annotator has finished records: no inter-annotator agreement")
    if ctx.dual and n_ann > 2:
        ctx.warn(f"--protocol dual with {n_ann} annotators {ctx.ann_order}: agreement on the pair with the most "
                 "scenes in common (every pair in numbers.json)")
    # proposal-free: from the record fields, and (single/dual protocol = annotator kit) not created before the hide time
    R["prop_free"] = (R.prop_free_fields & ~R.early) if ctx.oct11 else R.prop_free_fields.copy()
    R.to_csv(os.path.join(ctx.work, "records.csv"), index=False)
    done = R[R.done]
    usable_any = done[~done.nonwoven & (done.cleared == "")]
    partial = usable_any[usable_any.partial]
    usable = usable_any[~usable_any.partial]          # the headline label set (= the scoring files written below)
    write_policy_files(ctx, merge_dir, names, ann, partial)
    V["partial_lines"] = [dict(annotator=r.annotator, id=int(r.id), span_h=round(100 * r.span_h), span_v=round(100 * r.span_v),
                               n_h=int(r.n_h), n_v=int(r.n_v), occluded=bool(r.occluded), notes=r.notes,
                               justified=bool(r.justified), in_sensitivity=bool(r.sens_ok),
                               h_tpi=r.h_tpi, v_tpi=r.v_tpi,
                               h_tpi_rescaled=_r2(r.h_tpi_rescaled) if r.partial_h else None,
                               v_tpi_rescaled=_r2(r.v_tpi_rescaled) if r.partial_v else None)
                          for _, r in partial.iterrows()]
    if len(partial):
        ctx.warn(f"{len(partial)} finished record(s) count only part of a guide line (span < {SPAN_MIN:.0%} of the crop): "
                 "dropped from the headline scoring files (policy: headline = both lines >= 75%); "
                 f"{int(partial.sens_ok.sum())} with 'partially occluded'/Notes enter the rescaled sensitivity rows: "
                 f"{[(d['id'], d['span_h'], d['span_v'], 'kept' if d['in_sensitivity'] else 'NOT kept (no reason given)') for d in V['partial_lines']]}")
    shortl = usable[((usable.span_h < SPAN_WARN) & (usable.n_h > 0)) | ((usable.span_v < SPAN_WARN) & (usable.n_v > 0))]
    V["short_span"] = [dict(annotator=r.annotator, id=int(r.id), span_h=round(100 * r.span_h, 1),
                            span_v=round(100 * r.span_v, 1), cov_h=round(100 * r.cov_h, 1), cov_v=round(100 * r.cov_v, 1),
                            n_h=int(r.n_h), n_v=int(r.n_v),
                            likely_incomplete=bool(min(r.cov_h, r.cov_v) < SPAN_WARN)) for _, r in shortl.iterrows()]
    if len(shortl):
        inc = [d for d in V["short_span"] if d["likely_incomplete"]]
        ctx.warn(f"{len(shortl)} headline record(s) have a line spanning {SPAN_MIN:.0%}-{SPAN_WARN:.0%} of the crop (kept); "
                 f"{len(inc)} of them stay below {SPAN_WARN:.0%} even with one dot spacing added (probably a yarn or more "
                 f"missed at an end; the others are coarse fabrics): "
                 f"{[(d['id'], d['span_h'], d['span_v']) for d in inc] if inc else 'none'}")
    nf = done[~done.prop_free]   # over every finished record (cleared / nonwoven / partial ones included)
    ctx.not_free = {(r.annotator, int(r.id)) for _, r in nf.iterrows()}
    V["proposal_free"] = dict(k=int(done.prop_free.sum()), n=int(len(done)),
                              not_free=[dict(annotator=r.annotator, id=int(r.id), fields=not bool(r.prop_free_fields),
                                             created_before_hide=bool(r.early), created=r.created)
                                        for _, r in nf.iterrows()], hidden_at=a.proposals_hidden_at)
    if ctx.oct11 and len(nf):
        ctx.warn(f"{proto} protocol, but only {V['proposal_free']['k']} of {len(done)} finished records are "
                 f"proven proposal-free: {len(nf)} had proposal dots in the record or were created before the proposals "
                 f"were hidden ({a.proposals_hidden_at}): {sorted(int(i) for i in nf.id)} (scored without them as a "
                 "sensitivity subset)")
    # per annotator summary
    per_ann = []
    for n, g in R.groupby("annotator"):
        d = g[g.done]
        per_ann.append(dict(annotator=n, records=len(g), done=len(d), not_done=int((~g.done).sum()),
                            reps_done=int(d.rep.sum()), copies_done=int((~d.rep).sum()),
                            nonwoven=int(d.nonwoven.sum()), cleared_lines=int((d.cleared != "").sum()),
                            blind_clean=int((d.group == "blind-clean").sum()), shown=int((d.group == "shown").sum()),
                            mixed=int((d.group == "mixed").sum()), proposal_free=int(d.prop_free.sum()), csv_only=int(d.csv_only.sum()),
                            conflicts=len({c["id"] for c in ann[n]["conflicts"]}),
                            median_min=round(float(d.time_s.median()) / 60, 1) if len(d) else None))
    V["per_annotator"] = per_ann
    V["annotators"] = sorted(R.annotator.unique().tolist())
    V["files"] = [dict(path=f["path"], annotators=f["annotators"], n_records=f["n_records"], n_done=f["n_done"],
                       error=f.get("error")) for f in rep["files"]]
    # issues worth a look (warnings, not errors)
    cl = done[(done.cleared != "") & ~done.nonwoven]
    V["cleared"] = []
    for _, r in cl.iterrows():
        other = usable[(usable.scene == r.scene) & (usable.annotator != r.annotator)]
        V["cleared"].append(dict(annotator=r.annotator, id=int(r.id), scene=int(r.scene), lines=r.cleared,
                                 scene_still_labelled=bool(len(other) or len(usable[usable.scene == r.scene]))))
        ctx.warn(f"cleared line ({r.cleared}) on #{r.id} by {r.annotator}: count missing; the record is not scored"
                 + ("" if len(usable[usable.scene == r.scene]) else f" and scene {r.scene} has no other usable label"))
    copies = done[~done.rep & ~done.nonwoven]
    V["copies_done"] = [dict(annotator=r.annotator, id=int(r.id), scene=int(r.scene), kind=r.kind)
                        for _, r in copies.iterrows()]
    if len(copies):
        ctx.warn(f"{len(copies)} finished record(s) on copies ('Edit anyway'): used only if the rep has no label: "
                 f"{[(r.annotator, int(r.id)) for _, r in copies.iterrows()]}")
    nw = done[done.nonwoven]
    V["nonwoven"] = [dict(annotator=r.annotator, id=int(r.id)) for _, r in nw.iterrows()]
    if len(nw):
        ctx.warn(f"{len(nw)} record(s) marked nonwoven (excluded): {[(r.annotator, int(r.id)) for _, r in nw.iterrows()]}")
    V["not_done"] = int((~R.done).sum())
    V["tool_versions"] = {str(k): int(v) for k, v in done.tool_version.value_counts().items()}
    # tool v2.2 shows a "check carefully" hint on high/medium-attention reps outside the IAA subset and never
    # on blind records: such records may be anchored by it (disclose with the anchoring analysis)
    hint = usable_any[usable_any.tool_version.str.startswith("2.2") & usable_any.rep & ~usable_any.iaa_scene &
                      (usable_any.group != "blind-clean") & usable_any.attention.isin(["high", "medium"])]
    V["attention_hint_records"] = int(len(hint))
    V["attention_hint_ids"] = sorted(int(i) for i in hint.id)
    ctx.hint_ids = set(V["attention_hint_ids"])
    if len(hint):
        ctx.warn(f"{len(hint)} finished record(s) were made with tool v2.2 on high/medium-attention scenes, where the "
                 "'check carefully' (high) / 'take care' (medium) badge with a machine-derived reason but no count was "
                 "shown: " + ("disclose it in supp S4 (scored without them as a sensitivity subset)" if ctx.oct11 else
                              "disclose it with the anchoring analysis"))
    implaus = usable[(pd.to_numeric(usable.h_tpi) < 5) | (pd.to_numeric(usable.h_tpi) > 400) |
                     (pd.to_numeric(usable.v_tpi) < 5) | (pd.to_numeric(usable.v_tpi) > 400)]
    for _, r in implaus.iterrows():
        ctx.warn(f"implausible count on #{r.id} by {r.annotator}: h {r.h_tpi}, v {r.v_tpi} TPI")
    shown_iaa = done[done.iaa_scene & done.rep & (done.group != "blind-clean")] if not ctx.oct11 else done.iloc[0:0]
    V["iaa_not_blind"] = [dict(annotator=r.annotator, id=int(r.id), group=r.group) for _, r in shown_iaa.iterrows()]
    for _, r in shown_iaa.iterrows():
        ctx.warn(f"IAA scene #{r.id} by {r.annotator} is '{r.group}' (proposals seen): not used for agreement")
    for n in names:
        for c in {c["id"]: c for c in ann[n]["conflicts"]}.values():
            ctx.log(f"  note: {n} #{c['id']} exists in several exports with different content; kept the "
                    f"{'newer' if c['kept_timestamp'] and c['dropped_timestamp'] and c['kept_timestamp'] >= c['dropped_timestamp'] else 'edited'} "
                    f"one ({c['kept_timestamp']}: {c['kept_counts']} vs {c['dropped_timestamp']}: {c['dropped_counts']})")
    # coverage (scenes labelled the way propagate_scene_labels.py labels them: any usable member record)
    sc_reps = scenes[scenes.is_scene_rep]
    lab_any = set(usable.scene)
    lab_rep = set(usable[usable.rep].scene)
    res = sc_reps.scene_resolution
    blind_ok = usable[usable.iaa_scene & usable.rep & (usable.group == "blind-clean")]
    per_iaa = blind_ok.groupby("id").annotator.nunique()
    n_blind2 = int((per_iaa >= 2).sum())
    unknown13 = []
    if os.path.exists(P_PSEUDO):
        ps = pd.read_csv(P_PSEUDO)
        unknown13 = sorted(int(s) for s in ps[ps.is_rep & (ps.weave_ours == "unknown")].scene)
    C = ctx.N["coverage"]
    C.update(n_scenes=int(len(sc_reps)), scenes_labelled=len(lab_any), scenes_labelled_on_rep=len(lab_rep),
             full_labelled=int(sum(res.get(s) == "full" for s in lab_any)), n_full=int((res == "full").sum()),
             quarter_labelled=int(sum(res.get(s) == "quarter_only" for s in lab_any)),
             n_quarter=int((res == "quarter_only").sum()),
             iaa_scenes=len(iaa), blind_double=n_blind2, blind_any=int(len(per_iaa)),
             unknown_weave_labelled=sum(s in lab_any for s in unknown13), unknown_weave_n=len(unknown13),
             unlabelled_scenes=sorted(int(s) for s in sc_reps.index if s not in lab_any),
             blind_double_scenes=sorted(int(i) for i in per_iaa.index[per_iaa >= 2]),
             partial_only_scenes=sorted(int(s) for s in set(partial.scene) - lab_any),
             partial_records=int(len(partial)), partial_justified=int(partial.sens_ok.sum()))
    C["verdict_scenes"] = verdict(C["scenes_labelled"], THR["scenes_go"], THR["scenes_reduced"])
    C["verdict_blind"] = "n/a" if ctx.oct11 else verdict(n_blind2, THR["blind_go"], THR["blind_reduced"])
    # dual protocol: scenes with a usable count by each annotator and by both (the agreement set of step 3)
    lab_by = {n: set(usable[usable.annotator == n].scene) for n in ctx.ann_order}
    both = set.intersection(*lab_by.values()) if len(lab_by) >= 2 else set()
    C.update(scenes_by_annotator={n: len(v) for n, v in lab_by.items()}, scenes_both=len(both),
             scenes_one_only=sorted(int(s) for s in lab_any - both) if len(lab_by) >= 2 else [])
    # go/no-go screening row: screening answers vs the legacy hand counts (constants, supp Tab. S4), max over the run's annotators
    scr_rows = screening_rows(V["annotators"])
    scr = [100 * abs(x - y) / y for _, _, ans, ref in scr_rows for x, y in zip(ans, ref)]
    C["screening_max_pct"] = max(scr)
    C["screening_annotators"] = sorted({n for n, _, _, _ in scr_rows})
    C["screening_max_by_annotator"] = {n: max(100 * abs(x - y) / y for m, _, ans, ref in scr_rows if m == n
                                              for x, y in zip(ans, ref)) for n in C["screening_annotators"]}
    if ctx.dual:
        miss = [n for n in ctx.ann_order if n not in SCREENING]
        if miss:
            ctx.warn(f"no screening constants for annotator(s) {miss} (SCREENING has {list(SCREENING)}): rename them "
                     "to FL_SA / FL_NG (--rename OLD=NEW, one person each) or the screening row covers only the others")
    C["verdict_screening"] = verdict(C["screening_max_pct"], THR["screen_go"], THR["screen_reduced"], higher_is_better=False)
    iaa_rows = []
    for i in sorted(iaa):
        st = {}
        for n in V["annotators"]:
            g = R[(R.id == i) & (R.annotator == n)]
            if g.empty:
                st[n] = "-"
            else:
                r = g.iloc[0]
                st[n] = ("done " if r.done else "open ") + r.group + (" CLEARED:" + r.cleared if r.cleared else "")
        iaa_rows.append([i, res.get(i, "?"), "yes" if per_iaa.get(i, 0) >= 2 else "no"] + [st[n] for n in V["annotators"]])
    ctx.iaa_table = None if ctx.oct11 else iaa_rows
    # report
    ctx.log("")
    ctx.log("  COVERAGE vs PLAN section 3")
    ctx.log(f"    scenes with a usable human count: {C['scenes_labelled']} / 76 "
            f"(on the rep itself: {C['scenes_labelled_on_rep']}; full-res {C['full_labelled']}/{C['n_full']}, "
            f"quarter-only {C['quarter_labelled']}/{C['n_quarter']}) -> {C['verdict_scenes']} "
            f"(GO >= {THR['scenes_go']}, REDUCED {THR['scenes_reduced']}-{THR['scenes_go'] - 1})")
    if C["partial_records"]:
        ctx.log(f"    + {C['partial_records']} record(s) with a partly counted line (not in the headline; "
                f"{C['partial_justified']} in the rescaled sensitivity rows); scenes labelled only that way: "
                f"{C['partial_only_scenes'] or 'none'}")
    if ctx.single:
        pf = V["proposal_free"]
        ctx.log(f"    single-annotator protocol (Oct 11): no IAA, no blind-only check; proposal-free records: "
                f"{pf['k']} of {pf['n']}; screening vs 2019 hand counts max {C['screening_max_pct']:.1f}% -> "
                f"{C['verdict_screening']} (GO <= {THR['screen_go']:.0f}%)")
    elif ctx.dual:
        pf = V["proposal_free"]
        ctx.log(f"    dual protocol (Oct 11, two annotators from scratch; annotator 1, 2 = {ctx.ann_order}): scenes "
                f"per annotator {C['scenes_by_annotator']}, counted by both {C['scenes_both']} (the agreement set); "
                f"no blind subset, no anchoring rule; proposal-free records: {pf['k']} of {pf['n']}; screening vs 2019 "
                f"hand counts max {C['screening_max_pct']:.1f}% ({C['screening_annotators']}) -> {C['verdict_screening']} "
                f"(GO <= {THR['screen_go']:.0f}%)")
    else:
        ctx.log(f"    blind scenes counted by >= 2 annotators (blind-clean, done, no cleared line): {n_blind2} / {len(iaa)} "
                f"-> {C['verdict_blind']} (GO >= {THR['blind_go']}, REDUCED {THR['blind_reduced']}-{THR['blind_go'] - 1})")
    ctx.log(f"    'unknown'-weave scenes labelled: {C['unknown_weave_labelled']} / {C['unknown_weave_n']}")
    for p in per_ann:
        ctx.log(f"    {p['annotator']}: {p['done']} done ({p['reps_done']} reps, {p['copies_done']} copies), "
                f"{p['not_done']} open, blind-clean {p['blind_clean']}, shown {p['shown']}, mixed {p['mixed']}, "
                f"cleared {p['cleared_lines']}, nonwoven {p['nonwoven']}, median {p['median_min']} min")
    if ctx.mode == "real" and C["verdict_scenes"] == "NO-GO" and not a.allow_partial:
        raise StepError(f"only {C['scenes_labelled']} scenes are labelled (NO-GO < {THR['scenes_reduced']}): --real "
                        "would fill Tab. 1 from partial labels. Use --out-root for a plumbing check, or --allow-partial.")
    if C["scenes_labelled"] < 76:
        ctx.warn(f"{76 - C['scenes_labelled']} scenes have no usable human count: {C['unlabelled_scenes']}")


# ============================================================================ step 2: propagate
def step_propagate(ctx):
    files = [os.path.join(ctx.prop_dir, f) for f in ("labels_per_file.csv", "labels_per_scene.csv", "labels_meta.json")]
    ctx.backup_existing(files)
    ctx.run([sys.executable, os.path.join(TOOL, "propagate_scene_labels.py"), *ctx.scoring_json,
             "--out-dir", ctx.prop_dir])
    ctx.written += files
    meta = json.load(open(os.path.join(ctx.prop_dir, "labels_meta.json")))
    ctx.N["coverage"]["propagated"] = dict(scenes=meta["n_scenes_labelled"], files=meta["n_files_labelled"],
                                           swapped=meta["n_swapped_files_labelled"],
                                           from_non_rep=meta["labels_from_non_rep"])
    if meta["n_scenes_labelled"] != ctx.N["coverage"]["scenes_labelled"]:
        raise StepError(f"propagate_scene_labels.py labelled {meta['n_scenes_labelled']} scenes, step 1 counted "
                        f"{ctx.N['coverage']['scenes_labelled']}: inconsistent label rules")
    ctx.log(f"  per-scene labels: {meta['n_scenes_labelled']} scenes, {meta['n_files_labelled']} files "
            f"({meta['n_swapped_files_labelled']} with h/v swapped); scenes labelled from a copy: "
            f"{meta['labels_from_non_rep']}")


# ============================================================================ step 3: agreement + anchoring
def _anchoring(AN, recs, props, group):
    """Same definition as analyze.py summary(): |final - proposal| / final per line (dot counts)."""
    errs = []
    for r in recs:
        if AN.blind_group(r) != group:
            continue
        i = int(r["id"])
        for line in ("h", "v"):
            final = len(r[line + "_dots"])
            if not final:
                continue
            if group == "shown":
                n_prop = r.get("n_prop_" + line) or 0
            elif i in props:
                n_prop = len(props[i][line + "_positions"])
            else:
                continue
            errs.append(abs(final - n_prop) / final)
    return errs


P_FREELANCER_PROPOSALS = os.path.join(ROOT, "freelancer", "web", "annotate_manifest_with_proposals.json")


def proposal_edits(ctx):
    """Single-annotator protocol: the machine proposals were never shown, so the edits a check-and-correct
    (assisted) annotator would have made are measured post hoc, free of anchoring. Per guide line of every finished
    representative in the headline label set, the proposal dots of annotation_tool/data/manifest.json (all 76 reps; the
    annotator kit's own copy with proposals, which lacks the 20 IAA reps, is checked to be identical) are matched
    one-to-one to the human dots (match_1d: maximum matching within --edit-tol x the human median dot spacing, then
    the smallest total distance):
      add    = human dots without a machine dot (missed yarns)
      delete = machine dots without a human dot (false crossings)
      move   = matched pairs further apart than --move-tol x the spacing (right yarn, off-centre; count-neutral)
      edits  = add + delete (what changes the count); edits_with_moves = add + delete + move
    Lines whose guide line the annotator turned (angle != 0) or moved by more than half a spacing are kept but flagged
    (positions are line parameters, so they stay roughly comparable); 'clean' also drops the badge and guide scenes."""
    G = ctx.N["agreement"]
    G["protocol"] = ctx.protocol   # single, or dual (then dual_agreement() fills the human-human numbers afterwards)
    ctx.N["coverage"]["hh_median_rel"] = None
    ctx.N["coverage"]["verdict_hh"] = "n/a"
    ctx.human_metrics = None
    props = {int(e["id"]): (e.get("proposal") or {}) for e in ctx.man["images"]}
    if os.path.exists(P_FREELANCER_PROPOSALS):   # same machine dots as the kit would have shown (where it had them)
        fp = {int(e["id"]): e.get("proposal") for e in json.load(open(P_FREELANCER_PROPOSALS))["images"]}
        bad = [i for i, p in fp.items() if p and json.dumps(p, sort_keys=True) != json.dumps(props.get(i), sort_keys=True)]
        if bad:
            ctx.warn(f"proposals differ between annotation_tool/data/manifest.json and the freelancer kit's copy on {bad}")
        G["proposal_source_check"] = dict(n_kit_with_proposal=sum(1 for p in fp.values() if p), n_differ=len(bad))
    iaa, tol, mtol = set(ctx.iaa), ctx.a.edit_tol, ctx.a.move_tol
    rows, times, times_by = [], [], {}
    for p in ctx.full_json:   # minutes per photo: every finished representative (partial / cleared ones included)
        for r in json.load(open(p))["records"]:
            if (r.get("done") and not r.get("nonwoven") and int(r["id"]) in ctx.scenes.index
                    and ctx.scenes.at[int(r["id"]), "is_scene_rep"]):
                times.append((r.get("time_spent_s") or 0) / 60)
                times_by.setdefault(r.get("annotator") or "anonymous", []).append(times[-1])
    for p in ctx.scoring_json:   # dot comparison: the headline label set (complete lines only)
        for r in json.load(open(p))["records"]:
            i = int(r["id"])
            if not r.get("done") or r.get("nonwoven") or i not in ctx.scenes.index or not ctx.scenes.at[i, "is_scene_rep"]:
                continue
            for line, key, lk in (("h", "h_positions", "h_line_y"), ("v", "v_positions", "v_line_x")):
                hum = sorted(float(x) for x in r.get(line + "_dots") or [])
                pro = sorted(float(x) for x in props.get(i, {}).get(key) or [])
                if len(hum) < 2 or not len(pro):
                    continue
                sp = float(np.median(np.diff(hum)))
                pairs = match_1d(hum, pro, tol * sp)
                off = [abs(hum[a_] - pro[b_]) for a_, b_ in pairs]
                rows.append(dict(annotator=r.get("annotator"), id=i, line=line, n_human=len(hum), n_prop=len(pro),
                                 matched=len(pairs), add=len(hum) - len(pairs), delete=len(pro) - len(pairs),
                                 move=int(sum(o > mtol * sp for o in off)),
                                 offset_med_px=round(float(np.median(off)), 2) if off else None,
                                 count_diff_pct=round(100 * abs(len(hum) - len(pro)) / len(hum), 2),
                                 line_offset_px=abs(float(r.get(lk) or 0) - float(props[i].get(lk) or 0)),
                                 angle_deg=float(r.get(line + "_angle_deg") or 0), spacing_px=round(sp, 2),
                                 attention_hint=i in ctx.hint_ids, guide_example=i in GUIDE_EXAMPLE_SCENES,
                                 iaa_scene=i in iaa, prop_free=(r.get("annotator") or "anonymous", i) not in ctx.not_free))
    E = pd.DataFrame(rows)
    os.makedirs(ctx.stage, exist_ok=True)
    G["time_min"] = dict(all=float(np.median(times)) if times else None, n=len(times),
                         by_annotator={k: dict(median=float(np.median(v)), n=len(v)) for k, v in times_by.items()})
    if E.empty:
        E.to_csv(os.path.join(ctx.stage, "proposal_edits150.csv"), index=False)
        ctx.warn("no finished representative has a manifest proposal: no post-hoc proposal check")
        return
    E["edits"] = E["add"] + E["delete"]
    E["edits_with_moves"] = E["edits"] + E["move"]
    E["moved_or_rotated"] = (E.line_offset_px > 0.5 * E.spacing_px) | (E.angle_deg != 0)
    E.to_csv(os.path.join(ctx.stage, "proposal_edits150.csv"), index=False)

    def summ(e):
        if e.empty:
            return None
        one = lambda g: dict(n_lines=int(len(g)), n_scenes=int(g.id.nunique()), edits_med=float(g.edits.median()),
                             add_med=float(g["add"].median()), delete_med=float(g["delete"].median()),
                             move_med=float(g["move"].median()),
                             precision_med=float((100 * g.matched / g.n_prop).median()),
                             recall_med=float((100 * g.matched / g.n_human).median()),
                             le2_edit_pct=float(100 * (g.edits <= 2).mean()),
                             zero_edit_pct=float(100 * (g.edits == 0).mean()))
        out = one(e)
        out.update(edits_pct_med=float((100 * e.edits / e.n_human).median()),
                   count_diff_med=float(e.count_diff_pct.median()),
                   by_line={l: one(g) for l, g in e.groupby("line")})
        return out
    never = E[~E.guide_example & E.prop_free]   # machine dots truly never shown (Tab. S4 middle rows)
    clean = never[~never.attention_hint & ~never.moved_or_rotated]
    G["proposal_edits"] = dict(tol=tol, move_tol=mtol, all=summ(E), never_shown=summ(never), clean=summ(clean),
                               no_guide=summ(E[~E.guide_example]), n_not_free_scenes=int(E[~E.prop_free].id.nunique()),
                               n_hint_scenes=int(E[E.attention_hint].id.nunique()),
                               n_guide_scenes=int(E[E.guide_example].id.nunique()),
                               n_moved_or_rotated_lines=int(E.moved_or_rotated.sum()))
    # dual protocol (and any run with several annotators): the same check per annotator (numbers.json, the report)
    G["proposal_edits"]["by_annotator"] = {
        str(n): dict(all=summ(g), never_shown=summ(g[~g.guide_example & g.prop_free]),
                     n_not_free_scenes=int(g[~g.prop_free].id.nunique()))
        for n, g in E.groupby(E.annotator.fillna("anonymous"))}
    s = G["proposal_edits"]["all"]
    ctx.log(f"  {'single annotator' if ctx.single else ctx.protocol + ' protocol (all annotators pooled)'}: "
            f"post-hoc proposal check on {s['n_scenes']} scenes / {s['n_lines']} lines (proposals "
            f"never shown): median {s['edits_med']:.0f} edits per line ({s['add_med']:.0f} add, {s['delete_med']:.0f} "
            f"delete, {s['move_med']:.0f} off-centre; {s['edits_pct_med']:.1f}% of the human dots); "
            f"{s['zero_edit_pct']:.0f}% of lines need none, {s['le2_edit_pct']:.0f}% at most 2; machine dots matched "
            f"{s['precision_med']:.0f}%, human dots found {s['recall_med']:.0f}%; median {fnum(G['time_min']['all'])} "
            "min per photo")


def _agree(ctx, A, B, wa, wb, da=None, db=None):
    """Agreement of two annotators' per-scene counts (A, B: rows aligned, columns h, v [TPI], weave = score150 coarse
    weave) with the definitions of annotation_tool/analyze.py: MAE, Bland-Altman bias and 95% limits (mean diff +- 1.96
    SD, A - B), ICC(2,1) absolute agreement, Pearson r, within 5% / 10% of each other (|A - B| / mean of the two),
    median |diff|% per axis and over both axes (the go/no-go number); weave kappa on the labels as entered
    (wa, wb; 'unsure' is a category) and on score150's coarse weave (None -> 'unsure'); warp direction agreement
    (da, db, in the representative's frame)."""
    AN = ctx.AN
    n = int(len(A))
    out = dict(n=n, scenes=[int(s) for s in A.index])
    if n == 0:
        return out
    rel_all, w5_both = [], np.ones(n, bool)
    for ax in ("h", "v"):
        a, b = [float(x) for x in A[ax]], [float(x) for x in B[ax]]
        d = np.array(a) - np.array(b)
        m = (np.array(a) + np.array(b)) / 2
        rel = 100 * np.abs(d) / m
        rel_all += rel.tolist()
        w5_both &= rel <= 5
        sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
        out[ax] = dict(MAE=float(np.mean(np.abs(d))), bias=float(np.mean(d)), loa_lo=float(np.mean(d) - 1.96 * sd),
                       loa_hi=float(np.mean(d) + 1.96 * sd), icc=float(AN.icc_2_1(a, b)), r=float(AN.pearson(a, b)),
                       w5=float(100 * np.mean(rel <= 5)), w10=float(100 * np.mean(rel <= 10)),
                       mean_rel=float(np.mean(rel)), median_rel=float(np.median(rel)))
    out["median_rel_both_axes"] = float(np.median(rel_all))
    out["both_axes_within5_pct"] = float(100 * np.mean(w5_both))
    out["weave_agree"] = float(100 * np.mean([p == q for p, q in zip(wa, wb)]))
    out["weave_kappa"] = float(AN.cohen_kappa(list(wa), list(wb)))
    ca = [x if isinstance(x, str) and x else "unsure" for x in A["weave"]]
    cb = [x if isinstance(x, str) and x else "unsure" for x in B["weave"]]
    out["weave_agree_coarse"] = float(100 * np.mean([p == q for p, q in zip(ca, cb)]))
    out["weave_kappa_coarse"] = float(AN.cohen_kappa(ca, cb))
    if da is not None and db is not None:
        out["warp_agree"] = float(100 * np.mean([p == q for p, q in zip(da, db)]))
    return out


def dual_agreement(ctx):
    """Dual protocol (two annotators counting every scene from scratch): inter-annotator agreement over EVERY
    scene finished by both, on the per-scene labels each annotator alone gives (score150 load_labels(annotator=) +
    gt_table + propagate = the same headline label set and scene unit as the scores), and on the badge / no-badge
    scenes, without the 4 guide scenes and on proven proposal-free records only. Feeds the go/no-go human-human row
    (median |diff|% over both axes: GO <= hh_go, REDUCED <= hh_reduced), the Tab. 1 'Human vs human' row (annotator 2's
    counts scored against annotator 1's with score150.metrics; n = scenes counted by both) and supp Tab. S4.
    No anchoring rule, no blind-only run, no blind NO-GO."""
    G, C = ctx.N["agreement"], ctx.N["coverage"]
    S = load_score150()
    tabs, recs = {}, {}
    for n in ctx.ann_order:
        try:
            gt_n, rec_n = build_scene_gt(ctx, annotator=n)
        except (ValueError, KeyError, IndexError) as e:
            ctx.warn(f"no per-scene labels for annotator {n} ({e.__class__.__name__}: {e})")
            continue
        gt_n = gt_n[gt_n[["h", "v"]].notna().all(axis=1)]
        if len(gt_n):
            tabs[n], recs[n] = gt_n, rec_n
    names = list(tabs)
    pairs = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            pairs.append((int(len(tabs[names[x]].index.intersection(tabs[names[y]].index))), names[x], names[y]))
    pairs.sort(key=lambda t: -t[0])   # stable: annotator order on ties
    G["pairs"] = [dict(a=p[1], b=p[2], n_common=p[0]) for p in pairs]
    G["scenes_by_annotator"] = {n: int(len(t)) for n, t in tabs.items()}
    if not pairs or pairs[0][0] < 3:
        ctx.warn(f"fewer than 3 scenes counted by two annotators ({pairs[0][0] if pairs else 0}): no inter-annotator "
                 "agreement numbers in this run (Tab. 1 human row and the Tab. S4 agreement rows stay \\TBD)")
        G["primary"] = None
        C["hh_median_rel"], C["verdict_hh"], C["hh_n"] = None, "n/a", int(pairs[0][0] if pairs else 0)
        ctx.human_metrics = None
        return
    _, a1, a2 = pairs[0]
    A, B = tabs[a1], tabs[a2]
    common = A.index.intersection(B.index).sort_values()

    def raw_field(n, gt_n, col):   # weave / warp_dir as the annotator entered it, from the record labelling the scene
        r = recs[n]
        w = dict(zip(r["id"].astype(int), r[col])) if col in r.columns else {}
        out = {}
        for s_ in gt_n.index:
            src = gt_n.at[s_, "label_src_id"] if "label_src_id" in gt_n.columns else s_
            v = w.get(int(src)) if pd.notna(src) else None
            v = v if isinstance(v, str) and v else "none"
            if col == "warp_dir" and pd.notna(src) and int(src) in ctx.scenes.index and ctx.scenes.at[int(src), "swap_hv"]:
                v = ctx.PS.WARP_SWAP.get(v, v)   # a turned copy's warp direction in the representative's frame
            out[int(s_)] = v
        return out
    W1, W2 = raw_field(a1, A, "weave"), raw_field(a2, B, "weave")
    D1, D2 = raw_field(a1, A, "warp_dir"), raw_field(a2, B, "warp_dir")
    hint_sc = {int(ctx.scenes.at[i, "scene_id"]) for i in ctx.hint_ids if i in ctx.scenes.index}
    nf_sc = {(n, int(ctx.scenes.at[i, "scene_id"])) for n, i in ctx.not_free if i in ctx.scenes.index}
    subsets = [("badge scenes ('check carefully' / 'take care')", [s for s in common if s in hint_sc]),
               ("scenes without a badge", [s for s in common if s not in hint_sc]),
               (GUIDE_SUBSET, [s for s in common if s not in GUIDE_EXAMPLE_SCENES]),
               ("proven proposal-free records only", [s for s in common if (a1, s) not in nf_sc and (a2, s) not in nf_sc])]

    def block(idx):
        idx = pd.Index(idx)
        return _agree(ctx, A.loc[idx], B.loc[idx], [W1[int(s)] for s in idx], [W2[int(s)] for s in idx],
                      [D1[int(s)] for s in idx], [D2[int(s)] for s in idx])
    P = dict(a=a1, b=a2, label_set="headline (both lines span >= 75%), scene unit", **block(common))
    P["subsets"] = {nm: block(idx) for nm, idx in subsets}
    P["scenes_only_one"] = {n: sorted(int(s) for s in tabs[n].index.difference(common)) for n in (a1, a2)}
    # per-scene disagreement (a list for spot checks; tables/human_agreement_per_scene.csv)
    res = {int(s): ctx.scenes.at[int(s), "scene_resolution"] for s in common}
    per = pd.DataFrame({"scene": [int(s) for s in common], "resolution": [res[int(s)] for s in common],
                        "badge": [int(s) in hint_sc for s in common], "guide_example": [int(s) in GUIDE_EXAMPLE_SCENES for s in common],
                        f"h_{a1}": A.loc[common, "h"].to_numpy(float), f"v_{a1}": A.loc[common, "v"].to_numpy(float),
                        f"h_{a2}": B.loc[common, "h"].to_numpy(float), f"v_{a2}": B.loc[common, "v"].to_numpy(float),
                        f"weave_{a1}": [W1[int(s)] for s in common], f"weave_{a2}": [W2[int(s)] for s in common]})
    for ax in ("h", "v"):
        x, y = per[f"{ax}_{a1}"], per[f"{ax}_{a2}"]
        per[f"diff_{ax}"] = x - y
        per[f"rel_{ax}_pct"] = 100 * (x - y).abs() / ((x + y) / 2)
    per["rel_mean_pct"] = 0.5 * (per.rel_h_pct + per.rel_v_pct)
    per = per.sort_values("rel_mean_pct", ascending=False)
    os.makedirs(ctx.stage, exist_ok=True)
    per.round(3).to_csv(os.path.join(ctx.stage, "human_agreement_per_scene.csv"), index=False)
    P["worst"] = [dict(scene=int(r["scene"]), rel_mean_pct=round(float(r["rel_mean_pct"]), 1),
                       a=f"{r['h_' + a1]:.1f}x{r['v_' + a1]:.1f}", b=f"{r['h_' + a2]:.1f}x{r['v_' + a2]:.1f}")
                  for _, r in per.head(10).iterrows()]
    G["primary"] = P
    hh = P["median_rel_both_axes"]
    C["hh_median_rel"], C["hh_n"] = hh, int(len(common))
    C["verdict_hh"] = verdict(hh, THR["hh_go"], THR["hh_reduced"], higher_is_better=False)
    if len(common) < 10:
        ctx.warn(f"only {len(common)} scenes counted by both annotators: the human-human numbers have wide intervals")
    # Tab. 1 'Human vs human': annotator 2's counts scored against annotator 1's (score150.metrics, as every row)
    ga, gb = A.loc[common, ["h", "v", "weave"]], B.loc[common, ["h", "v", "weave"]]
    ctx.human_metrics = S.metrics(ga, gb)
    ctx.human_metrics.update(reference=a1, scored=a2)
    G["human_row"] = ctx.human_metrics
    G["human_row_reverse"] = dict(S.metrics(gb, ga), reference=a2, scored=a1)
    ctx.log(f"  dual agreement {a1} vs {a2} on {len(common)} scenes counted by both: MAE h {P['h']['MAE']:.2f} / "
            f"v {P['v']['MAE']:.2f} TPI, ICC {P['h']['icc']:.3f} / {P['v']['icc']:.3f}, weave kappa "
            f"{P['weave_kappa']:.2f}, median |diff| {hh:.2f}% -> {C['verdict_hh']} (GO <= {THR['hh_go']}%, REDUCED <= "
            f"{THR['hh_reduced']}%); " + "; ".join(f"{nm}: n = {b['n']}" + (f", {b['median_rel_both_axes']:.2f}%"
                                                                              if b.get("n") else "")
                                                   for nm, b in P["subsets"].items()))


def step_agreement(ctx):
    if ctx.single:   # single protocol: no second annotator -> no IAA / anchoring; post-hoc proposal check instead
        return proposal_edits(ctx)
    if ctx.dual:     # dual protocol, two annotators from scratch: post-hoc proposal check per annotator + agreement on all scenes
        proposal_edits(ctx)
        return dual_agreement(ctx)
    AN, man = ctx.AN, ctx.man
    G = ctx.N["agreement"]
    stdout_path = os.path.join(ctx.work, "analyze_stdout.txt")
    _, out = ctx.run([sys.executable, os.path.join(TOOL, "analyze.py"), *ctx.scoring_json, "--manifest", MANIFEST,
                      "--csv", os.path.join(ctx.work, "analyze_merged.csv")], capture=stdout_path)
    sets = [AN.load(p) for p in ctx.scoring_json]        # (name, {id: done record})
    props = {e["id"]: e["proposal"] for e in man["images"]}
    iaa = set(ctx.iaa)
    # anchoring per annotator (cross-checked against analyze.py's own printout) and pooled
    anch = {}
    parsed = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"== (.+?): \d+ images done", line)
        if m:
            cur = m.group(1)
        m = re.search(r"anchoring \[(shown|blind-clean|mixed)\]: .*?/ final = ([0-9.]+)%", line)
        if m and cur:
            parsed[(cur, m.group(1))] = float(m.group(2))
    pooled = {"shown": [], "blind-clean": []}
    for name, recs in sets:
        for g in ("shown", "blind-clean"):
            e = _anchoring(AN, recs.values(), props, g)
            pooled[g] += e
            val = 100 * float(np.median(e)) if e else None
            anch[f"{name}|{g}"] = val
            if val is not None and (name, g) in parsed and abs(parsed[(name, g)] - val) > 0.051:
                ctx.warn(f"anchoring {name}/{g}: {val:.1f}% here vs {parsed[(name, g)]}% in analyze.py")
    a_blind = 100 * float(np.median(pooled["blind-clean"])) if pooled["blind-clean"] else None
    a_shown = 100 * float(np.median(pooled["shown"])) if pooled["shown"] else None
    gap = (a_blind - a_shown) if (a_blind is not None and a_shown is not None) else None
    per_ann_gap = {}
    for name, _ in sets:
        b, s = anch.get(f"{name}|blind-clean"), anch.get(f"{name}|shown")
        if b is not None and s is not None:
            per_ann_gap[name] = b - s
    fires = bool((gap is not None and gap > THR["anchoring_gap"]) or
                 any(v > THR["anchoring_gap"] for v in per_ann_gap.values()))
    G["anchoring"] = dict(blind_clean_pct=a_blind, shown_pct=a_shown, gap_points=gap, per_annotator=anch,
                          per_annotator_gap=per_ann_gap, rule_fires=fires,
                          n_lines_blind=len(pooled["blind-clean"]), n_lines_shown=len(pooled["shown"]))
    # time per image
    tb, ts = [], []
    for _, recs in sets:
        for r in recs.values():
            g = AN.blind_group(r)
            (tb if g == "blind-clean" else ts if g == "shown" else []).append((r.get("time_spent_s") or 0) / 60)
    G["time_min"] = dict(blind=float(np.median(tb)) if tb else None, shown=float(np.median(ts)) if ts else None)
    # pairwise agreement on the IAA scenes, both blind-clean (analyze.py agreement() selection)
    pairs = []
    for x in range(len(sets)):
        for y in range(x + 1, len(sets)):
            (n1, r1), (n2, r2) = sets[x], sets[y]
            common = sorted(i for i in set(r1) & set(r2) if i in iaa and AN.blind_group(r1[i]) == "blind-clean"
                            and AN.blind_group(r2[i]) == "blind-clean")
            pairs.append((len(common), n1, n2, r1, r2, common))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    G["pairs"] = [dict(a=p[1], b=p[2], n_blind_common=p[0]) for p in pairs]
    if not pairs or pairs[0][0] < 2:
        ctx.warn("fewer than 2 blind scenes counted by two annotators: no agreement numbers (PLAN 3: 'agreement study ongoing')")
        G["primary"] = None
        ctx.N["coverage"]["hh_median_rel"] = None
        ctx.N["coverage"]["verdict_hh"] = "n/a"
        ctx.human_metrics = None
        return
    n, n1, n2, r1, r2, common = pairs[0]
    P = dict(a=n1, b=n2, n=n, scenes=common)
    rel_all = []
    for line in ("h", "v"):
        A = [AN.tpi(r1[i], line) for i in common]
        B = [AN.tpi(r2[i], line) for i in common]
        d = np.array(A) - np.array(B)
        m = (np.array(A) + np.array(B)) / 2
        rel = 100 * np.abs(d) / m
        rel_all += rel.tolist()
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        P[line] = dict(MAE=float(np.mean(np.abs(d))), bias=float(np.mean(d)), loa_lo=float(np.mean(d) - 1.96 * sd),
                       loa_hi=float(np.mean(d) + 1.96 * sd), icc=float(AN.icc_2_1(A, B)), r=float(AN.pearson(A, B)),
                       w5=float(100 * np.mean(rel <= 5)), w10=float(100 * np.mean(rel <= 10)),
                       mean_rel=float(np.mean(rel)), median_rel=float(np.median(rel)))
    w1 = [r1[i].get("weave") for i in common]
    w2 = [r2[i].get("weave") for i in common]
    P["weave_agree"] = float(100 * np.mean([p == q for p, q in zip(w1, w2)]))
    P["weave_kappa"] = float(AN.cohen_kappa(w1, w2))
    d1 = [r1[i].get("warp_dir") for i in common]
    d2 = [r2[i].get("warp_dir") for i in common]
    P["warp_agree"] = float(100 * np.mean([p == q for p, q in zip(d1, d2)]))
    P["median_rel_both_axes"] = float(np.median(rel_all))
    # cross-check with analyze.py's printout of the same block
    blk = out.split(f"== agreement {n1} vs {n2} (IAA subset, both blind-clean)")
    if len(blk) > 1:
        for line, lab in (("h", "H-line"), ("v", "V-line")):
            m = re.search(lab + r"[^:]*: MAE ([0-9.]+) threads/inch", blk[1])
            if m and abs(float(m.group(1)) - P[line]["MAE"]) > 0.006:
                ctx.warn(f"IAA MAE {line}: {P[line]['MAE']:.2f} here vs {m.group(1)} in analyze.py")
    att = getattr(ctx, "attention", {})
    if att and len(common) >= 5:   # does the attention level predict annotator disagreement? (IAA reps)
        rel_s = [50 * (abs(AN.tpi(r1[i], "h") - AN.tpi(r2[i], "h")) / ((AN.tpi(r1[i], "h") + AN.tpi(r2[i], "h")) / 2) +
                       abs(AN.tpi(r1[i], "v") - AN.tpi(r2[i], "v")) / ((AN.tpi(r1[i], "v") + AN.tpi(r2[i], "v")) / 2))
                 for i in common]
        sc_ = [att.get(i, (None, None))[1] for i in common]
        ok = [k for k, x in enumerate(sc_) if x is not None]
        if len(ok) >= 5:
            rho, pv = stats.spearmanr([sc_[k] for k in ok], [rel_s[k] for k in ok])
            by = {}
            for k in ok:
                by.setdefault(att[common[k]][0], []).append(rel_s[k])
            P["attention_vs_disagreement"] = dict(n=len(ok), spearman=float(rho), p=float(pv),
                                                  median_rel_by_level={l: float(np.median(v)) for l, v in by.items()},
                                                  n_by_level={l: len(v) for l, v in by.items()})
    G["primary"] = P
    hh = P["median_rel_both_axes"]
    ctx.N["coverage"]["hh_median_rel"] = hh
    ctx.N["coverage"]["verdict_hh"] = verdict(hh, THR["hh_go"], THR["hh_reduced"], higher_is_better=False)
    # Tab. 1 "Human vs human (blind)": annotator b's counts scored against annotator a's, score150.metrics()
    S = load_score150()
    ga = pd.DataFrame({"h": [AN.tpi(r1[i], "h") for i in common], "v": [AN.tpi(r1[i], "v") for i in common],
                       "weave": [S.coarse_weave(r1[i].get("weave")) for i in common]}, index=common)
    gb = pd.DataFrame({"h": [AN.tpi(r2[i], "h") for i in common], "v": [AN.tpi(r2[i], "v") for i in common],
                       "weave": [S.coarse_weave(r2[i].get("weave")) for i in common]}, index=common)
    ctx.human_metrics = S.metrics(ga, gb)
    ctx.human_metrics["reference"] = n1
    G["human_row"] = ctx.human_metrics
    ctx.log(f"  blind agreement {n1} vs {n2} on {n} scenes: MAE h {P['h']['MAE']:.2f} / v {P['v']['MAE']:.2f} TPI, "
            f"ICC {P['h']['icc']:.3f} / {P['v']['icc']:.3f}, median |diff| {hh:.2f}% -> "
            f"{ctx.N['coverage']['verdict_hh']} (GO <= {THR['hh_go']}%)")
    ctx.log(f"  anchoring |final - proposal| / final: blind-clean {fnum(a_blind)}%, shown {fnum(a_shown)}%, gap "
            f"{fnum(gap)} points -> rule {'FIRES (headline uses --blind-only)' if fires else 'does not fire'}")


# ============================================================================ step 4: score
_S = None


def load_score150():
    global _S
    if _S is None:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        _S = load_module("score150", os.path.join(HERE, "score150.py"))
    return _S


def extra_sources(ctx):
    specs = []
    for rel, keys in EXTRA_SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            raise StepError(f"prediction file missing: {rel}")
        cols = pd.read_csv(p, nrows=1).columns
        avail = [c[:-3] for c in cols if c.endswith("__v") and c[:-3] + "__h" in cols]
        miss = [k for k in (keys or []) if k not in avail]
        if miss:
            raise StepError(f"{rel} has no columns for {miss}")
        specs.append(p + ("::" + ",".join(keys) if keys else ""))
    # the cascade row must be the FROZEN gate (tables/cascade_150_decisions.csv, md5-checked in setup)
    xw = pd.read_csv(os.path.join(PAPER, "tables", "src", "extra150_wide.csv")).set_index("image_id")
    dec = pd.read_csv(P_DECISIONS).set_index("image")
    dv = (xw["cascade-gate__v"] - dec.cascade_v_yarns.reindex(xw.index)).abs().max()
    dh = (xw["cascade-gate__h"] - dec.cascade_h_yarns.reindex(xw.index)).abs().max()
    if not (dv < 1e-3 and dh < 1e-3):
        raise StepError(f"tables/src/extra150_wide.csv cascade-gate differs from the frozen decisions ({dv}, {dh})")
    # robustness check: pseudo-label CNN without the probable re-shoots of its training scene 60
    h = pd.read_csv(P_HELDOUT).set_index("image_id")
    sc = ctx.scenes
    drop = [int(i) for i in sc.index if int(sc.at[i, "scene_id"]) in RESHOOT_SCENES]
    for s in RESHOOT_SCENES:
        if s not in set(sc.scene_id):
            raise StepError(f"scene {s} (REVIEW G-P0-3) is not a scene id of the manifest")
    w = pd.DataFrame({f"{PSEUDO_KEY}@39__v": h[f"{PSEUDO_KEY}__v"], f"{PSEUDO_KEY}@39__h": h[f"{PSEUDO_KEY}__h"],
                      f"{PSEUDO_KEY}@39__weave": h[f"{PSEUDO_KEY}__weave"]})
    w.loc[w.index.isin(drop), :] = np.nan
    w.index.name = "image_id"
    p46 = os.path.join(ctx.work, "extra_pseudo46_wide.csv")
    w.to_csv(p46)
    specs.append(p46)
    reps = sc[sc.is_scene_rep].index
    ctx.N["score"]["pseudo46"] = dict(heldout_reps=int(h.loc[reps, f"{PSEUDO_KEY}__v"].notna().sum()),
                                      after_drop=int(w.loc[reps, f"{PSEUDO_KEY}@39__v"].notna().sum()),
                                      dropped_files=drop)
    return specs + list(ctx.cv_wides)


def check_vlm_long(ctx):
    """The P1 roster is given as the LONG file; score150 reads the WIDE one: check they agree."""
    if not os.path.exists(P_VLM_LONG):
        return
    lg = pd.read_csv(P_VLM_LONG)
    wd = pd.read_csv(os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")).set_index("image_id")
    bad = 0
    for _, r in lg.iterrows():
        c = f"{r.model}__v"
        if c in wd and r.image_id in wd.index:
            a, b = r.vertical, wd.at[r.image_id, c]
            if not ((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) < 1e-6)):
                bad += 1
    if bad:
        ctx.warn(f"{bad} P1 predictions differ between the long and the wide VLM files (score150 uses the wide one)")
    else:
        ctx.log(f"  VLM P1 long file agrees with the wide file ({len(lg)} rows)")


def build_scene_gt(ctx, blind_only=False, paths=None, rescaled=False, annotator=None, exclude=None):
    """The reference table exactly as score150.py main() builds it (--unit scene). paths/rescaled: the partial-line
    sensitivity (ctx.partial_csv, counts taken from the separate <ax>_tpi_rescaled columns). annotator: one annotator's
    records only (score150 load_labels(annotator=); dual protocol: agreement, each annotator alone as the reference).
    exclude: set of (annotator, id) records to leave out (dual protocol: proven proposal-free records only)."""
    S = load_score150()
    rec = S.load_labels(paths or ctx.scoring_csv, blind_only, annotator)
    if exclude:
        rec = rec[np.array([(str(a_), int(i)) not in exclude for a_, i in zip(rec["annotator"], rec["id"])], bool)]
    if rescaled:
        for ax in ("h", "v"):
            rec[f"{ax}_count"] = pd.to_numeric(rec[f"{ax}_tpi_rescaled"], errors="coerce").where(lambda x: x > 0)
    if ctx.man.get("iaa_subset"):
        rec["iaa"] = rec["id"].astype(int).isin(set(ctx.man["iaa_subset"])).astype(int)
    gt = S.gt_table(rec)
    gt.index = gt.index.astype(int)
    _, gt_scene = ctx.PS.propagate(gt, ctx.scenes)
    gt = gt_scene.drop(columns=["label_src_kind"], errors="ignore")
    gt.index = gt.index.astype(int)
    side = {im["id"]: im["rect_size"][0] for im in ctx.man["images"]}
    gt["S"] = [side.get(int(i)) for i in gt.index]
    return gt, rec


def per_scene_ape(gt, pr):
    j = gt[["h", "v"]].join(pr[["h", "v"]], rsuffix="_p", how="inner").dropna()
    eh = 100 * (j.h_p - j.h) / j.h
    ev = 100 * (j.v_p - j.v) / j.v
    return pd.DataFrame({"ape_h": eh.abs(), "ape_v": ev.abs(), "se_h": eh, "se_v": ev,
                         "ape": 0.5 * (eh.abs() + ev.abs()), "h": j.h, "v": j.v, "h_p": j.h_p, "v_p": j.v_p})


def paired(gt, pa, pb, n_boot=10000, seed=0):
    """Paired difference of per-scene mean APE (method a - method b) on the scenes both answered."""
    ea, eb = per_scene_ape(gt, pa).ape, per_scene_ape(gt, pb).ape
    c = ea.index.intersection(eb.index)
    out = dict(n=int(len(c)))
    if len(c) < 3:
        return out
    d = (ea[c] - eb[c]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    out.update(mape_a=float(ea[c].mean()), mape_b=float(eb[c].mean()), delta=float(d.mean()),
               lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_wilcoxon=p,
               a_closer=int((d < 0).sum()), b_closer=int((d > 0).sum()))
    return out


def mrow(m, keys=("n_gt", "n_scored", "MAE", "MAPE", "MAPE_lo", "MAPE_hi", "w5", "w10", "both5", "both10", "bias_med",
                  "spread_mad", "weave_acc", "weave_n")):
    return {k: (float(m[k]) if isinstance(m.get(k), (int, float, np.floating, np.integer)) else m.get(k))
            for k in keys if k in m}


def step_score(ctx):
    S = load_score150()
    Sc = ctx.N["score"]
    check_vlm_long(ctx)
    ctx.extra_specs = extra_sources(ctx)
    lab = []
    for p in ctx.scoring_csv:
        lab += ["--labels", p]
    xw = []
    for s in ctx.extra_specs:
        xw += ["--extra-wide", s]
    st = ctx.stage
    d_all, d_blind, d_q = (os.path.join(st, x) for x in ("score", "score_blind", "quality"))
    sc150 = os.path.join(HERE, "score150.py")
    ctx.run([sys.executable, sc150, *lab, "--fig", "--fig-dir", os.path.join(d_all, "fig"), "--out", d_all,
             "--groups", P_GROUPS, *xw], capture=os.path.join(st, "score150_stdout.txt"))
    blind_ok = True
    if ctx.oct11:   # single/dual: every count was made without proposals -> the blind-only check is vacuous
        blind_ok, rc = False, 0
        ctx.log(f"  {'single-annotator' if ctx.single else 'dual'} protocol: no --blind-only run (every count was made "
                "without proposals)")
    else:
        rc, _ = ctx.run([sys.executable, sc150, *lab, "--blind-only", "--fig", "--fig-dir", os.path.join(d_blind, "fig"),
                         "--out", d_blind, *xw], capture=os.path.join(st, "score150_blind_stdout.txt"), check=False)
    if rc != 0:
        blind_ok = False
        ctx.warn("score150.py --blind-only failed (no blind records yet?): blind-only check skipped")
    ctx.run([sys.executable, os.path.join(HERE, "quality150.py"), *lab, "--out", d_q],
            capture=os.path.join(st, "quality150_stdout.txt"))
    ctx.blind_ok = blind_ok
    # ---------------------------------------------------------------- in-process analyses (same GT as score150)
    gt, rec = build_scene_gt(ctx)
    summ = json.load(open(os.path.join(d_all, "data150_summary.json")))
    if summ["n_images"] != len(gt):
        raise StepError(f"label_day's reference table has {len(gt)} scenes, score150.py scored {summ['n_images']}")
    ctx.gt = gt
    preds = S.load_predictions()
    preds.update(S.load_extra_wide(ctx.extra_specs))
    preds[LOO] = S.loo_mean(gt)
    for k in list(preds):
        preds[k] = preds[k].copy()
        preds[k].index = preds[k].index.astype(int)
    ctx.preds = preds
    main = pd.read_csv(os.path.join(d_all, "main150.csv"))
    ctx.main150 = main
    Sc["n_scenes"] = int(len(gt))
    Sc["annotators"] = summ["annotators"]
    Sc["n_scenes_2plus"] = summ["n_images_2plus_annotators"]
    Sc["human_weave"] = summ["weave"]
    Sc["h_range"], Sc["v_range"] = summ["h_min_med_max"], summ["v_min_med_max"]
    Sc["median_time_s"] = summ["median_time_s"]
    Sc["methods"] = {r.method: mrow(r._asdict()) for r in main.itertuples(index=False)}
    # per-scene errors of every method (for the worst-scene lists and traceability)
    per = []
    for name, pr in preds.items():
        e = per_scene_ape(gt, pr)
        e.insert(0, "method", name)
        per.append(e.reset_index().rename(columns={"index": "scene_id"}))
    per = pd.concat(per, ignore_index=True)
    per["resolution"] = [ctx.scenes.at[int(s), "scene_resolution"] for s in per.scene_id]
    per.round(3).to_csv(os.path.join(st, "main150_per_scene.csv"), index=False)
    ctx.per_scene = per
    # paired tests (scene-resampled bootstrap, Wilcoxon on per-scene mean APE)
    pairs = [(OURS, GPT), ("cascade-gate", GPT), ("cascade-gate", OURS), ("pg-lattice@ruler", OURS),
             (PSEUDO_KEY, OURS), (PSEUDO_KEY, CNN0), (f"{PSEUDO_KEY}@39", OURS), (OURS, "Log-Gabor bank"),
             (OURS, LOO), (CV_PRIMARY, OURS),
             ("combined-v1", OURS), ("combined-v1", "Log-Gabor bank"), ("combined-v1", GPT)]   # PREREG_combined_v1.json key label_day_plan
    Sc["paired"] = {}
    for a_, b_ in pairs:
        if a_ in preds and b_ in preds:
            Sc["paired"][f"{a_} vs {b_}"] = paired(gt, preds[a_], preds[b_])
    # subsets
    res = pd.Series({int(i): ctx.scenes.at[int(i), "scene_resolution"] for i in gt.index})
    subsets = {"full-resolution scenes": gt.index[res.reindex(gt.index) == "full"],
               "quarter-only scenes": gt.index[res.reindex(gt.index) == "quarter_only"]}
    unknown13 = []
    if os.path.exists(P_PSEUDO):
        ps = pd.read_csv(P_PSEUDO)
        unknown13 = sorted(int(s) for s in ps[ps.is_rep & (ps.weave_ours == "unknown")].scene)
        subsets["13 'unknown'-weave scenes"] = gt.index[gt.index.isin(unknown13)]
        subsets["other scenes (weave known to ours)"] = gt.index[~gt.index.isin(unknown13)]
    if ctx.N["coverage"].get("blind_double_scenes") and not ctx.oct11:   # single/dual: the IAA subset has no role
        subsets["20 blind IAA scenes"] = gt.index[gt.index.isin(ctx.iaa)]
    # pre-declared: scores without the 4 guide-example scenes (their machine dots were in the annotator guide),
    # without the badge scenes, and on the proven proposal-free records only (sensitivity checks)
    subsets[GUIDE_SUBSET] = gt.index[~gt.index.isin(GUIDE_EXAMPLE_SCENES)]
    hint_sc = {int(ctx.scenes.at[i, "scene_id"]) for i in ctx.hint_ids if i in ctx.scenes.index}
    if hint_sc:
        subsets[f"{len(hint_sc)} badge scenes ('check carefully' / 'take care')"] = gt.index[gt.index.isin(hint_sc)]
        subsets["scenes without a badge"] = gt.index[~gt.index.isin(hint_sc)]
    nf = ctx.N["validation"].get("proposal_free", {}).get("not_free") or []
    if ctx.single and nf:
        nf_sc = {int(ctx.scenes.at[d["id"], "scene_id"]) for d in nf}
        subsets["proven proposal-free records only"] = gt.index[~gt.index.isin(nf_sc)]
    sub_rows = []
    for sname, idx in subsets.items():
        for name in [m for m in KEY_METHODS if m in preds]:
            m = S.metrics(gt.loc[idx], preds[name], n_boot=2000) if len(idx) else {}
            sub_rows.append(dict(subset=sname, method=name, **mrow(m)))
    if ctx.dual:
        # dual protocol: the reference rebuilt from the proven proposal-free records only (a scene stays when the
        # other annotator's record is free), and each annotator alone as the reference (Tab. 1 = the mean of the two)
        if nf:
            gtf, _ = build_scene_gt(ctx, exclude=ctx.not_free)
            lab_f = f"proven proposal-free records only ({len(gtf)} scenes)"
            for name in [m for m in KEY_METHODS if m in preds]:
                sub_rows.append(dict(subset=lab_f, method=name, **mrow(S.metrics(gtf, preds[name], n_boot=2000))))
        Sc["reference_alone"] = {}
        for n_ in ctx.ann_order:
            gt_n, _ = build_scene_gt(ctx, annotator=n_)
            gt_n = gt_n[gt_n[["h", "v"]].notna().all(axis=1)]
            if gt_n.empty:
                continue
            lab_n = f"reference: {n_} alone ({len(gt_n)} scenes)"
            for name in [m for m in KEY_METHODS if m in preds]:
                sub_rows.append(dict(subset=lab_n, method=name, **mrow(S.metrics(gt_n, preds[name], n_boot=2000))))
            Sc["reference_alone"][n_] = dict(label=lab_n, n_scenes=int(len(gt_n)),
                                             paired={f"{a_} vs {b_}": paired(gt_n, preds[a_], preds[b_])
                                                     for a_, b_ in ((OURS, GPT), ("cascade-gate", GPT),
                                                                    (OURS, "Log-Gabor bank"))
                                                     if a_ in preds and b_ in preds})
    # partly counted lines (module docstring): the headline excludes them; the sensitivity rows add the justified
    # ones back with the short line scored over its counted part (separate <ax>_tpi_rescaled columns)
    k_part = ctx.N["coverage"].get("partial_justified", 0)
    if k_part and ctx.partial_csv:
        gts, _ = build_scene_gt(ctx, paths=ctx.partial_csv, rescaled=True)
        lab = f"sensitivity: + {k_part} partial-line record(s), short line rescaled ({len(gts)} scenes)"
        for name in [m for m in KEY_METHODS if m in preds]:
            sub_rows.append(dict(subset=lab, method=name, **mrow(S.metrics(gts, preds[name], n_boot=2000))))
        added = sorted(int(x) for x in gts.index.difference(gt.index))
        changed = sorted(int(x) for x in gts.index.intersection(gt.index)
                         if not np.allclose(gts.loc[x, ["h", "v"]].astype(float), gt.loc[x, ["h", "v"]].astype(float)))
        Sc["partial_sensitivity"] = dict(label=lab, n_scenes=int(len(gts)), scenes_added=added, scenes_changed=changed)
    Sc["subsets"] = sub_rows
    # the 13 'unknown' scenes: did the pseudo-label CNN correct or inherit ours' errors?
    if unknown13 and PSEUDO_KEY in preds:
        j = per[(per.scene_id.isin(unknown13))]
        o = j[j.method == OURS].set_index("scene_id")
        c = j[j.method == PSEUDO_KEY].set_index("scene_id")
        cc = o.index.intersection(c.index)
        if len(cc):
            follow = 100 * np.median(np.abs(0.5 * ((c.loc[cc, "h_p"] - o.loc[cc, "h_p"]) / o.loc[cc, "h_p"] +
                                                   (c.loc[cc, "v_p"] - o.loc[cc, "v_p"]) / o.loc[cc, "v_p"])))
            Sc["unknown13"] = dict(n=int(len(cc)), n_labelled=int(len(o)), ours_mape=float(o.loc[cc, "ape"].mean()),
                                   cnn_mape=float(c.loc[cc, "ape"].mean()),
                                   ours_median_signed=float(np.median(np.r_[o.loc[cc, "se_h"], o.loc[cc, "se_v"]])),
                                   cnn_median_signed=float(np.median(np.r_[c.loc[cc, "se_h"], c.loc[cc, "se_v"]])),
                                   cnn_vs_ours_median_abs_diff=float(follow), scenes=[int(x) for x in cc])
    # cascade frozen gate (all 76 scenes are held out: thresholds fitted on synthetic crops only)
    dec = pd.read_csv(P_DECISIONS).set_index("image")
    cs = gt.index.intersection(dec.index)
    esc = dec.loc[cs, "escalate_gate"].astype(bool)
    casc = dict(n=int(len(cs)), escalated=int(esc.sum()), escalated_pct=float(100 * esc.mean()) if len(cs) else None,
                usd_per_1000_cascade=float(1000 * dec.loc[cs[esc.to_numpy()], "gpt_usd"].sum() / max(1, len(cs))),
                usd_per_1000_gpt_only=float(1000 * dec.loc[cs, "gpt_usd"].sum() / max(1, len(cs))),
                escalated_scenes=[int(x) for x in cs[esc.to_numpy()]])
    for nm, idx in (("escalated", cs[esc.to_numpy()]), ("kept_offline", cs[~esc.to_numpy()])):
        for meth in (OURS, GPT, "cascade-gate"):
            if meth in preds and len(idx):
                casc[f"{nm}|{meth}"] = mrow(S.metrics(gt.loc[idx], preds[meth], n_boot=1000))
    Sc["cascade"] = casc
    # S6 strata for ours (and the key methods): flags, px per yarn of the native photo, resolution class
    strata = pd.read_csv(P_STRATA).set_index("id")
    ppy = strata.S.reindex(gt.index) / gt[["h", "v"]].max(axis=1)
    groups = {"all": gt.index}
    for f in ("glare", "blur", "shadow", "low_res", "perspective"):
        groups[f"flag {f}"] = gt.index[gt[f"flag_{f}"] == 1]
    groups["<4 px/yarn"] = gt.index[ppy < 4]
    groups["4-8 px/yarn"] = gt.index[(ppy >= 4) & (ppy < 8)]
    groups[">=8 px/yarn"] = gt.index[ppy >= 8]
    groups["full-resolution"] = subsets["full-resolution scenes"]
    groups["quarter-only"] = subsets["quarter-only scenes"]
    srows = []
    for g, idx in groups.items():
        for name in (OURS, GPT, "cascade-gate"):
            m = S.metrics(gt.loc[idx], preds[name], n_boot=1000) if len(idx) else {}
            srows.append(dict(stratum=g, method=name, n=int(len(idx)), **mrow(m)))
    Sc["strata"] = srows
    # blind-only check and ranking agreement on the SAME scenes (isolates anchoring from the scene set)
    if blind_ok:
        gtb, _ = build_scene_gt(ctx, blind_only=True)
        ctx.gt_blind = gtb
        rk = []
        for name, pr in preds.items():
            if name == LOO:
                continue
            ma = S.metrics(gt.loc[gt.index.intersection(gtb.index)], pr, n_boot=200)
            mb = S.metrics(gtb, pr, n_boot=200)
            if ma.get("n_scored", 0) >= 5 and mb.get("n_scored", 0) >= 5:
                rk.append((name, ma["MAPE"], mb["MAPE"]))
        mb = pd.read_csv(os.path.join(d_blind, "main150.csv"))
        Sc["methods_blind"] = {r.method: mrow(r._asdict()) for r in mb.itertuples(index=False)}
        Sc["n_scenes_blind"] = int(len(gtb))
        Sc["paired_blind"] = {f"{a_} vs {b_}": paired(gtb, preds[a_], preds[b_]) for a_, b_ in pairs
                              if a_ in preds and b_ in preds}
        if len(rk) >= 3:
            r = stats.spearmanr([x[1] for x in rk], [x[2] for x in rk])[0]
            top_a = [x[0] for x in sorted(rk, key=lambda t: t[1])[:5]]
            top_b = [x[0] for x in sorted(rk, key=lambda t: t[2])[:5]]
            Sc["blind_rank"] = dict(n_scenes=int(len(gtb)), n_methods=len(rk), spearman=float(r), top5_all=top_a,
                                    top5_blind=top_b, top1_same=top_a[0] == top_b[0])
    # per-file (secondary) and fabric-group sensitivity for the key methods
    pf = pd.read_csv(os.path.join(d_all, "main150_perfile.csv"))
    # n_files / n_scenes are the labelled totals; n_scored / n_clusters are what the method answered
    Sc["perfile"] = {r.method: mrow(r._asdict()) | {"n_files": int(r.n_files), "n_scenes": int(r.n_scenes),
                                                    "n_clusters": int(r.n_clusters) if pd.notna(r.n_clusters) else 0}
                     for r in pf[(pf.subset == "all real files") & pf.method.isin(KEY_METHODS)].itertuples(index=False)}
    fb = os.path.join(d_all, "main150_fabric.csv")
    if os.path.exists(fb):
        fab = pd.read_csv(fb).set_index("method")
        Sc["fabric"] = {m: dict(MAPE_lo=float(fab.at[m, "MAPE_lo"]), MAPE_hi=float(fab.at[m, "MAPE_hi"]),
                                n_clusters=int(fab.at[m, "n_clusters"]) if "n_clusters" in fab else None)
                        for m in KEY_METHODS if m in fab.index}
    # confidence vs error (quality150.py)
    cf = os.path.join(d_q, "conf150.csv")
    if os.path.exists(cf):
        Sc["confidence"] = pd.read_csv(cf).to_dict("records")
    # distortion basis: where the humans' guide lines sit, and what that implies
    Sc["distortion"] = distortion_basis(ctx, gt)
    # worst scenes per key method (for spot checks)
    worst = {}
    for name in [m for m in KEY_METHODS if m in preds]:
        j = per[per.method == name].sort_values("ape", ascending=False).head(10)
        worst[name] = [dict(scene=int(r.scene_id), res=r.resolution, h=r.h, v=r.v, h_p=r.h_p, v_p=r.v_p,
                            ape=round(float(r.ape), 1)) for r in j.itertuples()]
    Sc["worst"] = worst
    # headline choice (pre-declared: anchoring rule fires -> headline uses --blind-only)
    ah = ctx.a.headline
    fires = ctx.N["agreement"].get("anchoring", {}).get("rule_fires")
    ctx.headline = "blind" if (ah == "blind" or (ah == "auto" and fires and blind_ok)) else "all"
    if ctx.oct11 and ctx.headline == "blind":   # single/dual: no blind/assisted split exists (no --blind-only run)
        ctx.warn(f"--headline blind ignored: {ctx.protocol} protocol, every count was made without proposals")
        ctx.headline = "all"
    if ah == "auto" and fires and not blind_ok:
        ctx.warn("anchoring rule fires but the blind-only scoring failed: Tab. 1 uses all labels")
    Sc["headline"] = ctx.headline
    blind = ctx.headline == "blind"
    Sc["methods_headline"] = Sc["methods_blind"] if blind else Sc["methods"]
    Sc["paired_headline"] = Sc["paired_blind"] if blind else Sc["paired"]
    Sc["n_headline"] = Sc["n_scenes_blind"] if blind else Sc["n_scenes"]
    o = Sc["methods_headline"].get(OURS, {})
    ctx.log(f"  headline ({ctx.headline} labels): n = {Sc['n_headline']} scenes; ours MAPE {fnum(o.get('MAPE'))}% "
            f"[{fnum(o.get('MAPE_lo'))}, {fnum(o.get('MAPE_hi'))}], both within 5% {fnum(o.get('both5'), 0)}%")
    pg = Sc["paired_headline"].get(f"{OURS} vs {GPT}")
    if pg and "delta" in pg:
        ctx.log(f"  ours - GPT-6.1-sol: {pg['delta']:+.1f} points [{pg['lo']:+.1f}, {pg['hi']:+.1f}] on {pg['n']} scenes, "
                f"Wilcoxon p = {pg['p_wilcoxon']:.3g}")


def distortion_basis(ctx, gt):
    """Per human count, the radial-distortion factor of its guide line (supp S16: a line at normalised offset y
    spans g(sqrt(1+y^2))/g(sqrt2) of fabric, g = 1 + k1 r^2) and the area-mean factor; with the
    assumption (--lattice-distortion-mult) that PG-Lattice reads m x the area-mean excess, the distortion-only offset of ours vs the humans."""
    k1, mult = ctx.a.k1, ctx.a.lattice_distortion_mult
    f_area = (1 + 4.0 / 3.0 * k1) / (1 + 2 * k1)
    fl = []
    for p in ctx.scoring_json:
        for r in json.load(open(p))["records"]:
            i = int(r["id"])
            if not r.get("done") or i not in gt.index or not r.get("rect_size"):
                continue
            Sz = float(r["rect_size"])
            for line, pos in (("h", r.get("h_line_y")), ("v", r.get("v_line_x"))):
                if pos is None:
                    continue
                y = (float(pos) - Sz / 2) / (Sz / 2)
                fl.append((1 + k1 * (1 + y * y)) / (1 + 2 * k1))
    if not fl:
        return None
    fl = np.array(fl)
    exp_off = 100 * ((1 + mult * (f_area - 1)) / fl - 1)
    o = ctx.N["score"]["methods"].get(OURS, {})
    return dict(k1=k1, mult=mult, n_counts=int(len(fl)), human_line_excess_median=float(100 * (np.median(fl) - 1)),
                area_mean_excess=float(100 * (f_area - 1)), ours_expected_offset_median=float(np.median(exp_off)),
                ours_bias_med_observed=o.get("bias_med"))


# ============================================================================ step 5: CNN cross-validation
def cnn_cmd_args(ctx, cfg, synth, labels, out):
    args = ["train_cv.py", "--labels", ",".join(labels), "--out", out, "--synth-ckpt", f"resnet18={synth}",
            "--backbones", ctx.a.cnn_backbones, "--epochs", str(ctx.a.cnn_epochs), "--seed", "0"]
    args += CNN_CONFIGS[cfg]
    if ctx.a.cnn_args:
        args += shlex.split(ctx.a.cnn_args)
    return args


def ssh(ctx, script, check=True, quiet=False):
    return ctx.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SSH_HOST, script], check=check,
                   quiet=quiet)


def remote_preflight(ctx):
    """Read-only checks on gpu-server: env, venv, synthetic checkpoint md5, disk, GPUs (idle = no compute process,
    < 500 MiB, < 5 % util)."""
    _, out = ssh(ctx, f"set -e; test -f ~/{REMOTE_CA}/env.sh; test -x ~/{REMOTE_CA}/venv/bin/python; "
                      f"md5sum ~/{REMOTE_SYNTH} | cut -c1-32; df -Pk ~ | tail -1 | awk '{{print $4}}'; "
                      "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits; "
                      "echo APPS; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader", quiet=True)
    lines = [x.strip() for x in out.strip().splitlines()]
    if lines[0] != CNN_SYNTH_MD5:
        raise StepError(f"gpu-server synthetic checkpoint md5 {lines[0]} != local {CNN_SYNTH_MD5}")
    free_gb = int(lines[1]) / 1e6
    k = lines.index("APPS")
    gpus, apps = {}, {l.split(",")[0].strip() for l in lines[k + 1:] if l}
    for l in lines[2:k]:
        i, uuid, mem, util = [x.strip() for x in l.split(",")]
        gpus[int(i)] = dict(uuid=uuid, mem=int(mem), util=int(util), idle=int(mem) < 500 and int(util) < 5
                            and uuid not in apps)
    ctx.log(f"  gpu-server: synthetic checkpoint md5 OK, {free_gb:.0f} GB free; GPUs: " +
            ", ".join(f"{i}:{'idle' if g['idle'] else 'busy'}({g['mem']} MiB, {g['util']}%)" for i, g in gpus.items()))
    if free_gb < 5:
        raise StepError(f"gpu-server has only {free_gb:.1f} GB free")
    return gpus


CHAIN = r"""#!/bin/bash
# label_day.py CNN chain for GPU {gpu}: runs its configs one after another, each only once the GPU is idle
# (no compute process of anyone, < 500 MiB, < 5 % util). Never kills anything.
source ~/{ca}/env.sh
cd ~/{ld}/wvaq/cnn
RUNS=~/{ld}/runs/{stamp}
idle() {{
  local uuid used util
  read -r used util uuid < <(nvidia-smi -i {gpu} --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits | tr -d ",")
  [ "$used" -lt 500 ] && [ "$util" -lt 5 ] && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
}}
for cfg in {cfgs}; do
  echo "$(date '+%Y-%m-%dT%H:%M:%S') waiting for an idle GPU {gpu} for $cfg"
  until idle; do sleep 60; done
  echo "$(date '+%Y-%m-%dT%H:%M:%S') start $cfg"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu} bash $RUNS/$cfg.cmd.sh > $RUNS/$cfg.log 2>&1 < /dev/null &
  echo $! > $RUNS/$cfg.pid
  wait $!
  echo $? > $RUNS/$cfg.exit
  echo "$(date '+%Y-%m-%dT%H:%M:%S') end $cfg exit $(cat $RUNS/$cfg.exit)"
done
"""


def cnn_launch_remote(ctx, cfgs, dry):
    gpus = remote_preflight(ctx)
    cand = [int(g) for g in ctx.a.cnn_gpus.split(",") if g.strip()]
    idle = [g for g in cand if gpus.get(g, {}).get("idle")]
    use = idle or [g for g in cand if g in gpus]
    if not use:
        raise StepError(f"none of the GPUs {cand} exists on gpu-server")
    if not idle:
        ctx.warn(f"no candidate GPU {cand} is idle now; the chains will wait until one is")
    stamp = ctx.stamp
    ld = REMOTE_LD
    remote_labels = [f"$HOME/{ld}/labels/{stamp}/{os.path.basename(p)}" for p in ctx.scoring_csv]
    assign = {g: [] for g in use}
    for n, c in enumerate(cfgs):
        assign[use[n % len(use)]].append(c)
    need_aug = any("--extra-aug1016" in CNN_CONFIGS[c] for c in cfgs)
    sync = [
        ["rsync", "-a", "--include=*.py", "--include=*.json", "--exclude=*", os.path.join(ROOT, "cnn") + "/",
         f"{SSH_HOST}:{ld}/wvaq/cnn/"],
        ["rsync", "-a", os.path.join(TOOL, "propagate_scene_labels.py"), f"{SSH_HOST}:{ld}/wvaq/annotation_tool/"],
        ["rsync", "-a", MANIFEST, f"{SSH_HOST}:{ld}/wvaq/annotation_tool/data/"],
        ["rsync", "-a", os.path.join(TOOL, "data", "crops") + "/", f"{SSH_HOST}:{ld}/wvaq/annotation_tool/data/crops/"],
        ["rsync", "-a", os.path.join(ROOT, "methods", "predictions_ours_150.csv"), f"{SSH_HOST}:{ld}/wvaq/methods/"],
        ["rsync", "-a", *ctx.scoring_csv, f"{SSH_HOST}:{ld}/labels/{stamp}/"],
    ]
    if need_aug:
        A = os.path.join(ROOT, "aug1016")
        sync += [["rsync", "-a", os.path.join(A, "propagate_labels.py"), os.path.join(A, "manifest.csv"),
                  f"{SSH_HOST}:{ld}/wvaq/aug1016/"],
                 ["rsync", "-a", os.path.join(A, "review", "review_decisions.csv"), f"{SSH_HOST}:{ld}/wvaq/aug1016/review/"],
                 ["rsync", "-a", os.path.join(A, "crops") + "/", f"{SSH_HOST}:{ld}/wvaq/aug1016/crops/"]]
    mk = (f"mkdir -p ~/{ld}/wvaq/cnn ~/{ld}/wvaq/annotation_tool/data/crops ~/{ld}/wvaq/methods "
          f"~/{ld}/wvaq/aug1016/review ~/{ld}/labels/{stamp} ~/{ld}/runs/{stamp}")
    def dq(x):   # bash double quotes: only the literal $HOME expands
        x = x.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
        return '"' + x.replace("\\$HOME", "$HOME") + '"'
    argfiles = {c: "#!/bin/bash\n# label_day.py: real-label scene-grouped CV, config " + c + " (env.sh is sourced by the chain)\n"
                   + "cd ~/" + ld + "/wvaq/cnn && exec \"$PY\" " +
                   " ".join(dq(x) for x in cnn_cmd_args(ctx, c, f"$HOME/{REMOTE_SYNTH}", remote_labels,
                                                         f"$HOME/{ld}/runs/{stamp}/{c}")) + "\n"
                for c in cfgs}
    if dry:
        ctx.log("  --cnn-dry: would run on gpu-server (nothing synced or launched):")
        ctx.log("    ssh " + SSH_HOST + " '" + mk + "'")
        for s in sync:
            ctx.log("    " + " ".join(shlex.quote(x) for x in s))
        for g, cs in assign.items():
            ctx.log(f"    GPU {g} chain (waits for the GPU to be idle before each config): {cs}")
            for c in cs:
                ctx.log(f"      {c}.cmd.sh: " + argfiles[c].splitlines()[-1])
        return None
    if any(looks_fake(n) for n in ctx.N["validation"]["annotators"]):
        raise StepError("refusing to upload labels that look like test data to gpu-server")
    ssh(ctx, mk)
    for s in sync:
        ctx.run(s)
    state = dict(stamp=stamp, host="gpu-server", remote_runs=f"{ld}/runs/{stamp}", configs={}, labels=ctx.scoring_csv,
                 label_md5={os.path.basename(p): md5(p) for p in ctx.scoring_csv}, launched=_dt.datetime.now().isoformat())
    for g, cs in assign.items():
        chain = CHAIN.format(gpu=g, ca=REMOTE_CA, ld=ld, stamp=stamp, cfgs=" ".join(cs))
        local = os.path.join(ctx.work, f"chain_gpu{g}.sh")
        with open(local, "w") as f:
            f.write(chain)
        ctx.run(["rsync", "-a", local, f"{SSH_HOST}:{ld}/runs/{stamp}/"])
        for c in cs:
            af = os.path.join(ctx.work, f"{c}.cmd.sh")
            with open(af, "w") as f:
                f.write(argfiles[c])
            ctx.run(["rsync", "-a", af, f"{SSH_HOST}:{ld}/runs/{stamp}/"])
            state["configs"][c] = dict(gpu=g, local_dir=os.path.join(ctx.cnn_local, f"cv_labelday_{stamp}_{c}"))
        ssh(ctx, f"cd ~/{ld}/runs/{stamp} && bash -n chain_gpu{g}.sh && "
                 f"nohup bash chain_gpu{g}.sh > chain_gpu{g}.log 2>&1 < /dev/null & echo $! > ~/{ld}/runs/{stamp}/chain_gpu{g}.pid")
    return state


def cnn_poll_remote(ctx, state):
    cfgs = list(state["configs"])
    script = (f"cd ~/{state['remote_runs']} && for c in {' '.join(cfgs)}; do printf '%s ' $c; "
              "if [ -f $c.exit ]; then echo EXIT $(cat $c.exit); "
              "elif [ -f $c.pid ] && ps -p $(cat $c.pid) > /dev/null; then echo RUNNING $(tail -c 400 $c.log | tr '\\n\\r' '  ' | tail -c 160); "
              "elif [ -f $c.pid ]; then echo DIED; else echo QUEUED; fi; done")
    _, out = ssh(ctx, script, quiet=True)
    st = {}
    for line in out.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2 and parts[0] in state["configs"]:
            st[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return st


def cnn_pull_remote(ctx, state):
    for c, info in state["configs"].items():
        local = info["local_dir"]
        os.makedirs(local, exist_ok=True)
        ctx.run(["rsync", "-a", "--exclude=*.pt", "--exclude=ckpts/",
                 f"{SSH_HOST}:{state['remote_runs']}/{c}/", local + "/"])
        ctx.run(["rsync", "-a", f"{SSH_HOST}:{state['remote_runs']}/{c}.log", f"{SSH_HOST}:{state['remote_runs']}/{c}.exit",
                 local + "/"], check=False)


def cnn_run_local(ctx, cfgs):
    if not os.path.exists(DEF_CNN_PY if not ctx.a.cnn_python else ctx.a.cnn_python):
        raise StepError(f"CNN python not found: {ctx.a.cnn_python or DEF_CNN_PY}")
    py = ctx.a.cnn_python or DEF_CNN_PY
    state = dict(stamp=ctx.stamp, host="local", configs={}, labels=ctx.scoring_csv,
                 label_md5={os.path.basename(p): md5(p) for p in ctx.scoring_csv}, launched=_dt.datetime.now().isoformat())
    env = ctx.env()
    env["TORCH_HOME"] = os.path.join(ctx.work, "torch_home")
    env["HF_HOME"] = os.path.join(ctx.work, "hf_home")
    for c in cfgs:
        out = os.path.join(ctx.cnn_local, f"cv_labelday_{ctx.stamp}_{c}")
        ctx._guard(os.path.join(out, "x"))
        args = cnn_cmd_args(ctx, c, CNN_SYNTH_LOCAL, ctx.scoring_csv, out)
        ctx.run([py, *args], cwd=os.path.join(ROOT, "cnn"), env=env)
        state["configs"][c] = dict(local_dir=out, status="EXIT 0")
    return state


def cnn_wides(ctx, state):
    """predictions.csv of each finished config -> a wide CSV (VLM format; <key>__v = vertical yarns = our h)."""
    wides, summ = [], {}
    for c, info in state["configs"].items():
        p = os.path.join(info["local_dir"], "predictions.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["image_id"] = d.image.astype(int)
        cols = {}
        for (bb, ex), g in d.groupby(["backbone", "experiment"]):
            g = g.set_index("image_id")
            key = f"cnn-cv-{c}-{bb}-{ex}"
            cols[f"{key}__v"] = g.pred_h
            cols[f"{key}__h"] = g.pred_v
            cols[f"{key}__weave"] = g.pred_weave.map(lambda k: WEAVE_CLASSES[int(k)] if 0 <= int(k) < 4 else None)
        w = pd.DataFrame(cols)
        w.index.name = "image_id"
        out = os.path.join(ctx.work, f"cv_{c}_wide.csv")
        w.sort_index().to_csv(out)
        wides.append(out)
        sp = os.path.join(info["local_dir"], "summary.csv")
        if os.path.exists(sp):
            summ[c] = pd.read_csv(sp).to_dict("records")
    return wides, summ


def latest_state(ctx):
    p = os.path.join(ctx.state_root, "CNN_LATEST.txt")
    return open(p).read().strip() if os.path.exists(p) else None


def step_cnn(ctx):
    a = ctx.a
    if a.cnn == "off" and not a.cnn_collect:
        ctx.log("  skipped (no --cnn / --cnn-collect); the 'CNN, 5-fold FT' row stays \\TBD")
        ctx.N["cnn"]["status"] = "not run"
        return
    cfgs = [c.strip() for c in a.cnn_configs.split(",") if c.strip()]
    bad = [c for c in cfgs if c not in CNN_CONFIGS]
    if bad:
        raise StepError(f"unknown --cnn-configs {bad}; known: {list(CNN_CONFIGS)}")
    state = None
    if a.cnn_collect:
        sp = latest_state(ctx) if a.cnn_collect == "latest" else a.cnn_collect
        if not sp or not os.path.exists(sp):
            raise StepError(f"no CNN state to collect ({a.cnn_collect}); launch first with --cnn launch")
        state = json.load(open(sp))
        ctx.log(f"  collecting CNN runs of {state['stamp']} ({state['host']}): {list(state['configs'])}")
        now = {os.path.basename(p): md5(p) for p in ctx.scoring_csv}
        if now != state.get("label_md5"):
            ctx.warn("the CNN runs were trained on a different label export than this run scores "
                     "(re-launch with --cnn launch for the final labels)")
    elif a.cnn_host == "local":
        state = cnn_run_local(ctx, cfgs)
        sp = os.path.join(ctx.work, "cnn_state.json")
        with open(sp, "w") as f:
            json.dump(state, f, indent=1)
        with open(os.path.join(ctx.state_root, "CNN_LATEST.txt"), "w") as f:
            f.write(sp + "\n")
    else:
        state = cnn_launch_remote(ctx, cfgs, dry=a.cnn_dry)
        if state is None:
            ctx.N["cnn"]["status"] = "dry (commands printed)"
            return
        sp = os.path.join(ctx.work, "cnn_state.json")
        with open(sp, "w") as f:
            json.dump(state, f, indent=1)
        with open(os.path.join(ctx.state_root, "CNN_LATEST.txt"), "w") as f:
            f.write(sp + "\n")
        ctx.log(f"  launched; state {sp}")
        if a.cnn == "launch":
            ctx.N["cnn"]["status"] = f"launched on gpu-server ({state['stamp']}); collect with --cnn-collect latest"
            return
    if state["host"] == "gpu-server":
        t0 = time.time()
        while True:
            st = cnn_poll_remote(ctx, state)
            ctx.log("  " + time.strftime("%H:%M:%S") + "  " + "; ".join(f"{c}: {s[0]} {s[1][-80:]}" for c, s in st.items()))
            finished = all(s[0] in ("EXIT", "DIED") for s in st.values()) and len(st) == len(state["configs"])
            if finished or a.cnn_collect and not a.cnn_wait:
                break
            if time.time() - t0 > a.cnn_timeout_h * 3600:
                raise StepError(f"CNN runs not finished after {a.cnn_timeout_h} h; collect later with --cnn-collect")
            time.sleep(a.cnn_poll_s)
        failed = [c for c, s in st.items() if s[0] == "DIED" or (s[0] == "EXIT" and s[1].split()[0:1] != ["0"])]
        if failed:
            ctx.warn(f"CNN configs failed: {failed} (see their .log)")
        pending = [c for c, s in st.items() if s[0] in ("RUNNING", "QUEUED")]
        if pending:
            ctx.warn(f"CNN configs still running: {pending}; scoring the finished ones only")
        cnn_pull_remote(ctx, state)
        for c in state["configs"]:
            state["configs"][c]["status"] = " ".join(st.get(c, ("?", "")))
    wides, summ = cnn_wides(ctx, state)
    ctx.N["cnn"].update(status="scored", host=state["host"], stamp=state["stamp"],
                        configs={c: info.get("status") for c, info in state["configs"].items()}, summary=summ)
    if not wides:
        ctx.warn("no finished CNN run to score")
        return
    ctx.cv_wides = wides
    ctx.log(f"  CV predictions -> {len(wides)} wide file(s); re-scoring every method with them (step 4 again)")
    step_score(ctx)


# ============================================================================ step 6: tables + figures
def rows_by_method(stage_dir):
    lines = [l for l in open(os.path.join(stage_dir, "main150_rows.tex")).read().splitlines()
             if l.strip() and not l.startswith("%")]
    methods = pd.read_csv(os.path.join(stage_dir, "main150.csv")).method.tolist()
    if len(lines) != len(methods):
        raise StepError(f"main150_rows.tex has {len(lines)} rows for {len(methods)} methods")
    return dict(zip(methods, lines))


def iaa_withheld(ctx):
    """Pre-declared rule: fewer than 10 blind scenes counted by two annotators -> no IAA numbers ('agreement study ongoing')."""
    return ctx.N["coverage"].get("verdict_blind") == "NO-GO" or not ctx.N["agreement"].get("primary")


def tab1_rows(ctx, stage_dir, spec, n_table=None):
    S = load_score150()
    by = rows_by_method(stage_dir)
    out, missing = [], []
    for item in spec:
        if item == "midrule":
            out.append(r"\midrule")
            continue
        key, disp = item
        if key == "@human" and ctx.single:   # single protocol: one annotator, no human-vs-human row
            continue
        if key == "@human" and ctx.dual:   # dual protocol: every scene counted by both, no withholding rule
            k_h = (ctx.N["agreement"].get("primary") or {}).get("n", 0)
            line = S.tex_row("@human", ctx.human_metrics) if ctx.human_metrics else None
            esc = S.tex_escape("@human")
            if line is not None and n_table and k_h < n_table:   # covers only the scenes counted by both
                disp = f"{disp}$^{{{k_h}/{n_table}}}$"
        elif key == "@human":
            if "(blind)" not in disp:   # iaa protocol: the human row covers the blind IAA scenes only
                disp = disp + " (blind)"
            k_h = (ctx.N["agreement"].get("primary") or {}).get("n", 0)
            if iaa_withheld(ctx):   # pre-declared rule: no IAA numbers below 10 double-annotated blind scenes
                out.append(f"{disp} & \\multicolumn{{5}}{{c}}{{agreement study ongoing ($n={k_h}$)}} \\\\")
                continue
            line = S.tex_row("@human", ctx.human_metrics) if ctx.human_metrics else None
            esc = S.tex_escape("@human")
            # the human row covers only the double-annotated blind scenes; mark it like any partial row
            if line is not None and n_table and k_h < n_table:
                disp = f"{disp}$^{{{k_h}/{n_table}}}$"
        else:
            line, esc = by.get(key), S.tex_escape(key)
        if line is None:
            out.append(f"{disp} & \\TBD & \\TBD & \\TBD & \\TBD & \\TBD \\\\")
            missing.append(key)
            continue
        if not line.startswith(esc):
            raise StepError(f"cannot match row of {key!r} in main150_rows.tex: {line[:60]}")
        out.append(ci_format(disp + line[len(esc):], ctx.a.tab1_ci_decimals))
    return out, missing


def ci_format(row, d):
    """MAPE [lo, hi] with d decimals on the bounds (Tab. 2 uses integers); the MAPE itself keeps one decimal."""
    if d == 1:
        return row
    return re.sub(r"(\d+\.\d) \[(-?\d+\.\d), (-?\d+\.\d)\]",
                  lambda m: f"{m.group(1)} [{float(m.group(2)):.{d}f}, {float(m.group(3)):.{d}f}]", row, count=1)


SUPP_IAA_PENDING = "\n".join([
    "% AUTO-GENERATED by tools/label_day.py: no blind scene was counted by two annotators in this run (agreement pending).",
    r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
    r"Agreement on the blind scenes ($n=\TBD$ of 20) & $h$ (\TPI) & $v$ (\TPI) \\", r"\midrule",
    r"MAE between annotators & \TBD & \TBD \\", r"Bland--Altman bias [95\% LoA] & \TBD & \TBD \\",
    r"ICC(2,1), absolute agreement & \TBD & \TBD \\", r"Within 5\% / 10\% of each other & \TBD & \TBD \\", r"\midrule",
    r"Weave $\kappa$ / warp direction agreement & \multicolumn{2}{c}{\TBD{} / \TBD} \\",
    r"Anchoring $|$final$-$proposal$|$/final, blind / assisted & \multicolumn{2}{c}{\TBD{} / \TBD} \\",
    r"Median time per image, blind / assisted (min) & \multicolumn{2}{c}{\TBD{} / \TBD} \\",
    r"\bottomrule", r"\end{tabular}", ""])


def supp_labelcheck_tex(ctx):
    """Single-annotator protocol: the WHOLE tabular of supp Tab. S4 (supp.tex reads
    tables/supp_labelcheck_rows.tex; same rows as its fallback table): screening constants, the post-hoc proposal check
    per axis, ours MAPE on all scenes / without the 4 guide-example scenes, median minutes per photo."""
    G, Sc = ctx.N["agreement"], ctx.N["score"]
    pe = G.get("proposal_edits") or {}
    by = (pe.get("never_shown") or {}).get("by_line", {})   # without the guide scenes and non-proposal-free records

    def row(label, fmt):
        cells = [fmt(by[l]) if l in by else "--" for l in ("h", "v")]
        return f"{label} & {cells[0]} & {cells[1]} \\\\"
    o = Sc.get("methods_headline", Sc.get("methods", {})).get(OURS, {})
    og = next((r for r in Sc.get("subsets", []) if r["method"] == OURS and r["subset"] == GUIDE_SUBSET), {})
    t = (G.get("time_min") or {}).get("all")
    out = ["% AUTO-GENERATED by tools/label_day.py (single-annotator protocol, Oct 11) -- do not edit by hand.",
           f"% Proposal check: tables/proposal_edits150.csv (match within {pe.get('tol', ctx.a.edit_tol)} x the human dot "
           f"spacing), lines whose machine dots were never shown: without the {pe.get('n_guide_scenes', 0)} guide scenes "
           f"and {pe.get('n_not_free_scenes', 0)} record(s) not proven proposal-free; all lines in LABEL_DAY_REPORT.md.",
           f"% Ours MAPE: all n = {fnum(o.get('n_scored'), 0)} scenes, without the guide scenes "
           f"n = {fnum(og.get('n_scored'), 0)}. Median time over {(G.get('time_min') or {}).get('n', 0)} finished photos.",
           r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           r"One annotator, counting from scratch & $h$ (\TPI) & $v$ (\TPI) \\", r"\midrule"]
    for _, name, ans, ref in screening_rows(ctx.N["validation"].get("annotators") or []):
        out.append(f"Screening {name}: answer / 2019 hand count & {ans[0]} / {ref[0]} & {ans[1]} / {ref[1]} \\\\")
    out += [r"\midrule",
            row("Lines compared with our counter's dots", lambda b: f"{b['n_lines']}"),
            row(r"Machine dots matched by a human dot (median \%)", lambda b: f"{b['precision_med']:.0f}"),
            row("Dots to add / delete per line (median)", lambda b: f"{b['add_med']:.0f} / {b['delete_med']:.0f}"),
            row(r"Lines needing $\le$2 edits (\%)", lambda b: f"{b['le2_edit_pct']:.0f}"),
            r"\midrule",
            f"\\ours MAPE, all / without 27, 65, 93, 110 & \\multicolumn{{2}}{{c}}{{{fnum(o.get('MAPE'))} / "
            f"{fnum(og.get('MAPE'))}}} \\\\",
            f"Median time per photo (min) & \\multicolumn{{2}}{{c}}{{{fnum(t)}}} \\\\",
            r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def supp_labelcheck_dual_tex(ctx):
    """Dual protocol (two annotators, both from scratch): the WHOLE tabular of supp Tab. S4 (supp.tex reads
    tables/supp_labelcheck_rows.tex): screening constants of both annotators, the inter-annotator agreement over every
    scene counted by both (MAE, Bland-Altman bias and 95% limits, ICC(2,1), within 5% / 10%, median |diff| also on the
    badge / no-badge scenes and without the 4 guide scenes, weave kappa), the post-hoc proposal check (both annotators'
    lines pooled; per annotator in LABEL_DAY_REPORT.md), ours MAPE all / without the guide scenes, median minutes per
    photo per annotator. Annotator codes A1, A2 = ctx.ann_order (the SCREENING order)."""
    G, Sc = ctx.N["agreement"], ctx.N["score"]
    order = (ctx.ann_order + ["A2"])[:2] if ctx.ann_order else ["A1", "A2"]
    P = G.get("primary") or {}
    pe = G.get("proposal_edits") or {}
    by = (pe.get("never_shown") or {}).get("by_line", {})
    TBD = r"\TBD"

    def two(fmt, src):   # one cell per axis
        return " & ".join(fmt(src[l]) if src and l in src else TBD for l in ("h", "v"))
    o = Sc.get("methods_headline", Sc.get("methods", {})).get(OURS, {})
    og = next((r for r in Sc.get("subsets", []) if r["method"] == OURS and r["subset"] == GUIDE_SUBSET), {})
    tb = (G.get("time_min") or {}).get("by_annotator") or {}
    out = ["% AUTO-GENERATED by tools/label_day.py (dual protocol: two annotators counting from scratch, Oct 11) -- do not "
           "edit by hand.",
           f"% A1 = {order[0]}, A2 = {order[1]}. Agreement: each annotator's per-scene labels (headline label set: both lines "
           f"span >= 75%), n = {P.get('n', 0)} scenes counted by both; Bland-Altman = A1 - A2; within 5% / 10% and "
           "median |diff| relative to the mean of the two (median over both axes = PLAN 3 go/no-go: "
           f"{fnum(P.get('median_rel_both_axes'), 2)}%). Per-scene list: tables/human_agreement_per_scene.csv.",
           f"% Proposal check (both annotators' lines pooled): tables/proposal_edits150.csv (match within "
           f"{pe.get('tol', ctx.a.edit_tol)} x the human dot spacing), without the {pe.get('n_guide_scenes', 0)} guide "
           f"scenes and the records not proven proposal-free ({pe.get('n_not_free_scenes', 0)} scene(s)).",
           f"% Ours MAPE against the mean of the two: all n = {fnum(o.get('n_scored'), 0)} scenes, without the guide scenes "
           f"n = {fnum(og.get('n_scored'), 0)}.",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           r"Two annotators (A1, A2), from scratch & $h$ (\TPI) & $v$ (\TPI) \\", r"\midrule"]
    # rows as in the supp.tex fallback table: screening, agreement, proposal check, sensitivity + time
    scr = {n: SCREENING[n] for n in order if n in SCREENING}
    if scr:
        first = next(iter(scr.values()))
        for k, (crop, _, ref) in enumerate(first):
            ans = [scr[n][k][1] if n in scr else None for n in order]
            cells = [" / ".join("--" if a_ is None else str(a_[ax]) for a_ in ans) for ax in (0, 1)]
            out.append(f"Screening {crop}: 2019 hand count & {ref[0]} & {ref[1]} \\\\")
            out.append(f"\\quad answers of A1 / A2 & {cells[0]} & {cells[1]} \\\\")
        out.append(r"\midrule")
    sub = P.get("subsets") or {}
    badge_nm = "badge scenes ('check carefully' / 'take care')"

    def f_mae(x):
        return format(x["MAE"], ".2f")

    def f_ba(x):   # bias [95% limits of agreement], A1 - A2
        return f"{tex_signed(x['bias'], 2)} [{tex_signed(x['loa_lo'], 1)}, {tex_signed(x['loa_hi'], 1)}]"

    def f_icc(x):
        return format(x["icc"], ".3f")

    def f_medrel(x):
        return format(x["median_rel"], ".1f")
    bd, nb = sub.get(badge_nm) or {}, sub.get("scenes without a badge") or {}

    def f_badge(ax):   # median |diff| on the badge / no-badge scenes
        if not P:
            return TBD
        return " / ".join(format(b[ax]["median_rel"], ".1f") if b.get("n") else "--" for b in (bd, nb))
    n_both = P["n"] if P else TBD
    if P:
        kw = f"{fnum(P.get('weave_kappa'), 2)} / {fnum(P.get('warp_agree'), 0)}\\%"
    else:
        kw = TBD + "{} / " + TBD
    out += [f"Scenes counted by both & \\multicolumn{{2}}{{c}}{{{n_both}}} \\\\",
            f"MAE between annotators & {two(f_mae, P)} \\\\",
            f"Bland--Altman bias [95\\% LoA] & {two(f_ba, P)} \\\\",
            f"ICC(2,1), absolute agreement & {two(f_icc, P)} \\\\",
            f"Median $|$difference$|$ (\\%), all & {two(f_medrel, P)} \\\\",
            f"\\quad badge / no badge & {f_badge('h')} & {f_badge('v')} \\\\",
            f"Weave $\\kappa$ / warp direction agreement & \\multicolumn{{2}}{{c}}{{{kw}}} \\\\",
            r"\midrule",
            f"Lines compared with our counter's dots & {two(lambda b: str(b['n_lines']), by)} \\\\",
            f"Machine dots matched by a human dot (median \\%) & {two(lambda b: format(b['precision_med'], '.0f'), by)} \\\\",
            f"Dots to add / delete per line (median) & {two(lambda b: format(b['add_med'], '.0f') + ' / ' + format(b['delete_med'], '.0f'), by)} \\\\",
            f"Lines needing $\\le$2 edits (\\%) & {two(lambda b: format(b['le2_edit_pct'], '.0f'), by)} \\\\",
            r"\midrule",
            f"\\ours MAPE, all / without 27, 65, 93, 110 & \\multicolumn{{2}}{{c}}{{{fnum(o.get('MAPE'))} / "
            f"{fnum(og.get('MAPE'))}}} \\\\",
            f"Median time per photo, A1 / A2 (min) & \\multicolumn{{2}}{{c}}{{"
            + " / ".join(fnum((tb.get(n) or {}).get("median")) for n in order) + "} \\\\",
            r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(out)


def human_agreement_rows(ctx):
    """tables/human_agreement.csv in the dual protocol: one row per subset and axis (+ the weave kappa)."""
    P = ctx.N["agreement"].get("primary")
    if not P:
        return "note\ndual protocol: fewer than 3 scenes counted by both annotators in this run\n"
    rows = []
    for nm, b in [("all scenes counted by both", P)] + list((P.get("subsets") or {}).items()):
        for ax in ("h", "v"):
            if not b.get("n"):
                rows.append(dict(subset=nm, pair=f"{P['a']} vs {P['b']}", n_scenes=0, axis=ax))
                continue
            x = b[ax]
            rows.append(dict(subset=nm, pair=f"{P['a']} vs {P['b']}", n_scenes=b["n"], axis=ax, MAE=x["MAE"],
                             bias=x["bias"], loa_lo=x["loa_lo"], loa_hi=x["loa_hi"], icc_2_1=x["icc"], pearson=x["r"],
                             within5_pct=x["w5"], within10_pct=x["w10"], median_rel_pct=x["median_rel"],
                             median_rel_both_axes_pct=b["median_rel_both_axes"], weave_kappa=b["weave_kappa"],
                             weave_agree_pct=b["weave_agree"], weave_kappa_coarse=b["weave_kappa_coarse"],
                             warp_agree_pct=b.get("warp_agree")))
    return pd.DataFrame(rows).round(4).to_csv(index=False)


def supp_iaa_tex(ctx):
    """--protocol iaa only (the first two-annotator Tab. S4; supp.tex no longer reads tables/supp_iaa_rows.tex)."""
    if ctx.oct11:
        return None
    G = ctx.N["agreement"]
    P = G.get("primary")
    if "anchoring" not in G:
        return None
    an = G["anchoring"]
    t = G["time_min"]
    loa = lambda x: f"{tex_signed(x['loa_lo'])}, {tex_signed(x['loa_hi'])}"
    pct = lambda x: "--" if x is None else f"{x:.1f}\\%"
    mins = lambda x: "--" if x is None else f"{x:.1f}"
    if iaa_withheld(ctx):
        # pre-declared rule: < 10 double-annotated blind scenes -> no IAA numbers; anchoring and time are still reported
        k = (P or {}).get("n", 0)
        head = [f"Agreement on the blind scenes ($n={k}$ of {len(ctx.iaa)}) & $h$ (\\TPI) & $v$ (\\TPI) \\\\", r"\midrule",
                r"Inter-annotator agreement & \multicolumn{2}{c}{agreement study ongoing} \\", r"\midrule"]
    else:
        h, v = P["h"], P["v"]
        head = [
            f"Agreement on the blind scenes ($n={P['n']}$ of {len(ctx.iaa)}) & $h$ (\\TPI) & $v$ (\\TPI) \\\\", r"\midrule",
            f"MAE between annotators & {h['MAE']:.2f} & {v['MAE']:.2f} \\\\",
            f"Bland--Altman bias & {tex_signed(h['bias'], 2)} & {tex_signed(v['bias'], 2)} \\\\",
            f"\\quad 95\\% limits of agreement & {loa(h)} & {loa(v)} \\\\",
            f"ICC(2,1), absolute agreement & {h['icc']:.3f} & {v['icc']:.3f} \\\\",
            f"Within 5\\% / 10\\% of each other & {h['w5']:.0f} / {h['w10']:.0f} & {v['w5']:.0f} / {v['w10']:.0f} \\\\",
            r"\midrule",
            f"Weave $\\kappa$ / warp direction agreement & \\multicolumn{{2}}{{c}}{{{P['weave_kappa']:.2f} / {P['warp_agree']:.0f}\\%}} \\\\"]
    return "\n".join([
        "% AUTO-GENERATED by tools/label_day.py (annotation_tool/analyze.py definitions) -- do not edit by hand.",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        *head,
        f"Anchoring $|$final$-$proposal$|$/final, blind / assisted & \\multicolumn{{2}}{{c}}{{{pct(an['blind_clean_pct'])} / {pct(an['shown_pct'])}}} \\\\",
        f"Median time per image, blind / assisted (min) & \\multicolumn{{2}}{{c}}{{{mins(t['blind'])} / {mins(t['shown'])}}} \\\\",
        r"\bottomrule", r"\end{tabular}}", ""])


SUPP_S4_HOOK = "supp_labelcheck_rows"   # single/dual: supp.tex reads this file for Tab. S4


def supp_hooks(ctx):
    """Generated supplement fragments that supp.tex must \\input (Tab. S4 hook depends on the protocol)."""
    return (SUPP_S4_HOOK if ctx.oct11 else "supp_iaa_rows", "supp_flags_rows", "supp_main150_all")


def supp_flags_tex(ctx):
    rows = pd.DataFrame(ctx.N["score"]["strata"])
    o = rows[rows.method == OURS].set_index("stratum")

    def cell(names, col, d):
        vals = []
        for n in names:
            x = o.at[n, col] if n in o.index and col in o.columns else np.nan
            vals.append("--" if pd.isna(x) or o.at[n, "n"] == 0 else f"{x:.{d}f}")
        return " / ".join(vals)

    def nn(names):
        return "/".join(str(int(o.at[n, "n"])) if n in o.index else "0" for n in names)
    spec = [("All scenes", ["all"]), ("Flagged glare / blur / shadow", ["flag glare", "flag blur", "flag shadow"]),
            ("Flagged low resolution / perspective", ["flag low_res", "flag perspective"]),
            (r"$<$4 / 4--8 / $\ge$8\,px per yarn", ["<4 px/yarn", "4-8 px/yarn", ">=8 px/yarn"]),
            ("Full-resolution / quarter-only scenes", ["full-resolution", "quarter-only"])]
    out = ["% AUTO-GENERATED by tools/label_day.py (score150.metrics on the label-day scene table) -- do not edit by hand.",
           "% px per yarn = native px per inch of the representative (tables/strata150.csv S) / the denser human count.",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"\dataset, one row per scene & $n$ & \ours MAPE & both within 5\% \\", r"\midrule"]
    for lab, names in spec:
        out.append(f"{lab} & {nn(names)} & {cell(names, 'MAPE', 1)} & {cell(names, 'both5', 0)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(out)


def supp_all_tex(ctx, rows_all, n, n_files):
    lines = [l for l in rows_all.splitlines() if l.strip() and not l.startswith("%")]
    pf = ctx.N["score"].get("perfile", {})
    out = ["% AUTO-GENERATED by tools/label_day.py from tables/main150_rows_all.tex and tables/main150_perfile.csv"
           " (tools/score150.py) -- do not edit by hand.",
           r"\section{All methods on \dataset}", r"\label{sec:supp_all150}",
           (f"\\Cref{{tab:supp_all150}} lists every scored method and variant, one row per scene ($n={n}$); keys are those "
            "of the released prediction files. Per-file scores over the real files of the labelled scenes "
            f"({n_files} files) use scene-clustered bootstrap intervals (\\cref{{tab:supp_perfile150}})."),
           r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
           r"Method (key) & MAE & MAPE [95\% CI] & $\le$5\% & $\le$10\% & Weave \\", r"\midrule", *lines,
           r"\bottomrule", r"\end{tabular}}",
           (f"\\caption{{All methods on \\dataset against human counts ($n={n}$ scenes, one row per scene"
            + (r", all labels, assisted and blind; Tab.~1 uses the blind labels only" if ctx.headline == "blind" else "")
            + "). MAE in \\TPI; other columns in \\%. Superscripts give coverage when a method did not answer for every scene. "
            r"\texttt{@39}: without the 9 held-out scenes that re-shoot the fabric patch of a training scene.}"),
           r"\label{tab:supp_all150}", r"\end{table*}", "",
           r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           r"Method & files / scenes & MAPE [95\% CI] & $\le$5\% & $\le$10\% \\", r"\midrule"]
    S = load_score150()
    for m in KEY_METHODS:
        if m in pf:
            r = pf[m]
            if r.get("MAPE") is None or not math.isfinite(r["MAPE"]):
                continue
            # files / scenes the method was scored on (V3-V5 answer only on their held-out scenes)
            out.append(f"{S.tex_escape(m)} & {int(r['n_scored'])} / {r['n_clusters']} & {r['MAPE']:.1f} [{r['MAPE_lo']:.1f}, "
                       f"{r['MAPE_hi']:.1f}] & {r['w5']:.0f} & {r['w10']:.0f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{Secondary per-file scores (every real file of a labelled scene; scene-clustered bootstrap CI). "
            r"Files / scenes: those the method was scored on.}",
            r"\label{tab:supp_perfile150}", r"\end{table}", r"\clearpage   % flush both tables before the reference list", ""]
    return "\n".join(out)


def macros_tex(ctx):
    Sc, C = ctx.N["score"], ctx.N["coverage"]
    o = Sc["methods_headline"].get(OURS, {})
    g = Sc["methods_headline"].get(GPT, {})
    P = ctx.N["agreement"].get("primary") or {}
    if ctx.oct11:   # single/dual: minutes per photo and the proposal-free count; dual: + the human-human agreement
        pf = ctx.N["validation"].get("proposal_free", {})
        m = {"LDnScenes": Sc["n_headline"], "LDnAnnotators": len(Sc["annotators"]),
             "LDoursMAPE": fnum(o.get("MAPE")), "LDoursCI": f"[{fnum(o.get('MAPE_lo'))}, {fnum(o.get('MAPE_hi'))}]",
             "LDoursBothFive": fnum(o.get("both5"), 0), "LDoursBias": fnum(o.get("bias_med"), 1, True),
             "LDgptMAPE": fnum(g.get("MAPE")), "LDgptCI": f"[{fnum(g.get('MAPE_lo'))}, {fnum(g.get('MAPE_hi'))}]",
             "LDminPerPhoto": fnum((ctx.N["agreement"].get("time_min") or {}).get("all")),
             "LDpropFree": f"{pf.get('k', '--')} of {pf.get('n', '--')}",
             "LDscreenMaxDiff": fnum(C.get("screening_max_pct"))}
        if ctx.dual:   # the Sec. 3 sentence: n scenes counted by both, MAE h/v, ICC(2,1) h/v, weave kappa, median |diff|
            h, v = P.get("h") or {}, P.get("v") or {}
            m.update({"LDnBoth": P.get("n", 0), "LDhhMAEh": fnum(h.get("MAE"), 2), "LDhhMAEv": fnum(v.get("MAE"), 2),
                      "LDhhICCh": fnum(h.get("icc"), 3), "LDhhICCv": fnum(v.get("icc"), 3),
                      "LDhhKappa": fnum(P.get("weave_kappa"), 2), "LDhhMedRel": fnum(C.get("hh_median_rel"), 1)})
        lines = ["% AUTO-GENERATED by tools/label_day.py: label-day numbers as macros (optional; \\input in preamble to use)."]
        return "\n".join(lines + [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in m.items()]) + "\n"
    m = {"LDnScenes": Sc["n_headline"], "LDnAnnotators": len(Sc["annotators"]),
         "LDnBlind": P.get("n", 0), "LDoursMAPE": fnum(o.get("MAPE")),
         "LDoursCI": f"[{fnum(o.get('MAPE_lo'))}, {fnum(o.get('MAPE_hi'))}]", "LDoursBothFive": fnum(o.get("both5"), 0),
         "LDoursBias": fnum(o.get("bias_med"), 1, True), "LDgptMAPE": fnum(g.get("MAPE")),
         "LDgptCI": f"[{fnum(g.get('MAPE_lo'))}, {fnum(g.get('MAPE_hi'))}]",
         "LDhhMedRel": "--" if iaa_withheld(ctx) else fnum(C.get("hh_median_rel"), 1)}   # pre-declared withholding rule
    lines = ["% AUTO-GENERATED by tools/label_day.py: label-day numbers as macros (optional; \\input in preamble to use)."]
    lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    return "\n".join(lines) + "\n"


def step_tables(ctx):
    a = ctx.a
    st_all, st_blind, st_q = (os.path.join(ctx.stage, x) for x in ("score", "score_blind", "quality"))
    head_dir = st_blind if ctx.headline == "blind" else st_all
    spec = json.load(open(a.tab1_spec)) if a.tab1_spec else TAB1_SPEC
    n = ctx.N["score"]["n_scenes"] if ctx.headline == "all" else len(ctx.gt_blind)
    rows, missing = tab1_rows(ctx, head_dir, spec, n_table=n)
    hdr = (f"% AUTO-GENERATED by tools/label_day.py ({ctx.stamp}) from main150_rows_all.tex of tools/score150.py "
           f"({ctx.headline} labels, n = {n} scenes); row selection: PLAN 1b item 9 (label_day.py TAB1_SPEC). "
           "Columns: method & MAE & MAPE [95% CI] & <=5% & <=10% & weave acc\n")
    T = ctx.tables
    # \LDtabN: the caption's n (5_experiments.tex: \ifdefined\LDtabN); \gdef in the first cell is global
    ctx.put(os.path.join(T, "main150_rows.tex"), hdr + f"\\gdef\\LDtabN{{{n}}}%\n" + "\n".join(rows) + "\n")
    ctx.N["score"]["tab1"] = dict(rows=len([r for r in spec if r != "midrule"]), missing=missing, headline=ctx.headline,
                                  n=n, rows_tex=rows)
    if missing:
        ctx.warn(f"Tab. 1 rows without predictions yet (left \\TBD): {missing}")
    copies = [(st_all, f, f) for f in ("main150.csv", "main150_perfile.csv", "main150_fabric.csv", "strat150.csv",
                                       "coverage_ours.csv", "human_agreement.csv", "data150_summary.json")]
    copies += [(st_all, "main150_rows.tex", "main150_rows_all.tex")]
    if ctx.blind_ok:
        copies += [(st_blind, "main150.csv", "main150_blind.csv"), (st_blind, "main150_rows.tex", "main150_rows_blind_all.tex")]
    else:   # never leave an earlier run's blind-only outputs looking current (nothing is deleted: overwritten, backed up)
        for f, txt in (("main150_blind.csv", "method\n"), ("main150_rows_blind_all.tex",
                                                          "% no blind-only scoring in this label-day run\n")):
            if os.path.exists(os.path.join(T, f)):
                ctx.put(os.path.join(T, f), txt)
    copies += [(st_q, f, f) for f in ("conf150.csv", "degrade150_vs_gt.csv", "crops150_vs_gt.csv")]
    copies += [(ctx.stage, "main150_per_scene.csv", "main150_per_scene.csv")]
    for d, f, g in copies:
        p = os.path.join(d, f)
        if os.path.exists(p):
            ctx.put(os.path.join(T, g), src=p)
    # the analyses as CSVs (every number of the report traceable to a file)
    Sc = ctx.N["score"]
    pd.DataFrame([dict(comparison=k, **v) for k, v in Sc["paired"].items()]).round(3).to_csv(
        os.path.join(ctx.stage, "main150_paired.csv"), index=False)
    pd.DataFrame(Sc["subsets"]).round(3).to_csv(os.path.join(ctx.stage, "main150_subsets.csv"), index=False)
    pd.DataFrame(Sc["strata"]).round(3).to_csv(os.path.join(ctx.stage, "main150_strata.csv"), index=False)
    for f in ("main150_paired.csv", "main150_subsets.csv", "main150_strata.csv"):
        ctx.put(os.path.join(T, f), src=os.path.join(ctx.stage, f))
    if os.path.exists(os.path.join(ctx.stage, "proposal_edits150.csv")):   # single protocol: post-hoc proposal check
        ctx.put(os.path.join(T, "proposal_edits150.csv"), src=os.path.join(ctx.stage, "proposal_edits150.csv"))
    if ctx.dual:     # dual protocol: Tab. S4 with the agreement rows; agreement CSVs (score150's is replaced)
        ctx.put(os.path.join(T, SUPP_S4_HOOK + ".tex"), supp_labelcheck_dual_tex(ctx))
        ctx.put(os.path.join(T, "human_agreement.csv"), human_agreement_rows(ctx))
        p = os.path.join(ctx.stage, "human_agreement_per_scene.csv")
        if os.path.exists(p):
            ctx.put(os.path.join(T, "human_agreement_per_scene.csv"), src=p)
    elif ctx.single:   # single protocol: supp Tab. S4 = tables/supp_labelcheck_rows.tex (the whole tabular; supp.tex reads it)
        ctx.put(os.path.join(T, SUPP_S4_HOOK + ".tex"), supp_labelcheck_tex(ctx))
        # score150 writes a header-less human_agreement.csv with one annotator: replace it by an explicit note
        ctx.put(os.path.join(T, "human_agreement.csv"),
                "note\nsingle annotator (Oct 11 protocol): no inter-annotator agreement; see supp Tab. S4\n")
    else:
        iaa = supp_iaa_tex(ctx)
        if iaa:
            ctx.put(os.path.join(T, "supp_iaa_rows.tex"), iaa)
        elif os.path.exists(os.path.join(T, "supp_iaa_rows.tex")):   # an earlier run's numbers must not stay in the PDF
            ctx.put(os.path.join(T, "supp_iaa_rows.tex"), SUPP_IAA_PENDING)
    ctx.put(os.path.join(T, "supp_flags_rows.tex"), supp_flags_tex(ctx))
    rows_all = open(os.path.join(st_all, "main150_rows.tex")).read()
    n_files = ctx.N["score"]["perfile"].get(OURS, {}).get("n_files", "--")
    ctx.put(os.path.join(T, "supp_main150_all.tex"), supp_all_tex(ctx, rows_all, ctx.N["score"]["n_scenes"], n_files))
    ctx.put(os.path.join(T, "labelday_macros.tex"), macros_tex(ctx))
    fig_dir = os.path.join(st_blind if ctx.headline == "blind" else st_all, "fig")
    for f in ("fig_coverage.pdf", "fig_coverage.png"):
        p = os.path.join(fig_dir, f)
        if os.path.exists(p):
            ctx.put(os.path.join(ctx.figs, f), src=p)
    ctx.log(f"  Tab. 1: {len(rows)} lines ({ctx.headline} labels, n = {n}); wrote {len(ctx.written)} files so far")
    hooks = open(os.path.join(ctx.paper, "supp.tex")).read()
    for f in supp_hooks(ctx):
        if f"tables/{f}" not in hooks:
            ctx.warn(f"supp.tex does not \\input tables/{f}.tex yet (see the report)")


# ============================================================================ step 7: compile
PDF_SCRIPT = r"""
import json, re, sys
import pypdf
r = pypdf.PdfReader(sys.argv[1])
pages = []
for p in r.pages:
    pages.append(p.extract_text() or "")
refs = None
for i, t in enumerate(pages):
    m = re.search(r"(^|\n)References\s*\n", t)
    if m and re.search(r"\[1\]|\n[A-Z][a-z]+", t[m.end():m.end() + 400]):
        refs = (i + 1, len(re.sub(r"\s+", "", t[:m.start()])))
        break
print(json.dumps(dict(n_pages=len(pages), refs=refs, tbd=[t.count("TBD") for t in pages],
                      todo=[t.count("TODO") for t in pages])))
"""


def step_compile(ctx):
    a = ctx.a
    if a.skip_compile:
        ctx.log("  skipped (--skip-compile)")
        ctx.N["build"]["status"] = "skipped"
        return
    tect = a.tectonic
    if not os.path.exists(tect):
        raise StepError(f"Tectonic not found at {tect} (set --tectonic or LABEL_DAY_TECTONIC; or --skip-compile)")
    env = ctx.env()
    env["TECTONIC_CACHE_DIR"] = env["XDG_CACHE_HOME"] = a.tectonic_cache
    B = ctx.N["build"]
    for name in ("main", "supp"):
        outdir = os.path.join(ctx.work, "build", name)
        os.makedirs(outdir, exist_ok=True)
        rc, out = ctx.run([tect, "-X", "compile", f"{name}.tex", "--outdir", outdir, "--keep-logs",
                           "--keep-intermediates"], cwd=ctx.paper, env=env, check=False, quiet=True,
                          capture=os.path.join(outdir, "tectonic_stdout.txt"))
        logp = os.path.join(outdir, f"{name}.log")
        log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
        info = dict(rc=rc, overfull=len(re.findall(r"^Overfull", log, re.M)),
                    underfull=len(re.findall(r"^Underfull", log, re.M)),
                    undefined_refs=len(re.findall(r"Reference `[^']+' on page \d+ undefined", log)),
                    undefined_cites=len(re.findall(r"Citation `[^']+' on page \d+ undefined", log)),
                    errors=[l for l in log.splitlines() if l.startswith("!")][:5])
        pdf = os.path.join(outdir, f"{name}.pdf")
        if rc != 0 or not os.path.exists(pdf):
            B[name] = info
            raise StepError(f"{name}.tex did not compile (exit {rc}); log {logp}\n" + out[-1500:])
        pdf_py = a.pdf_python
        if pdf_py and os.path.exists(pdf_py):
            _, js = ctx.run([pdf_py, "-c", PDF_SCRIPT, pdf], check=False, quiet=True)
            try:
                pj = json.loads(js.strip().splitlines()[-1])
                info.update(n_pages=pj["n_pages"], tbd_rendered=sum(pj["tbd"]), todo_rendered=sum(pj["todo"]))
                if name == "main" and pj["refs"]:
                    pg, before = pj["refs"]
                    info.update(refs_page=pg, body_pages=pg if before > 300 else pg - 1)
            except (ValueError, IndexError, KeyError):
                ctx.warn(f"could not analyse {name}.pdf pages")
        else:
            info["n_pages"] = len(re.findall(rb"/Type\s*/Page[^s]", open(pdf, "rb").read()))
        B[name] = info
        ctx.put(os.path.join(ctx.paper, f"{name}.pdf"), src=pdf)
        ctx.log(f"  {name}.pdf: {info.get('n_pages')} pages" +
                (f", body {info.get('body_pages')} pages (references from p. {info.get('refs_page')})"
                 if name == "main" and info.get("refs_page") else "") +
                f"; overfull {info['overfull']}, underfull {info['underfull']}, undefined refs "
                f"{info['undefined_refs']}, cites {info['undefined_cites']}; rendered TBD {info.get('tbd_rendered', '?')}, "
                f"TODO {info.get('todo_rendered', '?')}")
        if name == "main" and info.get("body_pages", 0) > 8:
            ctx.warn(f"main body runs to {info['body_pages']} pages (limit 8): apply the PLAN section 4 cut list")
        if info["undefined_refs"] or info["undefined_cites"] or info["overfull"]:   # say it, not only count it
            ctx.warn(f"{name}.pdf: {info['undefined_refs']} undefined reference(s), {info['undefined_cites']} undefined "
                     f"citation(s), {info['overfull']} overfull box(es) (see {logp})")
    vt = os.path.join(ctx.paper, "tools", "validate_tex.py") if ctx.mode == "dry" else os.path.join(HERE, "validate_tex.py")
    rc, out = ctx.run([sys.executable, vt], check=False, quiet=True,
                      capture=os.path.join(ctx.work, "validate_tex_stdout.txt"))
    B["validate_tex"] = dict(rc=rc, tail=out.strip().splitlines()[-6:])
    ctx.log(f"  validate_tex.py: exit {rc}; " + " | ".join(out.strip().splitlines()[-2:]))


# ============================================================================ step 8: report
def inactive_ranges(text, paper_dir):
    """Line ranges of \\IfFileExists{F}{A}{B} branches that will not render (B when F exists, A otherwise)."""
    out = []
    for m in re.finditer(r"\\IfFileExists\{([^}]*)\}\{", text):
        f = m.group(1)
        i = m.end() - 1

        def match(i):
            depth = 0
            for k in range(i, len(text)):
                c = text[k]
                if c == "\\":
                    continue
                if c == "{" and (k == 0 or text[k - 1] != "\\"):
                    depth += 1
                elif c == "}" and (k == 0 or text[k - 1] != "\\"):
                    depth -= 1
                    if depth == 0:
                        return k
            return len(text) - 1
        a_end = match(i)
        j = text.find("{", a_end)
        b_end = match(j)
        exists = os.path.exists(os.path.join(paper_dir, f))
        s, e = (j, b_end) if exists else (i, a_end)
        out.append((text.count("\n", 0, s) + 1, text.count("\n", 0, e) + 1))
    return out


def marker_scan(paper_dir):
    files = [os.path.join(paper_dir, "main.tex"), os.path.join(paper_dir, "supp.tex")]
    files += sorted(glob.glob(os.path.join(paper_dir, "sections", "*.tex")))
    hits = []
    for p in files:
        text = open(p).read()
        dead = inactive_ranges(text, paper_dir)
        for n, line in enumerate(text.splitlines(), 1):
            code = re.split(r"(?<!\\)%", line, maxsplit=1)[0].replace("\\else\\TBD\\fi", "")
            k = code.count("\\TBD") + code.count("\\TODO")
            if not k or any(a <= n <= b for a, b in dead):
                continue
            hits.append((os.path.relpath(p, paper_dir), n, code.count("\\TBD"), code.count("\\TODO"), code.strip()[:160]))
    return hits


def grep_sentences(paper_dir, patterns, files=None):
    files = files or ([os.path.join(paper_dir, "sections", f) for f in ("0_abstract.tex", "1_intro.tex", "3_dataset.tex",
                                                                         "5_experiments.tex", "6_impact.tex",
                                                                         "7_conclusion.tex")])
    out = []
    for p in files:
        if not os.path.exists(p):
            continue
        for n, line in enumerate(open(p).read().splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for s in re.split(r"(?<=[.;])\s+(?=[A-Z\\])", line):
                if any(re.search(pt, s) for pt in patterns):
                    out.append((os.path.relpath(p, paper_dir), n, s.strip()[:220]))
    return out


def rewording(ctx):
    """Rule-based list of sentences whose claims the human-label numbers now decide."""
    Sc, C, G = ctx.N["score"], ctx.N["coverage"], ctx.N["agreement"]
    P = ctx.paper
    items = []
    o = Sc.get("methods_headline", Sc.get("methods", {})).get(OURS, {})
    n = Sc.get("n_headline", Sc.get("n_scenes"))
    PH = Sc.get("paired_headline", Sc.get("paired", {}))

    def add(rule, why, pats, files=None):
        hits = grep_sentences(P, pats, files)
        items.append(dict(rule=rule, why=why, hits=hits))
    t1 = Sc.get("tab1", {})
    items.append(dict(rule="Tab. 1", why=f"auto-filled: {t1.get('rows', '?')} rows, n = {t1.get('n', n)} scenes "
                      f"({t1.get('headline', 'all')} labels; the caption's n comes from \\LDtabN in main150_rows.tex)"
                      + (f"; still \\TBD: {t1['missing']}" if t1.get("missing") else ""), hits=[]))
    if o:
        add("abstract result", f"'On \\dataset the counter's error is \\TBD': ours {fnum(o.get('MAPE'))}% "
            f"[{fnum(o.get('MAPE_lo'))}, {fnum(o.get('MAPE_hi'))}] MAPE, both within 5% {fnum(o.get('both5'), 0)}%, "
            f"n = {n} scenes", [r"counter's error is \\TBD"])
    pg = PH.get(f"{OURS} vs {GPT}", {})
    if "delta" in pg:
        sep = pg["lo"] > 0 or pg["hi"] < 0
        who = "GPT-6.1-sol is better" if pg["lo"] > 0 else ("ours is better" if pg["hi"] < 0 else "not separable")
        why = (f"ours - GPT-6.1-sol on {pg['n']} scenes: {pg['delta']:+.1f} points [{pg['lo']:+.1f}, {pg['hi']:+.1f}], "
               f"p = {pg['p_wilcoxon']:.3g} -> {who}.")
        if sep and pg["lo"] > 0:
            why += " PLAN 3: reframe contribution 5 as 'a transparent offline baseline whose confidence predicts capture failure' (title 3)."
        add("ours vs GPT on PickGlass", why,
            [r"cannot be separated[^.]*(vision-language|VLM|GPT)", r"(VLM|GPT)[^.]*not separable", r"best frontier VLM",
             r"cannot be distinguished[^.]*(VLM|GPT)"])
    pc, pcg = PH.get(f"combined-v1 vs {OURS}", {}), PH.get("combined-v1 vs Log-Gabor bank", {})
    if "delta" in pc:   # the pre-registered combined counter (PREREG_combined_v1.json), held-out test passed
        add("combined-v1 (pre-registered; held-out synthetic test passed)",
            f"combined-v1 - ours on {pc['n']} scenes: {pc['delta']:+.1f} points [{pc['lo']:+.1f}, {pc['hi']:+.1f}], "
            f"p = {pc['p_wilcoxon']:.3g}" + (f"; - log-Gabor {pcg['delta']:+.1f} [{pcg['lo']:+.1f}, {pcg['hi']:+.1f}]"
                                            if "delta" in pcg else "")
            + ". Held-out synthetic P5 72.4% vs ours 70.3% and log-Gabor 59.5% (methods/results/"
            "combined_v1_eval_synth.json); legacy-13 numbers are post hoc; never 'matches' without a paired CI",
            [r"[Cc]ombined", r"combined-v1"])
    hh = C.get("hh_median_rel")
    if hh is not None and hh > THR["hh_go"]:
        add("+-5% framing", f"human-human median relative difference {hh:.1f}% > 5%: report methods relative to the "
            "human noise floor (PLAN 3)", [r"\$\\pm\$5\\%", r"5\\% tolerance", r"tolerances of product standards"])
    if G.get("anchoring", {}).get("rule_fires"):
        add("anchoring", f"anchoring gap {fnum(G['anchoring'].get('gap_points'))} points > 3: the headline uses blind "
            "labels (Tab. 1 was built from them)", [r"anchoring is \\TBD", r"headline uses blind labels"])
    cv = C.get("verdict_scenes")
    if cv in ("REDUCED", "NO-GO"):
        add(f"coverage {cv}", f"{C['scenes_labelled']} of 76 scenes labelled: " +
            ("say so in the abstract and stratify by resolution class" if cv == "REDUCED"
             else "Tab. 1 becomes 'preliminary, n = k' (title 2)"),
            [r"76 distinct scenes", r"76 scenes", r"\\TBD\{\} human counts"])
    bv = C.get("verdict_blind")
    if ctx.dual:
        P_ = G.get("primary") or {}
        add("two annotators from scratch (Oct 11 protocol)",
            f"two external annotators ({', '.join(ctx.ann_order)}) each counted every scene from scratch: "
            f"{P_.get('n', 0)} scenes counted by both, MAE {fnum((P_.get('h') or {}).get('MAE'), 2)} / "
            f"{fnum((P_.get('v') or {}).get('MAE'), 2)} TPI (h / v), ICC(2,1) {fnum((P_.get('h') or {}).get('icc'), 3)} / "
            f"{fnum((P_.get('v') or {}).get('icc'), 3)}, weave kappa {fnum(P_.get('weave_kappa'), 2)}, median |diff| "
            f"{fnum(C.get('hh_median_rel'), 2)}% ({C.get('verdict_hh')}); the reference is their mean. No blind-subset, "
            "anchoring or 'assisted' claims may remain; one-annotator wording must go",
            [r"blind subset", r"20 blind", r"agreement study ongoing", r"headline uses blind", r"blind-only",
             r"blind/assisted", r"[Oo]ne (external )?annotator", r"single annotator", r"the annotator's count",
             r"no inter-annotator", r"no second annotator", r"no measured human noise floor", r"\\TBD\{\} annotators"])
    if ctx.single:
        add("single annotator, no IAA (Oct 11 protocol)",
            "one external annotator counted every scene from scratch, without proposals: no blind-subset, IAA, "
            "anchoring or 'assisted' claims may remain; the post-hoc proposal check (supp Tab. S4) replaces them",
            [r"blind subset", r"20 blind", r"agreement study ongoing", r"ICC", r"[Kk]appa", r"headline uses blind",
             r"blind-only", r"annotators' mean", r"two annotators", r"\\TBD\{\} annotators", r"blind/assisted"])
    if ctx.oct11:
        pf = ctx.N["validation"].get("proposal_free", {})
        if pf and pf.get("k") != pf.get("n"):
            add("proposal-free claim", f"only {pf['k']} of {pf['n']} records are proven proposal-free: say 'k of them "
                "without machine proposals' and give the sensitivity subset",
                [r"without machine proposals", r"from scratch", r"no machine dot", r"never saw"],
                [os.path.join(P, "sections", f) for f in ("0_abstract.tex", "1_intro.tex", "3_dataset.tex")] +
                [os.path.join(P, "supp.tex")])
        if C.get("partial_records"):
            add("partly counted lines", f"{C['partial_records']} record(s) with a line spanning < 75% are not in the "
                f"headline ({C['partial_justified']} in the rescaled sensitivity rows): state the rule once in Sec. 3/supp S4",
                [r"partially occluded", r"75\\%"], [os.path.join(P, "sections", "3_dataset.tex"), os.path.join(P, "supp.tex")])
    elif bv != "GO":
        add(f"blind agreement {bv}", f"{C['blind_double']} blind scenes double-annotated: " +
            ("flag the wide CI" if bv == "REDUCED" else "no IAA numbers; say 'agreement study ongoing'"),
            [r"20 blind scenes", r"blind subset"])
    cg = PH.get(f"cascade-gate vs {GPT}", {})
    ca = Sc.get("cascade", {})
    if "delta" in cg:
        add("cascade", f"frozen gate on {cg['n']} scenes: cascade - GPT-only {cg['delta']:+.1f} points "
            f"[{cg['lo']:+.1f}, {cg['hi']:+.1f}]; escalates {ca.get('escalated')}/{ca.get('n')} labelled scenes "
            f"(US$ {fnum(ca.get('usd_per_1000_cascade'), 2)} vs {fnum(ca.get('usd_per_1000_gpt_only'), 2)} per 1,000)"
            + ("; the cascade is WORSE than GPT-only here" if cg["lo"] > 0 else ""),
            [r"matches its error", r"GPT-only's error", r"escalates 15 of 76", r"quality gate"])
    rg = PH.get(f"pg-lattice@ruler vs {OURS}", {})
    if "delta" in rg:
        add("ruler", f"ruler crop - annotated corners: {rg['delta']:+.1f} points [{rg['lo']:+.1f}, {rg['hi']:+.1f}] "
            f"against human counts" + (" -> 'can replace corner annotation' needs softening" if rg["lo"] > 0 else ""),
            [r"can replace corner annotation", r"ruler \(\\cref\{sec:ruler\}\) calibrates", r"finds the window in 75 of 76"])
    wv = o.get("weave_acc")
    add("weave", f"ours' weave accuracy on PickGlass: {fnum(wv, 0)}% (n = {fnum(o.get('weave_n'), 0)})",
        [r"on \\dataset it gets \\TBD"])
    u = Sc.get("unknown13")
    if u:
        add("pseudo-label CNN, 13 'unknown' scenes",
            f"on the {u['n']} labelled 'unknown'-weave scenes the pseudo-label CNN was scored on: ours {u['ours_mape']:.1f}% MAPE (median signed "
            f"{u['ours_median_signed']:+.1f}%), CNN + pseudo-labels {u['cnn_mape']:.1f}% (signed {u['cnn_median_signed']:+.1f}%), "
            f"CNN within {u['cnn_vs_ours_median_abs_diff']:.1f}% of ours (median): decide 'corrected or inherited'",
            [r"awaits the human counts", r"corrected or inherited"])
    p46 = Sc.get("paired", {}).get(f"{PSEUDO_KEY}@39 vs {OURS}", {})
    if "delta" in p46:
        add("39-scene pseudo-label check (S19 promise)", f"CNN + pseudo-labels without the 9 re-shoot scenes on {p46['n']} scenes: "
            f"{p46['mape_a']:.1f}% MAPE, - ours {p46['delta']:+.1f} [{p46['lo']:+.1f}, {p46['hi']:+.1f}]",
            [r"38 and 61", r"46 scenes", r"two of these", r"39 scenes", r"nine of these"], [os.path.join(P, "sections", "supp_cnnaug.tex")])
    cf = [r for r in Sc.get("confidence", []) if str(r.get("estimator", "")).startswith("ours")]
    if cf:
        r = cf[0]
        add("confidence TODO (5.3)", f"against human counts per scene: Spearman(conf, error) "
            f"{fnum(r.get('spearman_conf_vs_err'), 2)}, AUC(both within 10%) {fnum(r.get('auc_conf_both_within10'), 2)} "
            f"(n = {r.get('n_images')})", [r"From \\texttt\{quality150.py\}"])
    d = Sc.get("distortion")
    if d:
        add("distortion basis (REVIEW G-P0-4)", f"ours' median signed error {fnum(d.get('ours_bias_med_observed'), 1, True)}% "
            f"vs about {d['ours_expected_offset_median']:+.1f}% expected from radial distortion alone (k1 {d['k1']}, "
            f"lattice {d['mult']}x area-mean excess; humans' lines read {d['human_line_excess_median']:+.1f}%): say so "
            "before calling it counting bias", [r"median signed error", r"counting bias"])
    add("conclusion TODO", "state the headline result on PickGlass and whether the legacy ranking holds "
        f"(top 5 here: {', '.join(Sc.get('blind_rank', {}).get('top5_all', [])[:5]) or 'see Tab. 1'})",
        [r"Revise once \\cref\{tab:main150\}"])
    return items


def numbers_diff(ctx):
    """Changes against the previous run in the same place: the headline numbers first, then a count."""
    prev = sorted(p for p in glob.glob(ctx.history_glob) if os.path.realpath(p) != os.path.realpath(
        os.path.join(ctx.work, "numbers.json")))
    if not prev:
        return None, [], 0
    old = flatten(json.load(open(prev[-1])))
    new = flatten(jsonable(ctx.N))
    tab1 = [k for k in (x[0] for x in (json.load(open(ctx.a.tab1_spec)) if ctx.a.tab1_spec else TAB1_SPEC)
                        if x != "midrule") if k != "@human"]
    key_re = re.compile(
        r"^(coverage\.(scenes_labelled|blind_double|hh_median_rel|verdict_\w+)"
        r"|agreement\.primary\.(n|median_rel_both_axes|[hv]\.(MAE|icc|bias))|agreement\.anchoring\.gap_points"
        r"|score\.n_scenes|score\.headline"
        r"|score\.methods\.(" + "|".join(re.escape(k) for k in tab1) + r")\.(n_scored|MAPE|MAPE_lo|MAPE_hi|w5|w10|both5|bias_med|weave_acc)"
        r"|score\.paired\.[^.]+\.(n|delta|lo|hi|p_wilcoxon)|score\.cascade\.(n|escalated))$")
    ch, other = [], 0
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        same = (a == b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-6)
        if same:
            continue
        if key_re.match(k):
            r = lambda x: (round(x, 3) if abs(x) < 10 else round(x, 2)) if isinstance(x, float) else x
            ch.append((k, r(a), r(b)))
        else:
            other += 1
    return prev[-1], ch, other


def dual_report_lines(ctx):
    """Report section 3 in the dual protocol: agreement (all scenes counted by both + subsets), the Tab. 1 human row in
    both directions, the most disagreeing scenes, each annotator alone as the reference, the post-hoc proposal check
    pooled and per annotator, minutes per photo per annotator."""
    G, C, Sc = ctx.N["agreement"], ctx.N["coverage"], ctx.N["score"]
    P = G.get("primary")
    L = []
    if not P:
        L.append(f"Dual protocol: fewer than 3 scenes counted by two annotators (pairs: {G.get('pairs')}); no agreement "
                 "numbers in this run (Tab. 1 human row and the Tab. S4 agreement rows stay \\TBD).")
    else:
        L.append(f"Dual protocol (Oct 11): **{P['a']} (annotator 1) vs {P['b']} (annotator 2) on {P['n']} scenes counted by "
                 f"both** (each annotator's own per-scene labels, headline label set; Bland-Altman = annotator 1 - "
                 f"annotator 2; relative differences against the pair's mean). Scenes counted by one only: "
                 f"{P.get('scenes_only_one')}. Every pair: {G.get('pairs')}.")
        L.append("")
        blocks = [("all scenes counted by both", P)] + list((P.get("subsets") or {}).items())
        hdr = ["subset", "n", "MAE h / v (TPI)", "BA bias h [95% LoA]", "BA bias v [95% LoA]", "ICC(2,1) h / v",
               "within 5% h / v", "within 10% h / v", "median |diff| h / v %", "median |diff| both axes %",
               "both axes within 5% (% scenes)", "weave kappa (agree %)", "weave kappa coarse", "warp agree %"]
        rows = []
        for nm, b in blocks:
            if not b.get("n"):
                rows.append([nm, 0] + ["--"] * (len(hdr) - 2))
                continue
            h, v = b["h"], b["v"]
            rows.append([nm, b["n"], f"{h['MAE']:.2f} / {v['MAE']:.2f}",
                         f"{h['bias']:+.2f} [{h['loa_lo']:+.2f}, {h['loa_hi']:+.2f}]",
                         f"{v['bias']:+.2f} [{v['loa_lo']:+.2f}, {v['loa_hi']:+.2f}]",
                         f"{fnum(h['icc'], 3)} / {fnum(v['icc'], 3)}", f"{h['w5']:.0f} / {v['w5']:.0f}",
                         f"{h['w10']:.0f} / {v['w10']:.0f}", f"{h['median_rel']:.2f} / {v['median_rel']:.2f}",
                         f"{b['median_rel_both_axes']:.2f}", f"{b['both_axes_within5_pct']:.0f}",
                         f"{fnum(b['weave_kappa'], 2)} ({b['weave_agree']:.0f})", fnum(b["weave_kappa_coarse"], 2),
                         fnum(b.get("warp_agree"), 0)])
        L.append(md_table(hdr, rows))
        L.append("")
        L.append(f"Go/no-go (PLAN 3): human-human median |diff| {fnum(C.get('hh_median_rel'), 2)}% -> "
                 f"**{C.get('verdict_hh')}** (GO <= {THR['hh_go']:.0f}%, REDUCED <= {THR['hh_reduced']:.0f}%).")
        hm, hr = G.get("human_row") or {}, G.get("human_row_reverse") or {}
        if hm.get("n_scored"):
            L.append(f"Tab. 1 'Human vs human' row = {hm.get('scored')}'s counts scored against {hm.get('reference')}'s "
                     f"(n = {hm['n_scored']}): MAE {hm['MAE']:.2f} TPI, MAPE {hm['MAPE']:.2f}% [{hm['MAPE_lo']:.2f}, "
                     f"{hm['MAPE_hi']:.2f}], within 5% {hm['w5']:.0f}%, within 10% {hm['w10']:.0f}%, both within 5% "
                     f"{fnum(hm.get('both5'), 0)}%, weave {fnum(hm.get('weave_acc'), 0)}%. Reverse direction "
                     f"({hr.get('scored')} vs {hr.get('reference')}): MAPE {fnum(hr.get('MAPE'), 2)}%, within 5% "
                     f"{fnum(hr.get('w5'), 0)}%.")
        if P.get("worst"):
            L.append("Most disagreeing scenes (author spot-check; all in `tables/human_agreement_per_scene.csv`): "
                     + ", ".join(f"#{w['scene']} {w['rel_mean_pct']:.1f}% ({w['a']} vs {w['b']})" for w in P["worst"]) + ".")
    ra = Sc.get("reference_alone") or {}
    if ra:
        L += ["", "Each annotator alone as the reference (sensitivity; Tab. 1 uses the mean of the two; all key methods in "
              "the 'reference: ... alone' rows of `main150_subsets.csv`):", ""]
        rows = []
        for n_, d in ra.items():
            o = next((r for r in Sc.get("subsets", []) if r["subset"] == d["label"] and r["method"] == OURS), {})
            g = next((r for r in Sc.get("subsets", []) if r["subset"] == d["label"] and r["method"] == GPT), {})
            pg = d["paired"].get(f"{OURS} vs {GPT}", {})
            rows.append([n_, d["n_scenes"], f"{fnum(o.get('MAPE'))} [{fnum(o.get('MAPE_lo'))}, {fnum(o.get('MAPE_hi'))}]",
                         f"{fnum(g.get('MAPE'))} [{fnum(g.get('MAPE_lo'))}, {fnum(g.get('MAPE_hi'))}]",
                         f"{pg['delta']:+.1f} [{pg['lo']:+.1f}, {pg['hi']:+.1f}]" if "delta" in pg else "--"])
        L.append(md_table(["reference", "scenes", "ours MAPE [95% CI]", "GPT-6.1-sol MAPE [95% CI]", "ours - GPT (points)"],
                          rows))
    pe = G.get("proposal_edits")
    if pe:
        L += ["", f"Post-hoc proposal check (supp Tab. S4 middle rows = 'never shown', both annotators pooled; "
              f"`tables/proposal_edits150.csv`; one-to-one 1-D matching within {pe['tol']} x the human dot spacing, "
              f"'off-centre' = matched but more than {pe['move_tol']} x the spacing apart):", ""]
        sets = [("all lines, pooled", pe.get("all")), ("never shown, pooled (Tab. S4)", pe.get("never_shown"))]
        ns = pe.get("never_shown") or {}
        sets += [(f"  {l} lines (Tab. S4 columns)", ns["by_line"][l]) for l in ("h", "v") if l in ns.get("by_line", {})]
        sets += [("clean (never shown, no badge, line not moved or turned)", pe.get("clean"))]
        for n_, d in (pe.get("by_annotator") or {}).items():
            sets += [(f"{n_}: all lines", d.get("all")), (f"{n_}: never shown", d.get("never_shown"))]
        L.append(md_table(["lines", "n lines (scenes)", "add / delete / off-centre (median)", "machine dots matched %",
                           "human dots found %", "<= 2 edits %", "0 edits %"],
                          [[nm, f"{x['n_lines']} ({x['n_scenes']})",
                            f"{x['add_med']:.0f} / {x['delete_med']:.0f} / {x['move_med']:.0f}",
                            f"{x['precision_med']:.0f}", f"{x['recall_med']:.0f}", f"{x['le2_edit_pct']:.0f}",
                            f"{x['zero_edit_pct']:.0f}"] for nm, x in sets if x]))
        L.append("")
        tb = (G.get("time_min") or {}).get("by_annotator") or {}
        L.append(f"Badge scenes among them: {pe.get('n_hint_scenes')}; guide lines moved or turned: "
                 f"{pe.get('n_moved_or_rotated_lines')}. Median minutes per photo: "
                 + ", ".join(f"{k} {v['median']:.1f} (n = {v['n']})" for k, v in tb.items())
                 + f"; pooled {fnum((G.get('time_min') or {}).get('all'))} (from scratch; no time-saving claim).")
    return L


def step_report(ctx):
    N = ctx.N
    with open(os.path.join(ctx.work, "numbers.json"), "w") as f:
        json.dump(jsonable(N), f, indent=1)
    C, G, Sc, B = N["coverage"], N["agreement"], N["score"], N["build"]
    # FAKE banner: fake-looking annotator names OR input file names (a fake export under a real annotator code has a real-looking name)
    fake = (any(looks_fake(n) for n in N["validation"].get("annotators", [])) or
            any(re.search(r"fake|dummy|smoke", os.path.basename(f), re.I) for f in ctx.inputs))
    L = []
    L.append(f"# Label-day report ({ctx.stamp})")
    L.append("")
    if fake:
        L.append("> **FAKE LABELS: this is a pipeline test. None of these numbers may enter the paper.**")
        L.append("")
    L.append(f"Mode **{ctx.mode.upper()}**" + (f" (everything written under `{ctx.run_dir}`)" if ctx.mode == "dry" else
                                              f" (logs and backups of overwritten files: `{ctx.work}`)") +
             f". Generated by `paper/tools/label_day.py`; full log `{os.path.join(ctx.work, 'label_day.log')}`, all numbers "
             f"`{os.path.join(ctx.work, 'numbers.json')}`.")
    if ctx.failed:
        L += ["", f"**STOPPED at step {ctx.failed[0]} ({ctx.failed[1]}):**", "", "```", ctx.failed[2][:3000], "```"]
    L += ["", "## 1. Go / no-go (PLAN.md section 3)", ""]
    rows = [["Scenes with a finished human count", f"{C.get('scenes_labelled', '?')} / 76 (full-res "
             f"{C.get('full_labelled', '?')}/{C.get('n_full', '?')}, quarter-only {C.get('quarter_labelled', '?')}/"
             f"{C.get('n_quarter', '?')})", "GO >= 51; REDUCED 25-50", C.get("verdict_scenes", "?")],
            ["Blind scenes counted by 2 annotators", f"{C.get('blind_double', '?')} / {C.get('iaa_scenes', 20)}",
             "GO >= 15; REDUCED 10-14", C.get("verdict_blind", "?")],
            ["Human-human median relative difference (blind, both axes)",
             f"{fnum(C.get('hh_median_rel'), 2)}% (n = {(G.get('primary') or {}).get('n')} scenes)"
             if C.get("hh_median_rel") is not None else "n/a",
             "GO <= 5%; REDUCED 5-10%", C.get("verdict_hh", "?")]]
    an = G.get("anchoring", {})
    single = ctx.single
    if single:   # single protocol: one external annotator, no IAA by design; nothing for the anchoring rule to compare
        pe = (G.get("proposal_edits") or {}).get("all") or {}
        pf = N["validation"].get("proposal_free", {})
        rows[0][0] = "Scenes with a finished FL_SA count (headline: both lines span >= 75%)"
        rows[1:] = [["FL_SA vs 2019 hand counts (screening r2 + a5, constants)",
                     f"max {fnum(C.get('screening_max_pct'))}% ("
                     + "; ".join(f"{nm} {a_[0]}x{a_[1]} vs {r_[0]}x{r_[1]}"
                                 for _, nm, a_, r_ in screening_rows(N["validation"].get("annotators") or [])) + ")",
                     "GO <= 5%; REDUCED 5-10%", C.get("verdict_screening", "?")],
                    ["Second annotator / inter-annotator agreement", "single-annotator protocol (Oct 11)",
                     "not part of the protocol", "n/a (by design)"],
                    ["Proposal-free records (n_prop, dot origins, proposal_sig, created after the hide time)",
                     f"{pf.get('k', '?')} of {pf.get('n', '?')}", "all (else: sensitivity subset + reword)",
                     "ok" if pf.get("k") == pf.get("n") else "CHECK"],
                    ["Post-hoc proposal check (proposals never shown)",
                     f"median {fnum(pe.get('edits_med'), 0)} edits per line; {fnum(pe.get('le2_edit_pct'), 0)}% of "
                     f"{pe.get('n_lines', 0)} lines need <= 2", "descriptive (no rule)", "--"]]
    elif ctx.dual:   # dual protocol, two annotators from scratch: coverage + screening + human-human, no blind/anchoring rows
        pe = (G.get("proposal_edits") or {}).get("all") or {}
        pf = N["validation"].get("proposal_free", {})
        P_ = G.get("primary") or {}
        rows[0][0] = "Scenes with a finished human count, any annotator (headline: both lines span >= 75%)"
        rows[0][1] += f"; per annotator {C.get('scenes_by_annotator')}; counted by both {C.get('scenes_both', '?')}"
        rows[1:] = [["Screening vs 2019 hand counts (r2 + a5, constants), max over the annotators",
                     f"max {fnum(C.get('screening_max_pct'))}% ("
                     + "; ".join(f"{an_} {nm} {a_[0]}x{a_[1]} vs {r_[0]}x{r_[1]}"
                                 for an_, nm, a_, r_ in screening_rows(N["validation"].get("annotators") or [])) + ")",
                     "GO <= 5%; REDUCED 5-10%", C.get("verdict_screening", "?")],
                    ["Human-human median |diff| (every scene counted by both, both axes, relative to the pair's mean)",
                     (f"{fnum(C.get('hh_median_rel'), 2)}% ({P_.get('a')} vs {P_.get('b')}, n = {P_.get('n')} scenes)"
                      if C.get("hh_median_rel") is not None else "n/a (fewer than 3 scenes counted by both)"),
                     f"GO <= {THR['hh_go']:.0f}%; REDUCED {THR['hh_go']:.0f}-{THR['hh_reduced']:.0f}%",
                     C.get("verdict_hh", "?")],
                    ["Proposal-free records (n_prop, dot origins, proposal_sig, created after the hide time)",
                     f"{pf.get('k', '?')} of {pf.get('n', '?')}", "all (else: sensitivity subset + reword)",
                     "ok" if pf.get("k") == pf.get("n") else "CHECK"],
                    ["Post-hoc proposal check (proposals never shown; both annotators pooled)",
                     f"median {fnum(pe.get('edits_med'), 0)} edits per line; {fnum(pe.get('le2_edit_pct'), 0)}% of "
                     f"{pe.get('n_lines', 0)} lines need <= 2", "descriptive (no rule)", "--"]]
    else:
        rows.append(["Anchoring gap (blind-clean - shown, median |final-proposal|/final)",
                     f"{fnum(an.get('blind_clean_pct'))}% - {fnum(an.get('shown_pct'))}% = {fnum(an.get('gap_points'))} points",
                     "> 3 points -> headline uses --blind-only",
                     ("FIRES" if an.get("rule_fires") else "does not fire") if an else "?"])
    L.append(md_table(["Check", "Value", "Threshold", "Verdict"], rows))
    vs = ([C.get("verdict_scenes"), C.get("verdict_screening")] if single else
          [C.get("verdict_scenes"), C.get("verdict_screening"), C.get("verdict_hh")] if ctx.dual else
          [C.get("verdict_scenes"), C.get("verdict_blind"), C.get("verdict_hh")])
    overall = "NO-GO" if "NO-GO" in vs[:1] else ("REDUCED" if any(v in ("REDUCED", "NO-GO", "n/a") for v in vs) else "GO")
    L += ["", f"**Overall: {overall}.** Tab. 1 uses **{ctx.headline}** labels." +
          (" NO-GO on scenes: the paper stands on task + dataset/tool + ruler + capture study + legacy; Tab. 1 is "
           "'preliminary, n = k' (title 2)." if vs[0] == "NO-GO" else "") +
          (" Blind NO-GO: no IAA numbers ('agreement study ongoing')." if not ctx.oct11 and vs[1] == "NO-GO" else "")
          + (" Single-annotator protocol (Oct 11): go/no-go on scene coverage (and the screening constants) only; "
             "no IAA, no anchoring rule, no blind-only headline." if single else "")
          + (" Dual protocol (Oct 11, two annotators from scratch): go/no-go on scene coverage, the screening constants "
             "and the human-human median |diff| over every scene counted by both; Tab. 1 reference = the mean of the two; "
             "no blind subset, no anchoring rule, no blind-only headline."
             + (f" Human-human {C.get('verdict_hh')}: report methods relative to the human noise floor "
                f"({fnum(C.get('hh_median_rel'), 1)}%), not a +-5% tolerance (PLAN 3)."
                if C.get("verdict_hh") in ("REDUCED", "NO-GO") else "") if ctx.dual else "")]
    if Sc.get("blind_rank"):
        br = Sc["blind_rank"]
        L.append(f"Blind-only check: on the {br['n_scenes']} scenes with blind labels, the method ranking with all "
                 f"labels vs blind labels has Spearman {br['spearman']:.2f} over {br['n_methods']} methods; top-1 "
                 f"{'unchanged' if br['top1_same'] else 'CHANGES'} ({br['top5_all'][0]} vs {br['top5_blind'][0]}).")
    # validation
    V = N["validation"]
    L += ["", "## 2. Exports and validation", ""]
    if V.get("per_annotator"):
        L.append(md_table(["Annotator", "records", "done", "reps", "copies", "open", "blind-clean", "shown", "mixed",
                           "proposal-free", "cleared", "nonwoven", "CSV-only", "conflicts", "median min"],
                          [[p["annotator"], p["records"], p["done"], p["reps_done"], p["copies_done"], p["not_done"],
                            p["blind_clean"], p["shown"], p["mixed"], p.get("proposal_free"), p["cleared_lines"], p["nonwoven"], p["csv_only"],
                            p["conflicts"], p["median_min"]] for p in V["per_annotator"]]))
    if ctx.oct11:
        L.append(f"\n({'Single' if single else 'Dual'} protocol: 'shown' is core.js's proposals_seen flag, which the kit sets on every non-blind record "
                 "even though its manifest has no proposal; whether machine dots were really on screen is the "
                 "'proposal-free' column.)")
    L += ["", f"Inputs ({len(ctx.inputs)} files): " + ", ".join(f"`{os.path.basename(f)}`" for f in ctx.inputs) + ".",
          f"Unlabelled scenes: {C.get('unlabelled_scenes')}." if C.get("unlabelled_scenes") else "Every scene is labelled."]
    if V.get("partial_lines"):
        L += ["", f"Partly counted lines (span < {SPAN_MIN:.0%}; NOT in the headline; justified ones enter the "
              "'sensitivity: partial lines rescaled' rows of `main150_subsets.csv`):", "",
              md_table(["annotator", "id", "span h %", "span v %", "dots h / v", "occluded", "notes", "TPI h / v",
                        "rescaled h / v", "sensitivity"],
                       [[d["annotator"], d["id"], d["span_h"], d["span_v"], f"{d['n_h']} / {d['n_v']}", d["occluded"],
                         d["notes"][:40], f"{d['h_tpi']} / {d['v_tpi']}",
                         f"{fnum(d['h_tpi_rescaled'], 2)} / {fnum(d['v_tpi_rescaled'], 2)}",
                         "kept" if d["in_sensitivity"] else "dropped (no reason given)"] for d in V["partial_lines"]])]
        ps = Sc.get("partial_sensitivity")
        if ps:
            L.append(f"Sensitivity label set: {ps['n_scenes']} scenes (added {ps['scenes_added'] or 'none'}, label "
                     f"changed {ps['scenes_changed'] or 'none'}).")
    if V.get("short_span"):
        inc = [d for d in V["short_span"] if d["likely_incomplete"]]
        L += ["", f"Lines spanning {SPAN_MIN:.0%}-{SPAN_WARN:.0%} of the crop (kept in the headline): {len(V['short_span'])} "
              f"records; {len(inc)} stay below {SPAN_WARN:.0%} with one dot spacing added (look at these): "
              + (", ".join(f"#{d['id']} (h {d['span_h']}%, v {d['span_v']}%)" for d in inc) if inc else "none") + "."]
    pf = V.get("proposal_free")
    if pf:
        L += ["", f"**Proposal-free records: {pf['k']} of {pf['n']}** (finished records; judged from n_prop, the dot "
              f"origins, proposal_method / proposal_sig and, in the freelancer kit, creation before {pf['hidden_at']}; "
              "core.js sets proposals_seen = true even when it inserts no dot)."
              + ("" if not pf["not_free"] else " Not proven free: " + ", ".join(
                  f"#{d['id']} ({'proposal dots in the record' if d['fields'] else 'created ' + d['created'] + ' before the hide time'})"
                  for d in pf["not_free"]) + " -> see the 'proven proposal-free records only' subset; reword the "
              "'without machine proposals' claims (abstract, Sec. 3, supp S4) if this is not 0.")]
    if getattr(ctx, "iaa_table", None):
        L += ["", "IAA subset (20 representatives):", "",
              md_table(["Scene", "Resolution", "2 blind"] + V["annotators"], ctx.iaa_table)]
    if ctx.warnings:
        L += ["", "Warnings:", ""] + [f"- {w}" for w in ctx.warnings]
    # agreement
    P = G.get("primary")
    L += ["", "## 3. Agreement and anchoring (supp Tab. S4, section 3 sentence)", ""]
    if ctx.dual:
        L += dual_report_lines(ctx)
    elif P:
        L.append(md_table(["", "h", "v"], [
            ["MAE (TPI)", f"{P['h']['MAE']:.2f}", f"{P['v']['MAE']:.2f}"],
            ["Bland-Altman bias [95% LoA]", f"{P['h']['bias']:+.2f} [{P['h']['loa_lo']:+.2f}, {P['h']['loa_hi']:+.2f}]",
             f"{P['v']['bias']:+.2f} [{P['v']['loa_lo']:+.2f}, {P['v']['loa_hi']:+.2f}]"],
            ["ICC(2,1)", f"{P['h']['icc']:.3f}", f"{P['v']['icc']:.3f}"],
            ["within 5% / 10% of each other", f"{P['h']['w5']:.0f} / {P['h']['w10']:.0f}",
             f"{P['v']['w5']:.0f} / {P['v']['w10']:.0f}"]]))
        L.append("")
        L.append(f"{P['a']} vs {P['b']} on {P['n']} blind scenes; weave kappa {P['weave_kappa']:.2f} (agreement "
                 f"{P['weave_agree']:.0f}%), warp direction {P['warp_agree']:.0f}%; anchoring blind {fnum(an.get('blind_clean_pct'))}% "
                 f"/ assisted {fnum(an.get('shown_pct'))}%; median time blind {fnum(G['time_min'].get('blind'))} / assisted "
                 f"{fnum(G['time_min'].get('shown'))} min. Tab. 1 human row: {P['b']}'s counts scored against {P['a']}'s.")
        if iaa_withheld(ctx):
            L.append("**Not written to the paper** (blind NO-GO, PLAN 3): Tab. S4 and the Tab. 1 human row say "
                     "'agreement study ongoing'; anchoring and time per image are still reported.")
        av = P.get("attention_vs_disagreement")
        if av:
            L.append(f"Attention level (tool v2.2, computed for the IAA reps but never shown) vs annotator disagreement on "
                     f"the {av['n']} blind scenes: Spearman {av['spearman']:.2f} (p = {av['p']:.3g}); median relative "
                     "difference by level " + ", ".join(f"{l} {v:.1f}% (n = {av['n_by_level'][l]})"
                                                        for l, v in sorted(av['median_rel_by_level'].items())) + ".")
    elif single and G.get("proposal_edits"):
        pe = G["proposal_edits"]
        s, c = pe["all"], pe.get("clean")
        L.append(f"Single-annotator protocol: no inter-annotator agreement. Post-hoc proposal check (supp Tab. S4, "
                 f"`tables/proposal_edits150.csv`; one-to-one 1-D matching within {pe['tol']} x the human dot spacing, "
                 f"'off-centre' = matched but more than {pe['move_tol']} x the spacing apart):")
        L.append("")
        ns = pe.get("never_shown") or {}
        sets = [("all lines", s), (f"never shown (Tab. S4: without the {pe['n_guide_scenes']} guide scenes and "
                                   f"{pe.get('n_not_free_scenes', 0)} not-proven records)", ns)]
        sets += [(f"  {l} lines (Tab. S4 columns)", ns["by_line"][l]) for l in ("h", "v") if l in ns.get("by_line", {})]
        sets += [("clean (never shown, no badge, line not moved or turned)", c)]
        L.append(md_table(["lines", "n lines (scenes)", "add / delete / off-centre (median)", "machine dots matched %",
                           "human dots found %", "<= 2 edits %", "0 edits %"],
                          [[nm, f"{x['n_lines']} ({x['n_scenes']})",
                            f"{x['add_med']:.0f} / {x['delete_med']:.0f} / {x['move_med']:.0f}",
                            f"{x['precision_med']:.0f}", f"{x['recall_med']:.0f}", f"{x['le2_edit_pct']:.0f}",
                            f"{x['zero_edit_pct']:.0f}"] for nm, x in sets if x]))
        L.append("")
        L.append(f"Median edits per line {s['edits_med']:.0f} ({s['edits_pct_med']:.1f}% of the human dots); "
                 f"|n_prop - n_human| / n_human median {s['count_diff_med']:.1f}%. Badge scenes among them: "
                 f"{pe['n_hint_scenes']}; guide lines moved or turned by the annotator: {pe['n_moved_or_rotated_lines']}. "
                 f"Median time per photo {fnum(G['time_min'].get('all'))} min (from scratch; no assisted time was "
                 "measured, so no time-saving claim).")
    else:
        L.append("Not enough double-annotated blind scenes.")
    if V.get("tool_versions"):
        L.append(f"Tool versions of the finished records: {V['tool_versions']}; records that may have seen the v2.2 "
                 f"'check carefully' hint: {V.get('attention_hint_records', 0)}.")
    # headline numbers
    L += ["", "## 4. Results", ""]
    if Sc.get("methods"):
        L.append(f"Scene unit (primary), n = {Sc['n_scenes']} scenes (Tab. 1: {Sc.get('headline', 'all')} labels, "
                 f"n = {Sc.get('n_headline', Sc['n_scenes'])}), annotators {Sc['annotators']}; human counts "
                 f"h {Sc['h_range']}, v {Sc['v_range']} TPI (min/median/max); weave {Sc['human_weave']}.")
        L.append("")
        tr = []
        for item in (json.load(open(ctx.a.tab1_spec)) if ctx.a.tab1_spec else TAB1_SPEC):
            if item == "midrule":
                continue
            k, disp = item
            if k == "@human" and single:
                continue
            m = G.get("human_row") if k == "@human" else Sc.get("methods_headline", Sc["methods"]).get(k)
            if k == "@human" and not ctx.dual and iaa_withheld(ctx):
                tr.append([disp, f"{(G.get('primary') or {}).get('n', 0)} blind pairs", "withheld (blind NO-GO, PLAN 3): "
                           "'agreement study ongoing'", "", "", "", "", "", ""])
                continue
            if not m or not m.get("n_scored"):
                tr.append([disp, "--", "TBD", "", "", "", "", "", ""])
                continue
            tr.append([disp, f"{int(m['n_scored'])}/{int(m['n_gt'])}", f"{m['MAPE']:.1f} [{m['MAPE_lo']:.1f}, {m['MAPE_hi']:.1f}]",
                       f"{m['MAE']:.1f}", f"{m['w5']:.0f}", f"{m['w10']:.0f}", fnum(m.get("both5"), 0),
                       fnum(m.get("bias_med"), 1, True), fnum(m.get("weave_acc"), 0)])
        L.append(md_table(["Tab. 1 row", "n", "MAPE [95% CI]", "MAE", "<=5%", "<=10%", "both<=5%", "median signed %",
                           "weave %"], tr))
        L += ["", "Paired differences (per-scene mean APE, method a - method b; scene bootstrap, Wilcoxon):", ""]
        L.append(md_table(["a vs b", "n", "MAPE a", "MAPE b", "delta [95% CI]", "p", "a closer / b closer"],
                          [[k, v["n"], fnum(v.get("mape_a")), fnum(v.get("mape_b")),
                            f"{v['delta']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]" if "delta" in v else "--",
                            f"{v['p_wilcoxon']:.3g}" if "delta" in v else "--",
                            f"{v.get('a_closer', '')} / {v.get('b_closer', '')}"] for k, v in Sc["paired"].items()]))
        if Sc.get("headline") == "blind" and Sc.get("paired_blind"):
            L += ["", f"Same comparisons with blind labels only (the headline; n = {Sc.get('n_scenes_blind')} scenes):", ""]
            L.append(md_table(["a vs b", "n", "MAPE a", "MAPE b", "delta [95% CI]", "p"],
                              [[k, v["n"], fnum(v.get("mape_a")), fnum(v.get("mape_b")),
                                f"{v['delta']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]" if "delta" in v else "--",
                                f"{v['p_wilcoxon']:.3g}" if "delta" in v else "--"] for k, v in Sc["paired_blind"].items()]))
        L += ["", "Subsets and sensitivity rows (resolution class, 'unknown'-weave scenes, guide/badge scenes, "
              "proposal-free records, partial lines rescaled):", ""]
        L.append(md_table(["subset", "method", "n", "MAPE [95% CI]", "both<=5%", "median signed %"],
                          [[r["subset"], r["method"], fnum(r.get("n_scored"), 0),
                            f"{r['MAPE']:.1f} [{r['MAPE_lo']:.1f}, {r['MAPE_hi']:.1f}]" if r.get("MAPE") is not None else "--",
                            fnum(r.get("both5"), 0), fnum(r.get("bias_med"), 1, True)] for r in Sc["subsets"]]))
        ca = Sc.get("cascade", {})
        L += ["", f"Cascade (frozen gate, `tables/cascade_150_decisions.csv`, thresholds fitted on synthetic crops only, "
              f"so every scene is held out): escalates {ca.get('escalated')} of {ca.get('n')} labelled scenes "
              f"({fnum(ca.get('escalated_pct'))}%), US$ {fnum(ca.get('usd_per_1000_cascade'), 2)} per 1,000 vs "
              f"{fnum(ca.get('usd_per_1000_gpt_only'), 2)} GPT-only."]
        for part in ("escalated", "kept_offline"):
            seg = [f"{m}: {fnum(ca[f'{part}|{m}'].get('MAPE'))}%" for m in (OURS, GPT, "cascade-gate") if f"{part}|{m}" in ca]
            if seg:
                L.append(f"- {part.replace('_', ' ')} scenes: " + "; ".join(seg))
        if Sc.get("unknown13"):
            u = Sc["unknown13"]
            L += ["", f"13 'unknown'-weave scenes ({u.get('n_labelled', '?')} labelled, {u['n']} of them also scored by the "
                  f"pseudo-label CNN; numbers on those {u['n']}): ours {u['ours_mape']:.1f}% (signed "
                  f"{u['ours_median_signed']:+.1f}%), CNN + pseudo-labels {u['cnn_mape']:.1f}% (signed "
                  f"{u['cnn_median_signed']:+.1f}%); the CNN sits a median {u['cnn_vs_ours_median_abs_diff']:.1f}% from ours."]
        p4 = Sc.get("pseudo46", {})
        L.append(f"Pseudo-label CNN: held out on {p4.get('heldout_reps')} scenes; without the 9 re-shoots of training scenes on "
                 f"{p4.get('after_drop')} (row `{PSEUDO_KEY}@39` in `main150.csv`).")
        d = Sc.get("distortion")
        if d:
            L.append(f"Distortion basis (REVIEW G-P0-4; k1 = {d['k1']}, lattice reads {d['mult']}x the area-mean excess): "
                     f"humans' guide lines read {d['human_line_excess_median']:+.2f}% above the physical density "
                     f"(median of {d['n_counts']} counts), the area mean {d['area_mean_excess']:+.2f}%; expected "
                     f"distortion-only offset of ours vs humans {d['ours_expected_offset_median']:+.2f}%, observed "
                     f"median signed error {fnum(d.get('ours_bias_med_observed'), 2, True)}%.")
        cf = Sc.get("confidence") or []
        if cf:
            L += ["", "Confidence vs error against human counts (`quality150.py`, per scene):", ""]
            L.append(md_table(["estimator", "n", "Spearman(conf, error)", "AUC both<=10%", "both<=10% all",
                               "coverage conf>=0.9", "both<=10% at conf>=0.9"],
                              [[r["estimator"], r["n_images"], fnum(r.get("spearman_conf_vs_err"), 2),
                                fnum(r.get("auc_conf_both_within10"), 2), fnum(r.get("both_within10_all"), 0),
                                fnum(r.get("coverage_conf_ge_0.9"), 0), fnum(r.get("both_within10_conf_ge_0.9"), 0)]
                               for r in cf]))
        if Sc.get("fabric"):
            L += ["", "Fabric-group sensitivity (44 fabric groups, tables/fabric_groups_v2.csv; MAPE CI clustered by fabric vs by scene): " +
                  "; ".join(f"{m}: [{v['MAPE_lo']:.1f}, {v['MAPE_hi']:.1f}] vs [{Sc['methods'][m]['MAPE_lo']:.1f}, "
                            f"{Sc['methods'][m]['MAPE_hi']:.1f}]" for m, v in Sc["fabric"].items() if m in Sc["methods"])]
        if Sc.get("perfile"):
            L += ["", "Per-file (secondary, scene-clustered CI): " +
                  "; ".join(f"{m} {v['MAPE']:.1f}% [{v['MAPE_lo']:.1f}, {v['MAPE_hi']:.1f}] ({int(v['n_scored'])} of "
                            f"{v['n_files']} files, {v['n_clusters']} scenes)"
                            for m, v in Sc["perfile"].items() if v.get("MAPE") is not None)]
        L += ["", "Worst 10 scenes per method (author spot-check: is the label or the method wrong?):", ""]
        for m, ws in Sc.get("worst", {}).items():
            L.append(f"- {m}: " + ", ".join(f"#{w['scene']} ({w['res'][0]}, {w['ape']:.0f}%: {w['h']:.0f}/{w['v']:.0f} vs "
                                             f"{w['h_p']:.0f}/{w['v_p']:.0f})" for w in ws))
    cn = N.get("cnn", {})
    L += ["", "## 5. CNN cross-validation", "", f"Status: {cn.get('status', 'not run')}."]
    for c, rs in (cn.get("summary") or {}).items():
        for r in rs:
            L.append(f"- {c} / {r.get('backbone')} / {r.get('experiment')}: MAPE {r.get('MAPE_mean')} +- {r.get('MAPE_std')} "
                     f"(n = {r.get('n_images')} scenes, train_cv.py fold mean)")
    # changes
    prev, ch, other = numbers_diff(ctx)
    L += ["", "## 6. Numbers that changed", ""]
    if prev is None:
        L.append("No earlier label-day run in this location: every number above is new (the PDFs showed `TBD`).")
    elif not ch and not other:
        L.append(f"No change against `{prev}`.")
    else:
        L.append(f"Against the previous run `{prev}`: {len(ch)} headline values changed"
                 + (f", plus {other} other values (all in numbers.json)" if other else "") + ":")
        L.append("")
        L += [f"- `{k}`: {a} -> {b}" for k, a, b in ch]
    # rewording
    L += ["", "## 7. Text that now needs rewording", ""]
    if Sc.get("methods"):
        for it in rewording(ctx):
            L.append(f"- **{it['rule']}**: {it['why']}")
            for f, n, s in it["hits"][:6]:
                L.append(f"  - `{f}:{n}`: {s}")
    hits = marker_scan(ctx.paper)
    main_hits = [h for h in hits if not h[0].startswith("supp") and "supp_" not in h[0]]
    L += ["", f"TBD/TODO markers still active in the sources: {sum(h[2] for h in hits)} TBD and {sum(h[3] for h in hits)} "
          f"TODO on {len(hits)} lines ({len(main_hits)} lines in the main paper; fallback branches of "
          "`\\IfFileExists` blocks whose file now exists are not counted):", ""]
    L += [f"- `{f}:{n}` ({t} TBD, {d} TODO): {s}" for f, n, t, d, s in hits]
    sup = open(os.path.join(ctx.paper, "supp.tex")).read()
    need = [f for f in supp_hooks(ctx) if f"tables/{f}" not in sup]
    if need:
        L += ["", "Supplement hooks missing in `supp.tex` (generated but not shown): " + ", ".join(f"`tables/{f}.tex`" for f in need)]
    # build
    L += ["", "## 8. Build", ""]
    for name in ("main", "supp"):
        b = B.get(name)
        if b:
            L.append(f"- `{name}.pdf`: {b.get('n_pages')} pages" +
                     (f", body {b.get('body_pages')} pages (references from p. {b.get('refs_page')})" if name == "main" else "") +
                     f"; overfull {b['overfull']}, underfull {b['underfull']}, undefined refs {b['undefined_refs']}, "
                     f"undefined cites {b['undefined_cites']}; rendered TBD {b.get('tbd_rendered', '?')}, "
                     f"TODO {b.get('todo_rendered', '?')}.")
    if B.get("validate_tex"):
        L.append(f"- `validate_tex.py`: exit {B['validate_tex']['rc']}: " + " / ".join(B["validate_tex"]["tail"][-2:]))
    if B.get("main", {}).get("body_pages", 0) > 8:
        L.append("- **Body exceeds 8 pages**: apply the PLAN section 4 cut list.")
    L += ["", "## 9. Not done by label_day.py (PLAN 1b / 2)", "",
          "- Author TODOs (names, site/sessions/fabrics, consent and licence, model-card URLs, Tab. S3 corner regressor): "
          "listed among the markers above.",
          "- Glass measurement (absolute TPI; REVIEW G-P0-1) and the combined scale/distortion sentence.",
          "- Optional: PickGlass-Stress on real150 with labels (`camera_aug/build_stress.py --set real150 --labels "
          "<labels-csv>`, per scene); Molmo2 S3 scaffold on the 150 (`scaffold_queue.sh rect150`).",
          "- Writing section 5.1 and the abstract/conclusion sentences from the numbers above (Claude, Oct 13), then an "
          "Overleaf compile and `validate_tex.py --final` (Oct 14)."]
    L += ["", "## 10. Files written", ""] + [f"- `{w}`" for w in dict.fromkeys(ctx.written)]
    md = "\n".join(L) + "\n"
    dest = os.path.join(ctx.paper, "LABEL_DAY_REPORT.md")
    if ctx.failed and ctx.mode == "real":
        dest = os.path.join(ctx.work, "LABEL_DAY_REPORT.md")
    ctx.put(dest, md)
    with open(os.path.join(ctx.work, "LABEL_DAY_REPORT.md"), "w") as f:
        f.write(md)
    ctx.log(f"  report: {dest}")


# ============================================================================ main
STEPS = [(1, "validate", step_validate), (2, "propagate", step_propagate), (3, "agreement", step_agreement),
         (4, "score", step_score), (5, "cnn", step_cnn), (6, "tables", step_tables), (7, "compile", step_compile),
         (8, "report", step_report)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="*", help="annotation exports (JSON/CSV, globs ok); default: "
                                               "annotation_tool/data/saved/*.json|csv")
    ap.add_argument("--out-root", help="DRY RUN: write everything under DIR/runs/<stamp>/ (scratch copy of the paper)")
    ap.add_argument("--real", action="store_true", help="write the real outputs (paper/tables, PDFs, report)")
    ap.add_argument("--allow-partial", action="store_true", help="--real even when fewer than 25 scenes are labelled")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="rename an annotator in the merged copies (e.g. anonymous=alice); exports are never changed")
    ap.add_argument("--skip-bad-files", action="store_true", help="skip unreadable exports instead of stopping")
    ap.add_argument("--protocol", choices=("auto", "single", "dual", "iaa"), default="auto",
                    help="dual (two annotators, A1 + A2): both count every scene from scratch, no proposals "
                         "-> agreement over every scene counted by both (go/no-go: human-human median |diff| GO <= 5%%, "
                         "REDUCED <= 10%%), Tab. 1 'Human vs human' row, reference = mean of the two, supp Tab. S4 with "
                         "the agreement rows, no anchoring rule / blind-only run; single: ONE annotator counts "
                         "every scene from scratch, no proposals, no IAA -> go/no-go on scene coverage only, no "
                         "--blind-only run, no human row in Tab. 1, supp Tab. S4 = post-hoc proposal check; iaa: the first "
                         "two-annotator protocol; auto: single when exactly one annotator has finished records, dual when "
                         "two or more have and no finished record carries proposal dots, else iaa (with a warning)")
    ap.add_argument("--edit-tol", type=float, default=0.5,
                    help="single/dual protocol: a proposal dot matches a human dot within this fraction of the human spacing")
    ap.add_argument("--move-tol", type=float, default=0.25,
                    help="single/dual protocol: a matched pair further apart than this fraction of the spacing counts as "
                         "'off-centre' (a drag an assisted annotator might make; it does not change the count)")
    ap.add_argument("--proposals-hidden-at", default=PROPOSALS_HIDDEN_AT,
                    help="single/dual protocol: non-IAA records created before this instant (ISO, UTC) may have shown proposal "
                         "dots in the annotator kit and are not counted as proposal-free")
    ap.add_argument("--headline", choices=("auto", "all", "blind"), default="auto",
                    help="labels for Tab. 1: auto = blind-only if the anchoring rule fires (pre-declared rule)")
    ap.add_argument("--tab1-spec", help="JSON list of [key, LaTeX name] or \"midrule\" replacing TAB1_SPEC")
    ap.add_argument("--tab1-ci-decimals", type=int, default=0, choices=(0, 1),
                    help="decimals of the Tab. 1 CI bounds (0 = as in Tab. 2; 1 = as score150.py writes them)")
    ap.add_argument("--stop-after", type=int, default=8, help="stop after this step (1-8); the report is still written")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--k1", type=float, default=-0.026, help="radial distortion k1 (supp S16) for the distortion basis")
    ap.add_argument("--lattice-distortion-mult", type=float, default=2.2,
                    help="assumption: PG-Lattice reads about this multiple of the area-mean distortion excess")
    g = ap.add_argument_group("CNN cross-validation (step 5)")
    g.add_argument("--cnn", choices=("off", "launch", "wait"), default="off",
                   help="launch the scene-grouped 5-fold train_cv.py runs (launch: do not wait; wait: poll until done)")
    g.add_argument("--cnn-host", choices=("gpu-server", "local"), default="gpu-server")
    g.add_argument("--cnn-configs", default="base,aug1016,wl,aug1016_wl",
                   help="subset of base (existing aug), aug1016 (--extra-aug1016), wl (--augment whitelist), aug1016_wl")
    g.add_argument("--cnn-backbones", default="resnet18", help="train_cv --backbones (synthetic checkpoint: resnet18)")
    g.add_argument("--cnn-epochs", type=int, default=15)
    g.add_argument("--cnn-args", default="", help="extra train_cv.py arguments (quoted)")
    g.add_argument("--cnn-gpus", default="0,3", help="candidate GPUs on gpu-server (only idle ones are used)")
    g.add_argument("--cnn-collect", nargs="?", const="latest", default=None,
                   help="pull + score an earlier launch (state JSON, default the latest)")
    g.add_argument("--cnn-wait", action="store_true", help="with --cnn-collect: wait until every run finished")
    g.add_argument("--cnn-poll-s", type=int, default=300)
    g.add_argument("--cnn-timeout-h", type=float, default=12.0)
    g.add_argument("--cnn-dry", action="store_true", help="gpu-server: read-only preflight, print the commands, launch nothing")
    g.add_argument("--cnn-python", default=None, help="python of the CNN venv for --cnn-host local")
    t = ap.add_argument_group("build")
    t.add_argument("--tectonic", default=DEF_TECTONIC)
    t.add_argument("--tectonic-cache", default=DEF_TECTONIC_CACHE)
    t.add_argument("--pdf-python", default=DEF_PDF_PY, help="python with pypdf (page / TBD analysis; optional)")
    a = ap.parse_args(argv)
    if a.real == bool(a.out_root):
        ap.error("choose exactly one of --out-root DIR (dry run into a scratch copy) or --real")
    if a.cnn_collect and a.cnn != "off":
        ap.error("--cnn-collect pulls an earlier launch: do not combine it with --cnn launch/wait")
    return a


def main(argv=None):
    a = parse_args(argv)
    ctx = Ctx(a)
    try:
        setup(ctx)
    except StepError as e:
        print(f"label_day.py: {e}", file=sys.stderr)
        return 2
    rc = 0
    t_all = time.time()
    for k, name, fn in STEPS:
        if k > a.stop_after and name != "report":
            continue
        if name == "report":
            pass
        ctx.log("")
        ctx.log(f"==== [label_day step {k}/8] {name} " + "=" * max(0, 60 - len(name)))
        t0 = time.time()
        try:
            fn(ctx)
            ctx.steps_done.append(name)
            ctx.log(f"  ok ({time.time() - t0:.0f} s)")
        except StepError as e:
            ctx.failed = (k, name, str(e))
            ctx.log(f"STOPPED at step {k} ({name}): {e}")
            rc = 1
        except Exception as e:  # noqa: BLE001 -- any bug must stop the pipeline with a traceback in the log
            ctx.failed = (k, name, traceback.format_exc())
            ctx.log(f"STOPPED at step {k} ({name}) with an unexpected error:\n{traceback.format_exc()}")
            rc = 1
        if ctx.failed:
            if name != "report":
                try:
                    ctx.log("")
                    ctx.log("==== [label_day step 8/8] report (after the failure) " + "=" * 20)
                    step_report(ctx)
                except Exception:  # noqa: BLE001
                    ctx.log("  (report could not be written)\n" + traceback.format_exc())
            break
    ctx.log("")
    ctx.log(f"label_day.py {'FAILED' if rc else 'done'} in {(time.time() - t_all) / 60:.1f} min; mode {ctx.mode}; "
            f"steps {ctx.steps_done}; warnings {len(ctx.warnings)}")
    if ctx.mode == "dry":
        ctx.log(f"  dry run: everything is under {ctx.run_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

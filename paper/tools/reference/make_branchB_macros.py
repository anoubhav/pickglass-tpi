"""make_branchB_macros.py -- writes branchB_numbers.tex: one LaTeX macro (\\BN...) per number on the two annotators,
re-shoot repeatability and the human noise floor (main Secs. 3, 5.1-5.4, Tab. 2; supplement Secs. S4, S17).
Re-run after any input changes. In the macro names, NG stands for A2 (the reference annotator) and SA for A1 (the
non-expert).

Inputs (in --in-dir, the outputs of the other scripts of this folder):
  reshoot_consistency.json  (reshoot_consistency.py)      human_floor_paired.json  (human_floor_paired.py)
  branchB_numbers.json      (branchB_numbers.py)          logratio_endpoint.json   (logratio_endpoint.py)
  and, for \\BNlooNG, \\BNgptNGfive and \\BNensNGfive (leave-one-out MAPE against A2; scenes with both counts within
  5% of A2 for GPT-6.1-sol and the CNN-VLM ensemble), either
    --per-annotator-table   the per-annotator Tab. 1 (method, MAPE against each reference and P5 per reference), which
                            was computed from paper/tools/label_day.py single-annotator runs (not part of this folder);
                            it also gives \\BNlooSA (leave-one-out MAPE against A1; the macro is unused, but the value
                            appears in Tab. S15, which comes from the per-annotator scoring of the frozen pipeline), or
    tab1_ref_rows.csv       (make_tab1_ref.py; same values at the printed precision; \\BNlooSA is then not written).
  --a1 / --a2 per-scene counts and methods/predictions_ours_150.csv (\\BNoursOct*: scenes on which PG-Lattice reads
  half or double the annotator, the paper's band | |log2 r| - 1 | <= 0.3).
Typed-in constants (no script in this folder): \\BNtime* (median active minutes per photo, from the annotation tool's
time records), \\BNcv* (median coefficient of variation of the dot spacing per line, from both annotators' dot
positions), \\BNoracle*SA (octave upper bound against A1; paired_ref.py, block octave_upper_bound_vs_A2 /
validation_same_code, reproduces 13.3 and 13.6; both macros are unused and the paper does not print these values),
\\BNrfour* (rule 4 on the paper's band; rule4_band.py).

    python paper/tools/reference/make_branchB_macros.py [--in-dir DIR] [--per-annotator-table CSV] [--a1 ..] [--a2 ..] [--out DIR]

Output: <out>/branchB_numbers.tex.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT_DEFAULT = os.path.join(ROOT, "paper", "tools", "reference", "out")
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--in-dir", default=None, help="directory with the JSON outputs of this folder's scripts (default --out)")
ap.add_argument("--per-annotator-table", default=None, help="per-annotator Tab. 1 CSV (optional; see above)")
ap.add_argument("--out", default=OUT_DEFAULT, help="output directory")
args = ap.parse_args()
O = args.in_dir or args.out

rs = json.load(open(f"{O}/reshoot_consistency.json")); hf = json.load(open(f"{O}/human_floor_paired.json")); bn = json.load(open(f"{O}/branchB_numbers.json"))
A = pd.read_csv(args.a1).set_index("scene_id"); B = pd.read_csv(args.a2).set_index("scene_id")
ours = pd.read_csv(os.path.join(ROOT, "methods/predictions_ours_150.csv")).set_index("id")
def octs(R):   # the paper's half-or-double band (Sec. 5.2): | |log2 r| - 1 | <= 0.3
    return int(sum(any(abs(abs(np.log2(ours.at[s, f"{ax}_count"] / R.at[s, f"{ax}_tpi"])) - 1) <= 0.3 for ax in "hv") for s in R.index))
lr = json.load(open(f"{O}/logratio_endpoint.json"))
sg2 = lambda x: f"\\ensuremath{{{x:+.2f}}}"
ci2 = lambda c: f"[{sg2(c[0])}, {sg2(c[1])}]"
f1 = lambda x: f"{x:.1f}"; sg = lambda x: f"\\ensuremath{{{x:+.1f}}}".replace("+-", "-")
def ci(c): return f"[{sg(c[0])}, {sg(c[1])}]"
# leave-one-out MAPE and P5 from the per-annotator table when given, else from make_tab1_ref.py's rows (A2 only)
if args.per_annotator_table:
    T = pd.read_csv(args.per_annotator_table).set_index("method")
    # the table's reference columns carry the annotator codes of the original run; read them under A1 / A2
    T = T.rename(columns=lambda c: c.replace("FL_NG", "A2").replace("FL_SA", "A1"))
    LOO = {"BNlooNG": f1(T.at["Leave-one-out mean", "A2"]), "BNlooSA": f1(T.at["Leave-one-out mean", "A1"]),
           "BNgptNGfive": str(int(T.at["GPT-6.1-sol", "A2_p5"])), "BNensNGfive": str(int(T.at["ens-v1 = CNN x GPT (frozen)", "A2_p5"]))}
else:
    t1 = pd.read_csv(f"{O}/tab1_ref_rows.csv").set_index("key")
    LOO = {"BNlooNG": f1(t1.at["Leave-one-out mean (reference)", "MAPE"]),
           "BNgptNGfive": str(int(t1.at["GPT-6.1-sol, P1", "P5_n"])), "BNensNGfive": str(int(t1.at["ens-v1", "P5_n"]))}
    print("note: no --per-annotator-table: \\BNlooNG, \\BNgptNGfive, \\BNensNGfive from tab1_ref_rows.csv; \\BNlooSA not written")
H2 = "GPT-6.1-sol vs A2 minus A1 vs A2"; H1 = "GPT-6.1-sol vs A1 minus A2 vs A1"
M = {
 "BNsamePhoto": f1(bn["same_photo_median_absdiff_pct_rel_to_pair_mean"]),
 "BNlinesBig": str(bn["N2_lines_diff_gt10"]), "BNlinesBigOct": str(bn["N2_of_which_about_2x"]),
 "BNdisputedLines": str(bn["prereg_2.1_disputed_lines_abs_log2_gt_0.5"]), "BNdisputedScenes": str(bn["prereg_2.1_disputed_scenes"]),
 "BNrsSA": f1(rs["A1"]["median_pct"]), "BNrsNG": f1(rs["A2"]["median_pct"]), "BNrsMean": f1(rs["mean"]["median_pct"]),
 "BNrsOurs": f1(rs["PG-Lattice"]["median_pct"]), "BNrsGpt": f1(rs["GPT-6.1-sol"]["median_pct"]), "BNrsEns": f1(rs["ens-v1"]["median_pct"]),
 "BNrsSAfive": str(rs["A1"]["within5"]), "BNrsNGfive": str(rs["A2"]["within5"]), "BNrsOursFive": str(rs["PG-Lattice"]["within5"]),
 "BNrsGptFive": str(rs["GPT-6.1-sol"]["within5"]), "BNrsEnsFive": str(rs["ens-v1"]["within5"]), "BNrsMeanFive": str(rs["mean"]["within5"]),
 "BNrsSAtwo": str(rs["A1"]["near2x"]), "BNrsNGtwo": str(rs["A2"]["near2x"]), "BNrsOursTwo": str(rs["PG-Lattice"]["near2x"]),
 "BNrsGptTwo": str(rs["GPT-6.1-sol"]["near2x"]), "BNrsEnsTwo": str(rs["ens-v1"]["near2x"]), "BNrsMeanTwo": str(rs["mean"]["near2x"]),
 "BNhumNG": f1(hf[H2]["human"]), "BNhumSA": f1(hf[H1]["human"]),
 "BNgptNG": f1(hf[H2]["method"]), "BNgptSA": f1(hf[H1]["method"]),
 "BNensNG": f1(hf["ens-v1 vs A2 minus A1 vs A2"]["method"]), "BNensSA": f1(hf["ens-v1 vs A1 minus A2 vs A1"]["method"]),
 "BNoursNG": f1(hf["PG-Lattice vs A2 minus A1 vs A2"]["method"]), "BNoursSA": f1(hf["PG-Lattice vs A1 minus A2 vs A1"]["method"]),
 "BNgptMinusHumNG": sg(hf[H2]["diff"]), "BNgptMinusHumNGci": ci(hf[H2]["ci"]),
 "BNgptMinusHumSA": sg(hf[H1]["diff"]), "BNgptMinusHumSAci": ci(hf[H1]["ci"]),
 "BNensMinusHumNG": sg(hf["ens-v1 vs A2 minus A1 vs A2"]["diff"]), "BNensMinusHumNGci": ci(hf["ens-v1 vs A2 minus A1 vs A2"]["ci"]),
 "BNensMinusHumSA": sg(hf["ens-v1 vs A1 minus A2 vs A1"]["diff"]), "BNensMinusHumSAci": ci(hf["ens-v1 vs A1 minus A2 vs A1"]["ci"]),
 "BNoursMinusHumNG": sg(hf["PG-Lattice vs A2 minus A1 vs A2"]["diff"]), "BNoursMinusHumNGci": ci(hf["PG-Lattice vs A2 minus A1 vs A2"]["ci"]),
 "BNoursOctSA": str(octs(A)), "BNoursOctNG": str(octs(B)),
 "BNtimeSA": "3.8", "BNtimeNG": "9.6",   # median active minutes per photo (annotation tool time records)
 "BNlooNG": LOO["BNlooNG"], "BNlooSA": LOO.get("BNlooSA"),
 "BNgptNGfive": LOO["BNgptNGfive"], "BNensNGfive": LOO["BNensNGfive"],
 "BNcvSA": "0.19", "BNcvNG": "0.16",   # median dot-spacing CV per line (both annotators' dot positions, all lines)
 "BNlogGptNG": f"{lr['GPT-6.1-sol P1 vs A2']['mean_abs_log2']:.2f}", "BNlogHumNG": f"{lr['A1 (as prediction) vs A2']['mean_abs_log2']:.2f}",
 "BNlogEnsNG": f"{lr['ens-v1 vs A2']['mean_abs_log2']:.2f}",
 "BNlogGptMinusHum": sg2(lr['GPT minus A1, vs A2 (paired mean |log2|)']['diff']), "BNlogGptMinusHumci": ci2(lr['GPT minus A1, vs A2 (paired mean |log2|)']['ci']),
 "BNlogEnsMinusHum": sg2(lr['ens-v1 minus A1, vs A2 (paired mean |log2|)']['diff']), "BNlogEnsMinusHumci": ci2(lr['ens-v1 minus A1, vs A2 (paired mean |log2|)']['ci']),
 "BNoracleOursSA": "13.3", "BNoracleGptSA": "13.6",   # octave upper bound against A1 (paired_ref.py reproduces both)
 "BNrfourN": "64", "BNrfourScenes": "45", "BNrfourNGSA": "15", "BNrfourNGours": "28",   # rule 4 on the paper's band (rule4_band.py)
 "BNrsSAfull": f1(bn["reshoot_by_resolution_A1"]["both_full"]["median_pct"]), "BNrsSAquarter": f1(bn["reshoot_by_resolution_A1"]["both_quarter"]["median_pct"]),
 "BNrsNGfull": f1(bn["reshoot_by_resolution_A2"]["both_full"]["median_pct"]), "BNrsNGquarter": f1(bn["reshoot_by_resolution_A2"]["both_quarter"]["median_pct"]),
}
M = {k: v for k, v in M.items() if v is not None}
L = ["% generated by paper/tools/reference/make_branchB_macros.py; do not edit by hand"]
L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "branchB_numbers.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L[1:]))

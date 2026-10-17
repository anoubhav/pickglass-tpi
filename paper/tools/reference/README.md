# Scoring against the reference annotator

The scripts in this folder produce the paper's numbers (WACV 2027 workshop WVAQ) that are scored against **A2**, the
textile-trained annotator whose counts are the reference for every main result, and the numbers on how the two
annotators agree with each other. **A1** is the annotator without textile training, who serves as a human baseline:
the first row of main Tab. 1 scores A1's counts against A2 (37.3% MAPE). Each annotator counted all 76 scenes from
scratch.

## Status: post hoc unless the paper says otherwise

The analysis plan fixed before labeling scored against the **mean of both annotators**. Results against that reference
come from the frozen pipeline (`paper/tools/label_day.py`, `paper/tools/score150.py`, `paper/tools/iqa150.py`) and are in
supplement Tabs. S18 and S13. Each annotator scored alone is in Tab. S15. Scoring against A2 was decided after both
annotators' counts had been seen, and the paper declares this as a deviation from the plan. Every result in this folder
is therefore **post hoc** unless the paper (and the script's docstring) names it as fixed in advance. These are the
exceptions, all fixed before A2's counts were opened:

- the plan for the reference annotator (not part of this repository; available on request; SHA-256 `9ae1d9caab117b97...`,
  listed with its MD5 in [CHECKSUMS.md](../../../CHECKSUMS.md)):
  - rule 1: the histogram of log2(A2/A1) and the octave-disputed lines (|log2(A2/A1)| > 0.5)
  - rule 2: the mean without the disputed lines as a sensitivity reference, with both counts listed on every disputed
    line (Tab. S17)
  - rule 3: scoring against each annotator alone
  - rule 4: how main Sec. 5.2 is worded for each outcome on the lines where PG-Lattice reads about half or double A1
  - rule 5: the log-ratio endpoint

  (rule k is item k of section 2 of the plan; some script docstrings use the item numbers)
- its addendum 1 (not part of this repository; available on request; SHA-256 `586b606c38d6f0f8...`, also listed in
  CHECKSUMS.md), which numbers its own rules:
  - the re-shoot definitions (rule 1)
  - the noise-floor comparison (rule 4)
- the blind tests of the frozen methods (`methods/octave/eval_octave_v1.py` with the `PREREG_*.json` files of
  `methods/`), which their own pre-registration records already score against A2 alone.

The paired tests run on these quantities are post hoc unless the plan specifies the test. Each output records the status
of its comparisons.

## Human counts

The per-scene counts of both annotators will be released after review at `data/human_counts/labels_per_scene_A1.csv`
and `data/human_counts/labels_per_scene_A2.csv`. Every script uses these paths by default; `--a1` and `--a2` point to
other copies. Until the release, the scripts run only on counts you supply.

Required columns: `scene_id`, `h_tpi` and `v_tpi` (threads per window side; one row per scene, identified by the scene
representative's id). `branchB_numbers.py` also needs `scene_resolution` (`full` / `quarter_only`), and so does
`make_branchB_macros.py`, which reads `branchB_numbers.json`; with only the three required columns every other script
runs and gives the same results. `h_count` and `v_count` are optional. `annotation_tool/propagate_scene_labels.py
<export> --out-dir <d>` writes these columns, `scene_resolution` included, to `<d>/labels_per_scene.csv` (at the precision
of the export; see below).

**Precision matters.** The scripts that score the way `label_day.py` does round the counts to 2 decimals, because the
annotation tool's export stored 2 decimals:

- `make_tab1_ref.py`
- `expert_iqa.py`
- `ruler_vs_ours_ref.py`
- the checks in `paired_ref.py` that reproduce the rows stored by `label_day.py`

The other scripts use the counts exactly as given. To reproduce the paper exactly, use the counts at full precision.
For example, with A2's counts rounded to 2 decimals, A2's re-shoot median between full-resolution photos in supplement
Sec. S4 reads 8.4% instead of 8.3%. Within-5% tallies can differ by one line between scripts for the same reason:
`octave_tallies.json` of `make_fig_octave_hist.py` (counts as given) has PG-Lattice within 5% of A2 on 44 of 152 lines,
while Tab. 1 (`make_tab1_ref.py`, counts rounded to 2 decimals as in `label_day.py`) prints 28%, that is 43 of 152; the
paper quotes neither tally.

**AUROC orientation.** `branchB_numbers.json` stores the AUROC of each generic quality score for the scenes where A1
and A2 differ by more than 10% as P(score of such a scene > score of another scene). The paper reports 1 minus the
stored value: MUSIQ and TOPIQ-NR at 1024 px, stored 0.27 [0.14, 0.44] and 0.28 [0.09, 0.48], are 0.73 [0.56, 0.86]
and 0.72 [0.52, 0.91] in supplement Sec. S9 (main Sec. 5.4 states only that every AUROC interval on the working image
spans 0.5). `iqa_disagreement_mwu.py` prints both orientations.

## Running

Python 3 with numpy, pandas and scipy is needed, plus matplotlib for the figure. The checks were run with Python 3.11 and
3.12, numpy 1.26.4, pandas 3.0.6, scipy 1.17.1 and matplotlib 3.11.2. The scripts can be run from any directory. They read the
predictions and frozen tools from their repository paths, and they never write outside `--out` (default
`paper/tools/reference/out/`, which git ignores).

To run everything, in dependency order:

```
bash paper/tools/reference/run_all.sh [A1.csv] [A2.csv] [out dir]     # python: $PYTHON or python3
```

One script at a time (`--a1`/`--a2` are optional when the released counts are in place):

```
python paper/tools/reference/<script>.py --a1 A1.csv --a2 A2.csv --out OUT
bash paper/tools/reference/blind_tests_A2.sh [A2.csv] [out dir]     # python: $PYTHON or python3
```

Three Python scripts take other arguments (as `run_all.sh` calls them): `molmo_reshoot.py` is label-free and takes only `--out`
(and `--reshoot-json`); `iqa_disagreement_mwu.py` takes `--a1` and `--a2` (and `--iqa-summary`), has no `--out` and
prints its results; `ruler_vs_ours_ref.py` takes only `--a2` (and `--per-scene`) and prints its results.
`make_fig_octave_hist.py` refuses to overwrite an existing figure: before re-running it into the same `--out`, delete
`fig_octave_hist_H2.pdf` and `.png` there (`run_all.sh` does this, and writes the figure to `<out>/fig/`).
`make_branchB_macros.py` reads the JSON outputs of `reshoot_consistency.py`, `human_floor_paired.py`,
`branchB_numbers.py` and `logratio_endpoint.py` (and, without `--per-annotator-table`, the `tab1_ref_rows.csv` of
`make_tab1_ref.py`) from its `--in-dir` (default `--out`), so run those first (as `run_all.sh` does); a missing one
stops it with a `FileNotFoundError` naming the file. `branchB_numbers.py` stops with a
`KeyError` if the counts lack the `scene_resolution` column.

Run on the paper's counts, every script reproduces the values of the original analysis outputs exactly: every number
is identical and every printed table row matches the paper. In the outputs, keys name the two annotators A1 and A2 (for
example the `A1` and `A2` keys of `reshoot_consistency.json`, `"A1 counts"` in `paired_ref.json` and `A2_posthoc` in
`expert_iqa.json`). The original analysis outputs used internal annotator codes in these keys, which have been renamed;
the values are unchanged. Older repository file names keep the codes: `FLSA` stands for A1 and `FLNG` for A2. For
example, `methods/ensemble/cnn_fabric_oof_FLSA_150.csv` is the CNN trained on A1's counts. In the LaTeX macro names,
`SA` stands for A1 and `NG` for A2.

## What each script makes

Item numbers refer to the paper (main text and supplement).

| Script | Paper items | Outputs (in `--out`) |
|---|---|---|
| `make_tab1_ref.py` | **Main Tab. 1**, every row against A2: MAPE [95% CI], counts within 5% / 10%, scenes with both counts within 5% (Molmo2-4B + LoRA 25.2, GPT-6.1-sol P1 27.1, leave-one-out mean 33.2, PG-Lattice 34.8, non-expert 37.3); the symmetric error of every row (Sec. 5.1) | `main150_rows_ref.tex` (rows as generated); `main150_rows_refB.tex` (the rows as printed: final names, best method value per column in bold, a step done by hand for the paper); `tab1_ref_rows.csv` (unrounded values, sources, status); `tab1_ref_checks.json` |
| `paired_ref.py` | Main Sec. 5.1 paired tests against A2: Molmo2-4B minus GPT −1.9 [−6.5, +2.6], on \|log2\| −0.13 [−0.22, −0.05]; each minus the non-expert (Molmo2-4B −12.1 [−19.7, −4.6]; for GPT the paper prints the CI of `human_floor_paired.py`, [−17.9, −2.7], while this script's own bootstrap gives [−17.9, −2.9] and keeps −2.7 only as its `existing` value); margins over the leave-one-out mean, Molmo2-4B 8.0 [2.9, 13.3] and GPT 6.1 [0.6, 11.7], and PG-Lattice minus that mean +1.6 [−6.1, +9.4]. Sec. 5.2: PG-Lattice reads half or double on 36 lines (24 half, 12 double), GPT on 37 (34 half), the non-expert on 48 (30 double); octave upper bound 34.8 → 10.8% (65 lines). Supplement Sec. S4: patch-clustered CIs. Validation against A1: 41.7 → 13.3, GPT 13.6 (13.3 and 13.6 are check values that the paper does not print) | `paired_ref.json` |
| `human_floor_paired.py` | Main Sec. 5.1, Tab. 1 first row and Tab. 2 MAPE column: non-expert 37.3, GPT 27.1, PG-Lattice 34.8, ensemble 25.4; GPT minus non-expert −10.2 [−17.9, −2.7] (the CI printed in main Sec. 5.1). Against A1: PG-Lattice 41.7, GPT 28.8. Supplement Secs. S1, S4 | `human_floor_paired.json` |
| `logratio_endpoint.py` | Main Sec. 5.1: GPT minus the non-expert on \|log2\|, −0.07 [−0.16, +0.02]. Supplement S4: ensemble −0.13 [−0.20, −0.05]. Supplement S1: octave check minus PG-Lattice −0.021 [−0.043, −0.003] (secondary endpoint of rule 5); GPT minus A2, scored against A1, −0.06 [−0.17, +0.04] | `logratio_endpoint.json` |
| `reshoot_consistency.py` | **Main Tab. 2** re-shoot columns: A2 8.9 / 12 / 2, A1 20.5 / 8 / 12, PG-Lattice 1.8 / 24 / 8, GPT 14.0 / 11 / 4, ensemble 9.7 / 13 / 0. Sec. 5.3 (the re-shoot numbers of Tab. 2; see *Numbers not computed here*). Supplement S4: pairs within 3 / 5 / 10% | `reshoot_consistency.json` |
| `molmo_reshoot.py` | Main Tab. 2 and Sec. 5.3: the Molmo2-4B + LoRA row (14.3, 11, 1); label-free | `molmo_reshoot.json` |
| `branchB_numbers.py` | Main Sec. 3: same-photo median difference 32.0%. Tab. S17 and supplement S1: 72 disputed lines in 50 scenes. Supplement S4: re-shoots by resolution, A2 8.3 / 10.1, A1 28.6 / 7.8. Main Sec. 5.4 and supplement S9: AUROC of the generic scores for the scenes where A1 and A2 differ by more than 10% | `branchB_numbers.json` |
| `make_disputed_table.py` | **Supplement Tab. S17** | `supp_disputed_lines.tex` |
| `make_branchB_macros.py` | the `\BN...` macros of main Secs. 3 and 5.1-5.4, Tab. 2 and supplement Secs. S4 and S17; reads the outputs above (see its docstring for the typed-in constants) | `branchB_numbers.tex` |
| `make_fig_octave_hist.py` | **Main Fig. 3** and its tallies (Sec. 5.2) | `fig_octave_hist_H2.pdf/.png`, `octave_tallies.json` |
| `expert_iqa.py` | Main Sec. 5.4 and supplement S9, Spearman ρ against A2: BRISQUE −0.02 [−0.28, 0.22], NIQE −0.12, MUSIQ 0.07, CLIP-IQA 0.25 [0.02, 0.46], TOPIQ-NR 0.15, η −0.06, PG-Lattice's confidence −0.15 [−0.38, 0.11]. Frozen gate minus PG-Lattice −0.3 [−3.1, +2.3]. The frozen test's values against the mean (S9) | `expert_iqa.json`, `expert_iqa_per_scene.csv`, `frozen_tool_runs/` |
| `iqa_disagreement_mwu.py` | Supplement S9: Mann-Whitney p 0.014 / 0.015 (Holm 0.14) and AUROC 0.73 / 0.72 for MUSIQ and TOPIQ-NR at 1024 px | printed |
| `rule1_histogram.py` | Supplement S1: rule 1 histogram 1 / 29 / 12 / 62 / 24 / 16 / 8; 44 of the 72 disputed lines (33 scenes) fall among the 64 lines where PG-Lattice reads half or double A1. Tab. S17: 72 lines, 50 scenes | `rule1_histogram.json` |
| `rule4_band.py` | Main Sec. 5.2 and supplement S17: rule 4 on 64 lines (45 scenes); A2 is within 10% of A1 on 15 and of PG-Lattice on 28, so both readings stand (the third outcome of rule 4), and every band gives the same outcome | `rule4_band.json` |
| `ruler_vs_ours_ref.py` | Main Sec. 5.5: ruler crop against PG-Lattice on A2's counts, 35.3 → 37.9% on 75 scenes; the difference prints as +2.59 [+0.04, +6.65], that is +2.6 [0.0, 6.6] at one decimal (the paper's upper bound, 6.7, rounds the printed +6.65 a second time) | printed |
| `blind_tests_A2.sh` | the pre-registered blind tests (frozen `methods/octave/eval_octave_v1.py`). Main Sec. 5.1: ensemble minus GPT −1.7 [−4.2, +0.7]; CNN trained on A1 28.1%. Main Sec. 5.2 and S17: octave check minus PG-Lattice −1.0 [−2.2, −0.1]; learned selector and its secondary endpoints. Supplement S1: ens-v1b 26.4%, −0.7 [−3.8, +2.4] against GPT and +1.0 [−1.2, +2.8] against the ensemble | `eval_A2/*.json`, `eval_A2/*.txt` |
| `run_all.sh` | all of the above, in dependency order | everything above, plus `logs/` |

**Numbers not computed here.** These numbers of the paper that need both annotators' counts come from no script in
this folder:

- main Sec. 5.3: of PG-Lattice's 24 pairs within 5%, the 8 on which both photos read about half or double A2 (with the
  band of Sec. 5.2, a recomputation gives 8 when each photo may fall in the band on either axis, and 7 when both photos
  must fall in it on the same axis);
- supplement Sec. S4: the CI [10.2, 31.8] of A1's re-shoot median; the 7 pairs within 5% under the h-v assignment given
  by the measured rotation; the badge split (Fisher p = 0.79, Mann-Whitney p = 0.002);
- main Sec. 3 and supplement Sec. S4: the ICC(2,1) of 0.20 / 0.28 and the median difference of 22.8% on the 38 scenes
  without a badge, which come from the agreement statistics of the frozen `paper/tools/label_day.py` (`--protocol dual`;
  supplement Tab. S4);
- the dot-spacing CVs (0.16 / 0.19) and the annotation times, typed into `make_branchB_macros.py` (see its docstring);
- supplement Sec. S8: Molmo2-4B + LoRA against A1 (47.6%, +5.9 against PG-Lattice) and its −9.6 against PG-Lattice on
  A2's counts.

**Inputs from the repository** (read from their repository paths; the frozen ones are listed in
`tools/freeze_checksums.json`):

- frozen tools: `paper/tools/score150.py`, `paper/tools/label_day.py`, `paper/tools/iqa150.py`, `paper/tools/iqa_generic.py`
  and `methods/octave/eval_octave_v1.py`
- predictions:
  - `methods/predictions_ours_150.csv` (PG-Lattice and the log-Gabor bank)
  - `vlm/results/rect150_P1_predictions_wide.csv` (VLMs, prompt P1; `<model>__v` = our h)
  - `ft_results/results/rect150_predictions_wide.csv` (Molmo2-4B + LoRA, P1 readout)
  - `gpu_results/countgd/outputs/results/rect150_predictions_wide.csv`
  - `paper/tables/src/extra150_wide.csv` (ruler crop, cascade gate)
  - `cnn/runs_aug/report/real150_aug_wide_heldout.csv`
  - `cnn/runs/synth_r18_v2mix_aniso/eval_final/real150_predictions.csv`
  - `methods/results/combined_v1_150_wide.csv`
  - `methods/ensemble/*.csv`, `methods/ensemble_aug/predictions_ens_v1b_150.csv`, `methods/octave/predictions_octave_150.csv`
    and `methods/octave_learned/predictions_octsel_*.csv`
- re-shoot pairs and patches: `paper/tables/fabric_groups_v2.json` and `.csv`
- quality scores: `paper/tables/iqa150_scores.csv` and `paper/tables/cascade_150_decisions.csv`

Some scripts take optional `--a2-run`, `--a1-run`, `--mean-run`, `--eval-dir`, `--frozen-iqa-dir`,
`--final-numbers-json/-tex`, `--per-annotator-table/-md`, `--fix-numbers` or `--tab1-spec` arguments. These add
cross-checks against intermediate files of the original analysis, such as `label_day.py` runs on the annotation tool's
raw exports. Those files are not part of this repository. Without these arguments the checks are skipped and the
results do not change. `make_tab1_ref.py --logratio-json` (default `<out>/logratio_endpoint.json` when present) instead
checks the mean |log2| and the share above 0.5 of five rows against the output of this folder's `logratio_endpoint.py`.

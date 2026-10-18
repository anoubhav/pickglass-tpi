# PickGlass-TPI: counting threads per inch through a pick glass

*Formerly `pickglass-thread-count`; old links redirect here. The paper's dataset, PickGlass-TPI, is called PickGlass in this repository's files and documentation.*

Code, predictions and the annotation tool for **Image Quality and the Reliability of Fabric-Density Measurement: Counting Threads from Smartphone Photos Through a Pick Glass** (under review at the WACV 2027 workshop WVAQ, Image/Video/Audio Quality Assessment in Computer Vision, VLM and Diffusion Model).

A textile inspector checks thread density, the warp ends and weft picks per inch, by counting yarns through a 10x pick glass. We keep that tool and photograph the fabric **through** it with a phone: the glass's opening and engraved scales make every photo metric, and the problem becomes one of capture quality. This repository holds the task definition, our two counters (PG-Lattice, a training-free reciprocal-lattice counter, and Molmo2-4B + synthetic LoRA, a small open VLM tuned only on synthetic crops), a pre-registered combined counter (PG-Lattice + log-Gabor), the glass-ruler window detector, the capture-quality studies (degradations, phone-camera stress, generative enhancement, generic no-reference IQA), the VLM / CNN / counting-specialist baselines, a quality-aware cascade, a scene-aware annotation tool, and every prediction on PickGlass (150 files = 76 scenes) and on 13 hand-counted legacy crops.

![Pipeline on a synthetic fabric: photo through the glass, rectified window with one dot per yarn from the frozen counter, zoom, output](docs/img/overview_synthetic.png)

*Every image in this repository is synthetic (`cnn/synth.py`) or a synthetic practice fabric. No factory photo is included; see [Data](#data).*

**Status.**
- Frozen before any human count was looked at ([CHECKSUMS.md](CHECKSUMS.md)): the counter, the scorer, the cascade decisions, the combined counter, the 76-scene generic-IQA test, and the analysis of the human counts, `paper/tools/label_day.py`, which scores every method against the mean of both annotators' counts, the reference fixed in the analysis plan.
- Two external annotators each counted all 76 PickGlass scenes from scratch: A2, who has textile training, and A1, who has none. The paper scores every main result against A2 and reports A1's counts as a non-expert human baseline. Scoring against A2 instead of the mean of both is a declared deviation from the plan, decided after both annotators' counts were seen. The results against the mean of both are in the supplement (Tab. S6), and against each annotator alone, which the plan did specify, in Tab. S7.
- This release has the code, the frozen methods and protocols, and every machine prediction. The photos, crops, renders and both annotators' counts (with dot positions) will be released after review ([DATASHEET.md](DATASHEET.md)). The author took the photos with the mill's agreement.

## Our counters

- **PG-Lattice** (`methods/`; frozen copy `methods/results/counting_frozen_final.py`) reads the yarn lattice from the 2-D power spectrum of the rectified window, needs no training, and runs offline on a laptop CPU (about 3.5 s per crop). It was frozen before any human count.
- **Molmo2-4B + synthetic LoRA** (`ft_results/`) is Molmo2, an open point-then-count VLM, tuned with a LoRA on synthetic crops only. Pointing lets the exact yarn crossings that our crop generator knows (through rectification, resampling and perspective) supervise the model yarn by yarn, and the 4B version is small enough to tune on two consumer GPUs (5.9 GPU-hours) and for a mill to host. No real image or label was used in training; the checkpoint and its direct-count readout (prompt P1) were fixed on synthetic validation before any human count was scored. It was added to the main table after the human counts were in. `ft_results/` holds the synthetic point generator, training, inference, readouts, logs and predictions (no weights).

Against the textile-trained reference A2 (76 scenes; paper Tabs. 1 and 2), no method counts within the ±5% tolerance:

| Counts from | MAPE against A2 (%) | Re-shoots: median worse-axis difference over 41 pairs (%) |
|---|---|---|
| Molmo2-4B + synthetic LoRA (ours) | 25.2 | 14.3 |
| GPT-6.1-sol given four magnified insets of the window (row added post hoc) | 24.9 | - |
| GPT-6.1-sol, prompt P1 | 27.1 | 14.0 |
| Leave-one-out mean (each scene guessed from the other scenes) | 33.2 | - |
| PG-Lattice (ours) | 34.8 | 1.8 |
| A1, the non-expert annotator | 37.3 | 20.5 |
| A2, the reference annotator | - | 8.9 |

Of the blind methods, the lowest errors come from Molmo2-4B + LoRA and from GPT-6.1-sol given four magnified insets of the window, which are level on both error measures (insets minus Molmo2-4B: −0.3 points [−5.1, +4.7] in MAPE, +0.04 [−0.03, +0.13] on the symmetric error); the insets beat the leave-one-out mean on MAPE only (post hoc; paper Sec. 5.1, supplement Sec. S6). Molmo2-4B's lead over GPT-6.1-sol P1 is not reliable on MAPE (−1.9 points, 95% CI [−6.5, +2.6]); it is closer to A2 on the symmetric error (−0.13 [−0.22, −0.05]) and is the only method that reliably beats the leave-one-out mean on both error measures (by 8.0 points [2.9, 13.3] in MAPE, also over patches and on the symmetric error; paper Sec. 5.1). The ranking depends on the reference: against the mean of both annotators Molmo2-4B scores 28.2%, level with the leave-one-out mean (28.7%; supplement Tab. S6), and against A1 47.6% (supplement Sec. S7). Its tuned point lists are near-perfectly regular, so it reads the pitch rather than locating each yarn (Sec. S7).

## Contents

| Folder | What |
|---|---|
| `methods/` | PG-Lattice (`results/counting_frozen_final.py`, frozen; `counting.py`), the combined counter combined-v1 (`combined_v1.py`; rule `results/combined_frozen_v1.py` and `results/PREREG_combined_v1.json`, frozen), glass-ruler calibration (`ruler_calib.py`), degradations (`degrade.py`, `ablate.py`), the native-resolution guard (`guard/`), the octave check and the learned octave selector (`octave/`, `octave_learned/`), the CNN–VLM ensembles (`ensemble/`, `ensemble_aug/`), the faithful legacy baseline (`legacy_2019/`), all results |
| `ft_results/` | Molmo2-4B + synthetic LoRA (ours): synthetic point generator, LoRA training, inference, readouts, logs, predictions and tables |
| `common/` | `rectify.py` (window -> S x S crop), `paths.py` (where data outside the repo is expected) |
| `annotation_tool/` | the browser tool (v2.2.2 with the completeness rule), scene propagation, analysis, tests, a synthetic demo set (`demo/`) and the annotation kit (`kit/`: bundles, screening, export checks) |
| `cnn/` | synthetic fabric generator (`synth.py`), label-exact augmentation, training, scene-grouped CV, predictions (code and CSVs only, no weights) |
| `vlm/` | VLM prompts, OpenRouter client (key from the environment only), parsers, scaffolds, roster, raw answers and results |
| `gpu_results/` | the counting specialists Molmo2-8B and CountGD, run on a GPU server (a local CountGD run on the legacy crops is in `cnn/countgd/`) |
| `camera_aug/` | PickGlass-Stress: phone-camera simulator, profiles, label audit, results |
| `genai_sr/` | pre-registered test of generative enhancement (protocol, server scripts, results) |
| `aug1016/` | label propagation to the notebook augmentations (CNN training) |
| `paper/` | `tools/` (the table and figure generators, scorer, human-count analysis runner, generic-IQA analysis, re-shoot audit; `tools/reference/`: the analyses against the reference annotator and the other scripts that need both annotators' counts) and `tables/` (their outputs) |
| `data/` | manifest, scene map, window polygons, photo provenance, legacy hand counts, all predictions (no pixels) |
| `tools/` | `count_photos.py` (new photos -> counts), `verify_checksums.py`, `make_readme_figure.py` |
| `freelancer/` | `same_patch_pairs.json` (label-free list of crops showing the same fabric patch) and its README |
| `docs/` | [problem definition](docs/problem_definition.md), [annotation protocol](docs/annotation_protocol.md), [annotator guide](docs/annotator_guide.md), [counting guide](docs/counting_guide.md), [how to extend](docs/how_to_extend.md), [open problems](docs/open_problems.md) |

## Quickstart

```bash
git clone https://github.com/anoubhav/pickglass-tpi && cd pickglass-tpi
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

python3 tools/verify_checksums.py               # the frozen files are the published ones
python3 tools/make_readme_figure.py             # the figure above: the frozen counter on a synthetic fabric -> scratch/
python3 annotation_tool/server.py --demo --open # the annotation tool on three synthetic practice fabrics
python3 tools/count_photos.py my_photos/*.jpg --out counts.csv   # your own pick-glass photos (ruler -> crop -> counts)
cd paper && python3 tools/scenes76_labelfree.py # re-derive the label-free per-scene numbers from the released CSVs
```

Scoring human counts (the annotators' counts once released, or your own annotations of the same crops). `tools/score150.py` scores every method against the mean of the exports it is given: pass the reference annotator's export alone to score against A2, as in the paper's main table; pass both exports, as below, to score against their mean, the reference fixed in the analysis plan (supplement Tab. S6; per-file scores: Sec. S17):

```bash
cd paper
python3 tools/score150.py --labels ../annotations_A1.csv --labels ../annotations_A2.csv --out ../scratch/tables \
  --fig --fig-dir ../scratch/tables --groups tables/fabric_groups_v2.csv \
  --extra-wide ../methods/results/combined_v1_150_wide.csv::combined-v1 \
  --extra-wide ../gpu_results/countgd/outputs/results/rect150_predictions_wide.csv::countgd,countgd@CG-W-ex3-lat \
  --extra-wide ../gpu_results/scored/molmo2_rect150_predictions_wide.csv::molmo2_8b_bf16,molmo2_8b_bf16@P4 \
  --extra-wide ../ft_results/results/rect150_predictions_wide.csv::molmo2_4b_synthft@P1,molmo2_4b_synthft@P4s \
  --extra-wide ../vlm/scaffold/results/rect150_S3_predictions_wide.csv::gpt-6.1-sol@S3,gemini-3.8-flash@S3,qwen3.8-27b@S3,s4-peaks \
  --extra-wide tables/src/extra150_wide.csv::pg-lattice@ruler,cascade-gate \
  --extra-wide ../cnn/runs_aug/report/real150_aug_wide_heldout.csv::cnn-aug-V4-b0,cnn-aug-V5-b0,cnn-aug-V2-r18
```

Always pass `--out`: the default overwrites `paper/tables/`. The paper scores only photos whose two guide lines were counted over at least 75% of the window; `tools/label_day.sh` applies that rule and writes the filtered exports to `work/merged/scoring_headline/` of its run folder.

## Reproduce the paper

Run `paper/tools/*` from `paper/`, everything else from the repository root. Numbers follow the paper and its supplement as under review. **Needs**: *repo* = runs from the files in this repository; *photos* = needs the 150 photos or the 13 legacy crops (released after review; `common/paths.py`); *sheet* = the legacy hand-count sheet (`$PICKGLASS_LEGACY_ROOT/Warp-Weft count.xlsx`; the counts themselves are in `data/legacy13_hand_counts.csv`); *labels (A2)* = the reference annotator's export; *labels (A1)* = the non-expert annotator's export; *labels (A1, A2)* = both annotators' exports (released after review; each filtered by the 75% rule of *Scoring human counts*); *GPU* / *API* = a CUDA machine or `OPENROUTER_API_KEY`.

*reference/* means [`paper/tools/reference/`](paper/tools/reference/README.md): the generators of the numbers scored against A2 and of the other analyses that need both annotators' counts, with their own README. Several of the paper's tables were edited after generation (row names, bold, caption wording): re-running a generator reproduces the numbers, not always the exact wording. Scored against A2 also means scored on A2's counts propagated to the copies of each scene: `python3 annotation_tool/propagate_scene_labels.py <A2 export> --out-dir <d>` writes `<d>/labels_per_scene.csv`, the input of `methods/octave/eval_octave_v1.py` below. The reference/ scripts read per-scene counts (`data/human_counts/labels_per_scene_A1.csv` and `_A2.csv`, as released, or the `labels_per_scene.csv` written by `annotation_tool/propagate_scene_labels.py <export> --out-dir <d>`).

### Main paper: figures and tables

| Item | Content | Command | Needs |
|---|---|---|---|
| Fig. 1 | factory photos and the task: (a) ruler ticks locate the window's corner on #118, (b) the rectified window, (c) PG-Lattice's dots, (d) part of the H line of #39 with A2's ticks (whole line 56) and PG-Lattice's (58) | no script for the printed layout in this repository; an earlier one-column layout of the same panels: `python3 tools/make_fig_dataset_pipeline.py --oct-scene 39 --oct-labels <A2's merged export CSV>` | photos, labels (A2) |
| Fig. 2 | the annotation tool on scene #118 with A2's dots | a headless-Chrome screenshot of `python3 annotation_tool/server.py --data-dir <folder with the manifest and crop #118> --saved-dir <folder with A2's export>` (the screenshot script is not in this repository) | photos, labels (A2) |
| Tab. 1 | every method and A1's counts on the 76 scenes against A2: MAPE [95% CI], counts within 5% and 10%, scenes with both counts within 5%; leave-one-out mean of A2's counts; the row GPT-6.1-sol, insets (GPT-6.1-sol given four magnified 0.1-inch insets of the window, the input the scaffold study pre-declared; counts exported before any human count, row added post hoc) | reference/ `make_tab1_ref.py` (the whole table, including the insets row, with the statistics of the frozen `tools/score150.py`; `main150_rows_refB.tex` has the printed row names, the bold count columns and the shading of our two methods). Most machine rows (the insets row: `gpt-6.1-sol@S3`) and the leave-one-out mean also come from `python3 tools/score150.py --labels <A2 export> ...` (options as in *Scoring human counts*); the MAPE and both-within-5% count of the CNN trained on A1's counts and of the CNN–VLM ensemble also from `methods/octave/eval_octave_v1.py` (Sec. 5.1 below) | labels (A1, A2) |
| Fig. 3 | half or double: every count of PG-Lattice, GPT-6.1-sol P1 and A1 divided by A2's count on the same line (152 lines) | reference/ `make_fig_octave_hist.py` (the figure and its tallies) | labels (A1, A2) |
| Tab. 2 | re-shoot repeatability over the 41 same-patch pairs (median worse-axis difference, pairs within 5%, pairs about twofold apart) for A2, A1, PG-Lattice, Molmo2-4B + LoRA, GPT-6.1-sol P1 and the CNN–VLM ensemble; MAPE against A2 | reference/ `reshoot_consistency.py` (every row but Molmo2-4B; `--machines-only` gives the machine rows without labels, from `tables/fabric_groups_v2.json` and released predictions) and reference/ `molmo_reshoot.py` (the Molmo2-4B row, label-free); MAPE column: Tab. 1 | labels (A1, A2) for the annotator rows; machine and Molmo2-4B rows: repo |
| Tab. 3 | capture stress without labels: share of the 76 scenes whose count stays within 5% of the clean count under each degradation (PG-Lattice, FFT peak) | `python3 tools/scenes76_labelfree.py` (the numbers: `tables/quality76.csv`; degraded runs, from the repository root: `python3 methods/ablate.py --set all150 --est ours,lattice,fft,profile,tiles --degrade all --out methods/results/all150_degrade.csv`, which needs the photos and overwrites a released file: run it in a branch); LaTeX rows: typeset from `tables/quality76.csv` (no script in this repository) | repo (degraded runs: photos) |
| Fig. 4 | legacy-13 MAPE under five degradations (PG-Lattice, log-Gabor bank, synthetic CNN, GPT-6.1-sol P1, leave-one-out mean), byte-identical pixels for every method | `python3 tools/degradation_table.py` (writes `tables/degradation_legacy13.csv`, `tables/degradation_legacy13_per_image.csv` and `figs/degradation_curves.pdf`, the same curves in an earlier layout; `--fig-only` redraws the figure from the table; inputs: `methods/degrade_vlmcrops.py`, `vlm/run_degrade_vlm.py`, `cnn/degrade_vlmcrops.py`); the script of the printed layout is not in this repository | repo (inputs: photos, API) |

### Main paper: numbers in the text

| Item | Content | Command | Needs |
|---|---|---|---|
| Sec. 3 | provenance audit (76 scenes, 44 patches, 41 re-shoot pairs); annotation time and same-photo agreement of A1 and A2 (median difference 32.0%); re-shoot repeatability (Tab. 2) | `python3 camera_aug/provenance_scan.py`; `python3 tools/scenes76_labelfree.py`; `python3 tools/fabric_groups_v2.py` (re-shoot audit -> `tables/fabric_groups_v2.csv`); agreement and time: Tab. S2; re-shoots: Tab. 2 | photos (audit); labels (A1, A2) |
| Sec. 4.2, Sec. S17 | combined counter combined-v1 (34.4 vs 34.8% against A2), thresholds fitted on synthetic data only, then its held-out synthetic test | `python3 methods/combined_v1.py --predict` (frozen rule on the frozen counter's outputs); `python3 methods/combined_v1_eval.py --synth` (fit and held-out renders: `methods/combined_v1.py --fit`, `--heldout-make`, `--heldout-score`); against A2: reference/ `make_tab1_ref.py` | repo (renders: minutes to hours of CPU); labels (A1, A2) for the numbers against A2 |
| Sec. 5.1 | paired tests against A2 (Molmo2-4B + LoRA vs GPT-6.1-sol P1, vs A1 and vs the leave-one-out mean; GPT-6.1-sol and PG-Lattice vs the leave-one-out mean; on MAPE and on the symmetric error) and their patch-clustered CIs (Sec. S3); GPT-6.1-sol given four magnified insets against Molmo2-4B + LoRA, GPT-6.1-sol P1, the leave-one-out mean and A1 (post hoc; also Sec. S6); methods compared with A1 as a human baseline (GPT-6.1-sol P1 −10.2 points; the CNN–VLM ensemble −11.9 and PG-Lattice −2.5 are in supplement Sec. S3); blind test of the CNN–VLM ensemble (25.4%, −1.7 points [−4.2, +0.7] vs GPT-6.1-sol P1) and the CNN trained on A1's counts (28.1%); weave κ; numbers against the mean of both and against A1 | reference/ `paired_ref.py`, `human_floor_paired.py`, `logratio_endpoint.py`; the insets: reference/ `scaffold_insets_ref.py` (`scaffold_insets_ref.json`); blind tests: `python3 methods/octave/eval_octave_v1.py --labels <d>/labels_per_scene.csv --names A2 --reference each --pred methods/ensemble/predictions_ens_v1_150.csv --out-json ens_v1.json --out-txt ens_v1.txt` (the CNN: `--pred methods/ensemble/cnn_fabric_oof_FLSA_150.csv`); weave κ: Tab. S2; against the mean: Tab. S6; against A1: Tab. S7 | labels (A2 for the blind tests; A1, A2 otherwise) |
| Sec. 5.2 | half-or-double tallies against A2 (PG-Lattice 36 of 152 lines, GPT-6.1-sol P1 37, A1 48) and the octave-oracle bound (34.8 -> 10.8%); the pre-registered rule 4 (15 and 28 of 64 lines); dot-spacing CVs; the octave check's blind test (−1.0 points [−2.2, −0.1]) | reference/ `paired_ref.py` (tallies and bound; the same tallies with Fig. 3: `make_fig_octave_hist.py`); rule 4: reference/ `rule4_band.py`; the dot-spacing CVs (0.16 / 0.19) were computed from the dot positions in the exports and have no script in this repository; octave check: `python3 methods/octave/eval_octave_v1.py --labels <d>/labels_per_scene.csv --names A2 --reference each --pred methods/octave/predictions_octave_150.csv --out-json octave_v1.json --out-txt octave_v1.txt`; learned selector: Sec. S16 | labels (A1, A2) |
| Sec. 5.3 | re-shoot numbers (Tab. 2), and the 8 of PG-Lattice's 24 repeating pairs on which both photos read about half or double A2 | Tab. 2 commands (the 8-of-24 tally against A2 has no script in this repository) | labels (A1, A2) |
| Sec. 5.4 | generic no-reference IQA against PG-Lattice's error on A2's counts (Spearman ρ for BRISQUE, NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR, η and the confidence); the frozen gate (its paired difference against A2, −0.3 points [−3.1, +2.3], is printed in Sec. S10); scenes where A1 and A2 differ by more than 10% | ρ: `python3 tools/iqa150.py --labels <A2 export> --out ../scratch/iqa_A2` (the frozen 76-scene test, run on A2's counts); gate: reference/ `expert_iqa.py`; disagreement AUROCs: reference/ `branchB_numbers.py` and `iqa_disagreement_mwu.py`; legacy crops: Tab. S3 | labels (A2; A1, A2 for the disagreement test) |
| Sec. 5.5 | capture stress (Tab. 3, Fig. 4); px per yarn (Sec. S12); the ruler on 75 of 76 scenes (0.19 mm, 92% vs 42% within 5%); the ruler crop against A2 (35.3 -> 37.9%) | Tab. 3 and Fig. 4 commands; `python3 camera_aug/label_audit.py summarise`; `python3 methods/ruler_eval.py calib` then `python3 methods/ruler_summary.py` and `python3 tools/scenes76_labelfree.py` (the 76-scene ruler numbers); against A2: reference/ `ruler_vs_ours_ref.py` (the `pg-lattice@ruler` predictions are in `tables/src/extra150_wide.csv`) | repo (runs: photos); labels (A2) for the last number |
| Sec. 5.6 | pre-registered generative-enhancement study (27 generative comparisons) and the native-resolution guard | Tab. S5 (study); Sec. S14 (guard) | repo (restorers: GPU, API) |

### Supplement: tables and figures

| Item | Content | Command | Needs |
|---|---|---|---|
| Tab. S1 | legacy test under one crop rule: the main rows (every prompt, MAE and the paired tests: `tables/legacy13_unified.csv`, `tables/legacy13_paired.csv` and the extended supplement) | `python3 tools/legacy_unified.py` | sheet |
| Tab. S2 | label checks and agreement of A1 and A2 (screening answers, MAE, Bland–Altman, ICC(2,1), median difference with and without a badge, weave κ, PG-Lattice without the guide scenes #27/65/93/110, annotation time) | `tools/label_day.sh --out-root ../scratch/ld --protocol dual <A1 exports> <A2 exports>` (`tables/supp_labelcheck_rows.tex`; needs the LaTeX sources, see [paper/README.md](paper/README.md)) | labels (A1, A2) |
| Tab. S3 | which quality signal predicts counting error: generic no-reference IQA (BRISQUE, NIQE, MUSIQ, CLIP-IQA, TOPIQ-NR) vs the counter's confidence, the resolvability and the gate | `python3 tools/iqa_generic.py` (per-image scores: `python3 tools/iqa_scores.py`, pyiqa) | repo (scores: photos + pyiqa) |
| Tab. S4 | PG-Lattice's count change against px per yarn, by route to low resolution | `python3 camera_aug/label_audit.py summarise` (`camera_aug/results/label_audit_sweep.csv`) | repo (sweep: photos) |
| Tab. S5 | enhancement before counting at ×8 down-sampling | `python3 genai_sr/make_paper_table.py` (after `python3 genai_sr/analyze.py`; restorers: `genai_sr/server/`; protocol: [genai_sr/README.md](genai_sr/README.md)) | repo (restorers: GPU, API) |
| Tab. S6 | all methods against the mean of A1 and A2, the reference fixed before labeling (MAE, MAPE, counts within 5% and 10%, both counts within 5%, weave) | `python3 tools/score150.py --labels <A1 export> --labels <A2 export> ...` (*Scoring human counts*: `main150.csv`), or `tools/label_day.sh --protocol dual` (`tables/main150_rows.tex`, same numbers; full command in the Tab. S2 row); final layout: edited by hand | labels (A1, A2) |
| Tab. S7 | pre-registered sensitivity analyses: every method against A1, A2, the mean, the mean without the disputed lines, and the log-ratio endpoint; its last block, added post hoc: the untuned counting specialists, Molmo2-4B + LoRA and GPT-6.1-sol given the four insets | A1, A2 and mean columns: `python3 tools/score150.py --labels <export(s)> ...` (A2 column also reference/ `make_tab1_ref.py`); log-ratio columns: reference/ `logratio_endpoint.py` (PG-Lattice, the octave check, GPT-6.1-sol P1, the ensemble and its CNN, and each annotator against the other) and, against A2 for every Tab. 1 row, `tab1_ref_rows.csv` of reference/ `make_tab1_ref.py`; the column without the disputed lines and the assembled table: no script in this repository; some CNN rows also need cross-validation prediction files that are not in this repository | labels (A1, A2) |
| Tab. S8 | the 72 lines (50 scenes) on which A1 and A2 differ by more than a factor of 1.41 | reference/ `make_disputed_table.py` | labels (A1, A2) |
| Fig. S1 | PG-Lattice on legacy a1, r6 and PickGlass #86 | `python3 tools/make_supp_figs.py` | photos |

### Supplement: sections

Numbers in each section that do not come from one of its tables or figures above:

| Item | Content | Command | Needs |
|---|---|---|---|
| Sec. S1 | task definition and protocol: bias against the mean, CNN fine-tuned with folds grouped by patch, the label-free invariance check on the augmented copies, the ensemble and the control ensembles against the mean (GPT-6.1-sol P1 combined with a second method), blind tests of the ensemble's CNN and of ens-v1b against A2, the Tab. S7 summary, rule 1's histogram | bias: `main150.csv` of `tools/score150.py` against the mean; invariance check: `python3 aug1016/invariance_check.py` (`aug1016/invariance/summary.md`); blind tests: `methods/octave/eval_octave_v1.py` as in Sec. 5.1 with `--pred methods/ensemble/cnn_fabric_oof_FLSA_150.csv` or `--pred methods/ensemble_aug/predictions_ens_v1b_150.csv` (ens-v1b against ens-v1: add `--frozen methods/ensemble/predictions_ens_v1_150.csv`); control ensembles: no script in this repository (per-axis geometric means of released prediction files, scored against the mean with `tools/score150.py --extra-wide`); the Tab. S7 summary: as Tab. S7; rule 1's histogram: reference/ `rule1_histogram.py` | repo; photos (invariance check); labels (A1, A2) |
| Sec. S2 | legacy crops: the crop rule, paired comparisons, counting conventions per count, VLM repeat spread | `python3 tables/src/make_crop_rule_check.py`; `python3 tools/make_supp_tables.py` (from `tables/legacy13_crop_sensitivity.csv`); `python3 tools/legacy_unified.py` (`tables/legacy13_paired.csv`, `tables/legacy13_unified_per_count.csv`); repeats: `python3 vlm/evaluate_vlm.py` (`vlm/results/stability_P1_legacy.csv`) | photos (crop check); sheet; repo |
| Sec. S3 | annotation protocol, agreement, re-shoots per annotator (pairs within 3, 5 and 10%), badge split, methods against the human noise floor, patch-clustered CIs | Tab. S2; reference/ `reshoot_consistency.py`, `human_floor_paired.py`, `logratio_endpoint.py`, `paired_ref.py` (re-shoots by resolution: reference/ `branchB_numbers.py`; the badge split, Fisher p = 0.79 and Mann-Whitney p = 0.002, has no script in this repository) | labels (A1, A2) |
| Sec. S4 | PG-Lattice: scoring function, fusion, confidence, examples | the method: `methods/results/counting_frozen_final.py`; Fig. S1 | photos (figure) |
| Sec. S5 | synthetic data and realism statistics | `python3 cnn/realism_stats.py` (renderer: `cnn/synth.py`) | photos |
| Sec. S6 | VLM prompts; up-sampling test against A2 on the 47 full-resolution scenes with a same-exposure quarter-width copy; measurement scaffolds (in full: the extended supplement) and GPT-6.1-sol given the four magnified insets against A2 (24.9%; paired tests, post hoc) | prompts: `vlm/prompts.py`; up-sampling test: no script in this repository; scaffolds: `vlm/run_scaffolds.py`, `vlm/evaluate_scaffolds.py` (on PickGlass: the `@S3` rows of *Scoring human counts*); the insets against A2: reference/ `make_tab1_ref.py` (24.9%) and `scaffold_insets_ref.py` (paired tests) | repo; labels (A2) for the up-sampling test; labels (A1, A2) for the insets |
| Sec. S7 | counting specialists; Molmo2-4B + synthetic LoRA: synthetic validation, legacy and PickGlass numbers | `python3 gpu_results/score_gpu_baselines.py` (specialists); `python3 ft_results/code/make_ft_tables.py` (LoRA; training `ft_results/code/train_lora.py`, inference `ft_results/code/infer_ft.py`); legacy pairs: `tables/legacy13_paired.csv`; against A2: reference/ `make_tab1_ref.py` (25.2%) and `paired_ref.py` (vs GPT-6.1-sol P1 and vs A1); the A1-reference numbers (47.6%, +5.9 vs PG-Lattice) and the −9.6 difference to PG-Lattice against A2: no script in this repository; against the mean: the `molmo2_4b_synthft@*` rows of *Scoring human counts* | repo (training, inference: GPU); labels (A1, A2) for PickGlass |
| Sec. S8 | capture quality: legacy conditions, strata against the mean, the frozen 76-scene IQA test against the mean and against A2, disagreement and failure AUROCs, subsets | the frozen test against the mean: `python3 tools/iqa150.py --labels <A1 export> --labels <A2 export> --out ../scratch/iqa_mean` (against A2: Sec. 5.4); MAPE by resolution and annotator flag: the `strat150.csv` of `tools/score150.py`; the other strata against the mean: `tools/label_day.sh --protocol dual` (strata from `tables/strata150.csv`, made by `python3 tools/strata150.py`); disagreement AUROCs: reference/ `branchB_numbers.py`, `iqa_disagreement_mwu.py` | repo (strata150.py: photos); labels (A1, A2) |
| Sec. S9 | window and scale from the glass's ruler; glass constants and the loupe's pincushion | method `methods/ruler_calib.py`; `python3 methods/ruler_eval.py calib` (and `count`, `perturb`, `photodeg`, `blur`); `python3 methods/ruler_summary.py`; pincushion k1: `python3 camera_aug/measure_distortion.py` (`camera_aug/results/distortion_measured.csv`) | photos; repo (summary) |
| Sec. S10 | quality-aware cascade: calibration and the legacy evaluation (every policy and the threshold sweep: the extended supplement); the frozen gate on PickGlass | `python3 tools/cascade_tradeoff.py --eval` (never `--fit`; it rewrites the frozen `tables/cascade_150_decisions.csv` byte-identically, check with `tools/verify_checksums.py`; inputs `methods/cascade_synth.py`; policies and sweep: `tables/cascade_legacy13.csv`, `tables/cascade_sweep.csv`); the gate's decisions: `tables/cascade_150_decisions.csv` (frozen), scored as the `cascade-gate` row of *Scoring human counts* | repo; labels (A1, A2) |
| Sec. S11 | provenance of the 150 files (file kinds), 76 scenes and 41 re-shoot pairs; rotation and label propagation; the controlled full/quarter resolution pairs | `python3 camera_aug/provenance_scan.py`; `python3 tools/scenes76_labelfree.py` (resolution pairs: key `pairs` of `tables/scenes76_labelfree.json`); `python3 tools/fabric_groups_v2.py`; propagation: `annotation_tool/propagate_scene_labels.py` | photos; repo |
| Sec. S12 | px per yarn (Tab. S4) | `python3 camera_aug/label_audit.py summarise` (simulator: `camera_aug/camera_sim.py`) | repo (sweep: photos) |
| Sec. S13 | generative enhancement and hallucinated threads (Tab. S5) | `python3 genai_sr/analyze.py`; `python3 genai_sr/make_paper_table.py` | repo; GPU, API (restorers) |
| Sec. S14 | the native-resolution guard | `python3 methods/guard/analyze_real.py`, `python3 methods/guard/posthoc_supplement.py --tables` (`methods/guard/results/real_table_cascade.csv`; tuning: `methods/guard/tune_synth.py`, synthetic only) | repo |
| Sec. S15 | pseudo-labels for the CNN | `python3 cnn/score_aug.py` (protocol `cnn/runs_aug/PREREG.json`; training `cnn/train_aug.py`) | sheet; repo (training: GPU) |
| Sec. S16 | half-or-double readings: halvings and doublings against A2 and the lines the octave oracle changes, spectral analysis against A1, rule 4, the octave check (octave-v1) and the learned octave selector (octsel-v1), their blind tests against A2 | against A2: reference/ `paired_ref.py`; blind tests: `python3 methods/octave/eval_octave_v1.py --labels <d>/labels_per_scene.csv --names A2 --reference each --pred methods/octave_learned/predictions_octsel_oof_FLSA_150.csv ...` (octave-v1: `methods/octave/predictions_octave_150.csv`; secondary tests: `--frozen methods/ensemble/predictions_ens_v1_150.csv`, and `predictions_octsel_argmax_oof_FLSA_150.csv`, `predictions_octsel_full_150.csv`, `predictions_octsel_geo_gpt_oof_FLSA_150.csv`, `predictions_octsel_geo_ens_v1_oof_FLSA_150.csv` in `methods/octave_learned/`); methods: `methods/octave/octave_v1.py`, `methods/octave_learned/octsel_v1.py`; rule 4: reference/ `rule4_band.py`; the spectral analysis against A1: no script in this repository | labels (A2 for the blind tests; A1, A2 otherwise) |
| Sec. S17 | all methods against the mean (Tab. S6) and per file, the s4-peaks baseline, the combined counter's tests | Tab. S6 (`s4-peaks` and `combined-v1` rows); s4-peaks against A2: its row of `python3 tools/score150.py --labels <A2 export> ...` (its paired tests: no script in this repository); per-file scores: `main150_perfile.csv` of *Scoring human counts*; held-out synthetic test: `python3 methods/combined_v1_eval.py --synth` (`methods/results/combined_v1_eval_synth.json`) | labels (A1, A2); repo |

### Extended supplement

The supplement refers to an extended supplement, to be added to this repository, for what it no longer prints (among it every legacy method and prompt with the paired tests, every cascade policy, every CNN variant with the pre-registered augmentation ablation) and for further studies. The extended supplement document is not part of this repository yet; the generators of its tables, figures and studies are here:

| Study | Command | Needs |
|---|---|---|
| thread-count test methods and product tolerances | hand-written (summary in [docs/problem_definition.md](docs/problem_definition.md)) | - |
| crop-rule sensitivity on the 13 legacy windows | `python3 tools/make_supp_tables.py` (from `tables/legacy13_crop_sensitivity.csv`; crop check: `python3 tables/src/make_crop_rule_check.py`) | repo (crop check: photos) |
| every legacy method and prompt, with the paired tests | `python3 tools/legacy_unified.py` (`tables/legacy13_unified.csv`, `tables/legacy13_paired.csv`) | sheet |
| comparison of PG-Lattice's dots with the annotators' dots (post hoc) | as Tab. S2 | labels (A1, A2) |
| capture degradations on the 13 legacy windows | `python3 tools/make_supp_tables.py` (motion, noise and shadow runs: `methods/degrade_vlmcrops_extra.py`) | repo |
| window from the glass's ruler on the 150 photos (table); window and scale from the glass's ruler (figure) | `python3 methods/ruler_summary.py` (inputs: `python3 methods/ruler_eval.py calib`, and `count`, `perturb`, `photodeg`, `blur`); `python3 tools/make_ruler_fig.py` | repo + scikit-learn (inputs: photos); photos (figure) |
| every cascade policy and the threshold sweep | as Sec. S10 | repo |
| the native-resolution guard on the degraded legacy crops (table) | as Sec. S14 | repo |
| secondary per-file scores (table) | `main150_perfile.csv` of *Scoring human counts* | labels (A1, A2) |
| control ensembles (table) | as Sec. S1 | labels (A1, A2) |
| error-versus-discard curves against the mean of A1 and A2 | no script in this repository | labels (A1, A2) |
| enhancement of legacy crops down-sampled ×8 (figure) | `python3 genai_sr/make_figure.py` (restorer outputs: `genai_sr/server/`) | photos (restorers: GPU, API) |
| legacy-13 MAPE of every final CNN variant | `python3 cnn/score_aug.py` (training: `cnn/train_aug.py`, `cnn/train_cv.py`) | sheet (training: GPU) |
| every count of PG-Lattice and GPT-6.1-sol P1 against A1's count | `python3 tools/make_fig_octave.py <work/stage/main150_per_scene.csv of a tools/label_day.sh run on A1's export>` (an earlier layout) | labels (A1) |
| VLM error by model and prompt | `python3 vlm/make_summary_plot.py` (`vlm/evaluate_vlm.py` scores `vlm/raw/`) | repo |
| VLM measurement scaffolds | `python3 vlm/make_scaffold_figs.py` (runs: `vlm/run_scaffolds.py`, `vlm/evaluate_scaffolds.py`) | photos (runs: API) |
| PickGlass-Stress on legacy-13; camera profiles | `python3 camera_aug/score_stress.py` then `python3 tools/make_camera_stress_supp.py`; `python3 camera_aug/make_camera_fig.py` (renders: `camera_aug/build_stress.py`) | repo; photos |
| window segmentation (legacy pipeline / notebook classical segmentation / centred box) | `python3 methods/legacy_2019/run_pipeline.py --out run1` (legacy pipeline), `python3 methods/legacy_2019/repro_150.py` (notebook classical segmentation, `seg_final.py`) | photos |
| label-exact augmentation: verification and minimum-resolution sweep | `python3 cnn/augment_optics_verify.py` (augmentation: `cnn/augment_optics.py`) | photos for the sweep |
| degradation examples; drift; confidence vs drift | `python3 methods/make_figs.py all` (inputs: `methods/ablate.py`) | photos for the examples; repo |
| Molmo2 pointing, CountGD; synthetic-pointing LoRA figures | `python3 gpu_results/make_paper_figs.py` (runs: `gpu_results/code/`, `gpu_results/countgd/code/`); `python3 ft_results/code/make_figs_ft.py` | photos (runs: GPU) |
| dataset figure; pipeline strip | `python3 tools/make_dataset_fig.py --rows raw,quarter`; `python3 tools/make_fig_pipeline.py --variant strip` | photos |
| tolerance-referenced legacy metrics (post hoc) | `python3 tools/tolerance_legacy13.py` | repo |

Predictions (all frozen in this repository): PG-Lattice `python3 methods/run_ours_150.py`, legacy `python3 methods/run_legacy_vlmcrops.py`; API VLMs `python3 vlm/openrouter_client.py --set rect150 --prompts P1` (`OPENROUTER_API_KEY`; `python3 vlm/evaluate_vlm.py` then writes `vlm/results/rect150_P1_predictions_wide.csv`; local MLX Molmo2-8B: `python3 vlm/run_vlm.py --model molmo2_8b --set rect150`); Molmo2-4B + LoRA `ft_results/code/infer_ft.py` (GPU); CNN `python3 cnn/evaluate.py --ckpt <checkpoint> --out <dir> --real150` (the checkpoint is not in git; [THIRD_PARTY.md](THIRD_PARTY.md)); ruler `python3 methods/ruler_eval.py calib`; combined-v1 `python3 methods/combined_v1.py --predict`; extra wide files `cd paper && python3 tools/make_extra150_wide.py`. Re-running them overwrites the frozen predictions: do it in a branch.

## Tests

Without the photos: `node annotation_tool/tests/test_core.js` (38 of 39 pass; the manifest check that every crop file exists needs the photo release), `node annotation_tool/tests/test_scenes.js` (11), `node annotation_tool/tests/test_attention.js` (9), `python3 annotation_tool/tests/test_scenes.py` (7), `python3 annotation_tool/tests/test_attention.py` (8). `annotation_tool/tests/test_examples.*` and `aug1016/tests/test_propagate.py` need the photo release (the Examples panel and the notebook augmentations). [SCAN_REPORT.md](SCAN_REPORT.md) is a snapshot of the secret and personal-data scan made before the first publication, including the tables that were then regenerated byte-identically from the repository.

## Data

`data/` holds everything that has no pixels: `manifest.json` (150 files, window polygons, scene fields, machine proposals), `scene_map.csv`, `window_polygons.csv`, `photo_provenance.json` (sizes and MD5 of the photos), `legacy13_hand_counts.csv` and `predictions/` (every method on PickGlass and legacy-13, with `INDEX.csv`). See [data/README.md](data/README.md).

**The photos, rectified crops, the 13 legacy crops, degraded copies, PickGlass-Stress renders, generative-enhancement outputs and the human counts with dot positions will be released after review**, on an archive with a DOI ([DATASHEET.md](DATASHEET.md), [DATA_LICENSE](DATA_LICENSE)). The code expects them where `common/paths.py` says (environment variables with repo-relative defaults): `PICKGLASS_NOTEBOOKS` (photos and polygons), `PICKGLASS_LEGACY_ROOT` (legacy material), `PICKGLASS_SCRATCH`, `PICKGLASS_GPU_HOST`, `PICKGLASS_REMOTE_ROOT`, `OPENROUTER_API_KEY`. Run `python3 common/paths.py` to see what is found.

## Freeze and checksums

[CHECKSUMS.md](CHECKSUMS.md) lists the SHA-256 of the frozen counter, the scorer, the frozen cascade decisions, the combined counter and its pre-registration, the 76-scene generic-IQA test, the analysis of the human counts (`paper/tools/label_day.py`) and the other pre-registration records in this repository, with the MD5 of every file released byte-identical (the paper's supplement discloses the MD5 of the counter, the scorer and the cascade decisions). The analysis plan for the reference annotator's counts, its addendum and the recipe of a second ensemble variant are not in this repository; they are available on request, and CHECKSUMS.md lists their SHA-256 and MD5. `SHA256SUMS` covers every file except itself, `EXPORT_PROVENANCE.json` and `SCAN_REPORT.md`. `python3 tools/verify_checksums.py` checks the frozen files against `tools/freeze_checksums.json` and every file against `SHA256SUMS`. The section *Commitments* of CHECKSUMS.md lists every file a pre-registration record commits to by SHA-256, including the few that carry per-scene human counts and will be released with both annotators' counts. The ensembles and the learned octave selector cannot be re-run from this repository until then (they read those files); their frozen predictions are here.

## Documentation

- [Problem definition](docs/problem_definition.md): task, measurand, metrics, protocol, leakage rules.
- [Annotation protocol](docs/annotation_protocol.md): the tool, the completeness rule, the screening test and pass rule, the two-annotator protocol (each annotator counts every scene from scratch; agreement; A2, the textile-trained annotator, is the reference and A1 the non-expert human baseline, with the pre-registered mean of both reported in the supplement).
- [Annotator guide](docs/annotator_guide.md) and [counting guide](docs/counting_guide.md) (synthetic pictures only).
- [How to extend](docs/how_to_extend.md): new photos -> window polygon or ruler -> rectify -> annotate -> propagate -> score.
- [Open problems](docs/open_problems.md): glass calibration, the human noise floor beyond two annotators, more mills / glasses / phones, defocus, a multi-window protocol.
- [MODEL_CARD.md](MODEL_CARD.md) (screening, not certification), [DATASHEET.md](DATASHEET.md), [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

```bibtex
@inproceedings{pickglass2027,
  title     = {Image Quality and the Reliability of Fabric-Density Measurement: Counting Threads from Smartphone Photos Through a Pick Glass},
  author    = {Agarwaal, Anoubhav},
  booktitle = {WACV 2027 Workshop on Image/Video/Audio Quality Assessment in Computer Vision, VLM and Diffusion Model (WVAQ)},
  year      = {2027},
  note      = {Under review}
}
```

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). Data: see [DATA_LICENSE](DATA_LICENSE), which defines which files are data and the terms that apply to them.

## Acknowledgements

The legacy baseline, the 13 legacy crops and their hand counts come from an earlier Textile Quality Analysis project of the Computer Vision and Intelligence club, IIT Madras; `methods/legacy_2019/` is a line-by-line re-implementation of its counting step.

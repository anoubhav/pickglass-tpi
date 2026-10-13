# ens-v2 recipe, fixed before the second annotator's (FL_NG) counts are opened

Written Mon 12 Oct 2026 by the coordinating Claude session. FL_NG's export not yet delivered. The SHA-256 of this file is recorded in paper/CHANGELOG.md ("ens-v2 recipe"). This is a RECIPE run on the FINAL labels; its numbers are a cross-validated estimate, not a blind test.

## Recipe (binding)
1. Reference for training and scoring: the final per-scene labels as fixed by `handover/results/PREREG_labelday_FLNG_20261012.md` and addendum 1 (mean of the two annotators by default; if the interpretation rule switches to an adjudicated or per-annotator reference, use that same reference here).
2. CNN: `cnn/train_cv.py` with label_day's base settings (ResNet-18, 15 epochs, tiles 16, batch 32), 5-fold cross-validation grouped by the 44 fabric groups (`paper/tables/fabric_groups_v2.csv` as the labels' `group` column), experiments `imagenet_ft` and `synth_ft` (synthetic checkpoint `cnn/runs/synth_r18_v2mix_aniso/model.pt`, md5 dad61b74...), seeds 0, 1, 2 (6 runs).
3. CNN ensemble: per file and axis, the geometric mean of the 6 out-of-fold predictions.
4. ens-v2: per file and axis, the geometric mean of the CNN ensemble and GPT-6.1-sol P1 (frozen predictions, `vlm/results/rect150_P1_predictions_wide.csv`, h = `gpt-6.1-sol__v`, v = `gpt-6.1-sol__h`); GPT missing -> CNN ensemble.
5. Scoring: `methods/octave/eval_octave_v1.py --pred <ens-v2 predictions in (id, h_count, v_count) format>` against the final reference; report MAPE, both axes within 5%, paired difference vs GPT-6.1-sol P1 and vs ens-v1, scene bootstrap 10,000, seed 20261012; fabric-clustered CIs as a secondary.
6. Report ens-v2 next to ens-v1 (whose blind FL_NG test is primary for "does combining help"), labelled "cross-validated on the final labels; recipe fixed before the second annotator's counts".

## Not allowed after opening FL_NG
Changing backbones, seeds, epochs, grouping, the combination rule or the reference rule for ens-v2; adding other members.

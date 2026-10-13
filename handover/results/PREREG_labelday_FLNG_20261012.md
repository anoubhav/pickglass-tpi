# Analysis plan for the second annotator (FL_NG), fixed before FL_NG's counts are opened

Written Mon 12 Oct 2026, about 12:30 EDT (16:30 UTC), by the coordinating Claude session. FL_NG's export has not been delivered or opened. FL_SA's counts have been seen (dry runs, diagnosis in `labelday_diag_FLSA_20261012/`). The SHA-256 of this file is recorded in `paper/CHANGELOG.md` ("PREREG FL_NG plan") at the time of writing. Nothing below may be changed after FL_NG's export is opened; any later analysis is reported as post hoc.

## 1. What stays as frozen
- `paper/tools/label_day.py` (SHA-256 a7151fb9...) runs with `--protocol dual` on both annotators' final exports. Its reference is the mean of the two annotators; its agreement statistics (MAE h/v, ICC(2,1), weave kappa, Bland-Altman, median |diff|%, on all scenes, badge / no-badge, without the guide scenes 27, 65, 93, 110) and the GO / REDUCED thresholds (human vs human median |diff| <= 5% / <= 10%) are used as they are.
- `tools/iqa150.py` (frozen 2026-10-12T03:58Z) is run on the final headline labels.
- octave-v1 is evaluated exactly as in `methods/octave/PREREG_octave_v1.json` with `methods/octave/eval_octave_v1.py` (both frozen before FL_NG's export is opened; check their SHA-256 first). Its primary reference is FL_NG alone.

## 2. Added before opening FL_NG (pre-specified, reported alongside the frozen outputs)
1. **Octave disagreement between annotators.** For every scene and axis, r = log2(FL_NG / FL_SA). A line is *octave-disputed* if |r| > 0.5 (a ratio beyond about 1.41). Report: the number of disputed lines and scenes; the histogram of r; and, separately, the disputed lines among the 47 scenes where PG-Lattice reads about half or double FL_SA (FL_SA-based classification, `labelday_diag_FLSA_20261012/`).
2. **Adjudicated reference (sensitivity).** All method errors are also reported against a reference that leaves out octave-disputed lines (scene MAPE over the remaining axes), with the counts of both annotators listed for every disputed line in the supplement. The frozen mean-reference results stay the primary numbers unless rule 4 applies.
3. **Per-annotator references.** Every Tab. 1 method is also scored against FL_SA alone and FL_NG alone. Human-noise comparison: method-vs-FL_NG error is compared with FL_SA-vs-FL_NG error (not with the mean, which has less noise).
4. **Decision rule for the octave question (wording of Sec. 5.2).** On the axes where PG-Lattice reads about half or double FL_SA (47 scenes; 59 axis counts):
   - if FL_NG is within 10% of FL_SA on at least 80% of them: write "both annotators count the ends the machines halve or double";
   - if FL_NG is within 10% of PG-Lattice on at least 50% of them: write "the annotators disagree on the octave; the reference itself is ambiguous", report per-annotator results as co-primary and the adjudicated reference of rule 2 as the main table;
   - otherwise: report both readings without choosing.
5. **Log-ratio endpoint.** For every method, also report the mean |log2(prediction / reference)| over scene axes, and the share of axes with |log2 ratio| > 0.5 ("octave errors"), against FL_NG alone and against the mean. For octave-v1 vs PG-Lattice, the paired change in mean |log2 ratio| against FL_NG alone, with a scene bootstrap (10,000 resamples, seed 20261012) 95% CI, is a secondary endpoint next to the PREREG_octave_v1 primary.
6. **Fabric-level uncertainty.** All headline CIs are also reported clustered by the 44 fabric groups (`score150.py --groups tables/fabric_groups_v2.csv`, already part of label_day).
7. **CNN 5-fold row.** The scene-grouped CNN cross-validation that label_day runs is reported as frozen; a fabric-grouped version (44 groups as CV groups, same settings) is reported as a pre-specified sensitivity row.

## 3. What is not allowed after opening FL_NG
Changing octave-v1, its thresholds or its evaluation; choosing among reference rules by results; dropping scenes other than by rule 2; re-tuning any counter. Anything else is labelled post hoc.

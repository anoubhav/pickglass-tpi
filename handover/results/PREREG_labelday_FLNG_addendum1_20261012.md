# Addendum 1 to the FL_NG analysis plan (fixed before FL_NG's counts are opened)

Written Mon 12 Oct 2026 by the coordinating Claude session, after a post-hoc reliability check on FL_SA and BEFORE FL_NG's export was delivered or opened. Adds to `PREREG_labelday_FLNG_20261012.md` (SHA-256 9ae1d9ca...); changes nothing in it.

## What triggered it (FL_SA only, post hoc)
On the 41 same-patch re-shoot pairs (`paper/tables/fabric_groups_v2.json`; different photos of the same fabric patch, so the true count is the same up to an h/v swap), FL_SA's counts differ by a median of 20.5% on the worse axis (8 of 41 pairs within 5% on both axes, 31 with an axis off by more than 10%), against 1.8% for the frozen spectral counter (24 of 41 within 5%). Per-photo human counts may therefore carry noise comparable to the method errors.

## Pre-specified reliability analysis (run as soon as FL_NG's labels exist)
1. **Re-shoot self-consistency per annotator.** For FL_SA, FL_NG and their mean: per pair, the larger relative axis difference under the better of the direct / swapped assignment; report the median, the number of pairs within 3%, 5% and 10%, and pairs near 2x. Same for the frozen counter, GPT-6.1-sol P1 and ens-v1 (labels not needed). Script: `analysis_A2/run_a2.py --annotator-labels` (as tested on fake labels) or an equivalent short script; the definitions above are binding.
2. **Same-photo agreement.** The frozen label_day agreement statistics (median |diff|, MAE, ICC) between FL_SA and FL_NG on the same photos.
3. **Interpretation rule (binding for the wording):**
   - If FL_NG's re-shoot median difference is <= 5% and FL_SA vs FL_NG same-photo median |diff| <= 5%: the human reference is reliable; FL_SA's re-shoot inconsistency is reported as one annotator's noise and the mean / FL_NG reference is used as planned.
   - If both annotators' re-shoot medians are > 10% while their same-photo agreement is <= 5%: humans agree on a photo but the photo changes the count; report this as a capture-quality effect on human counting (a primary finding) and present method errors relative to that noise floor.
   - If FL_SA vs FL_NG same-photo median |diff| > 10%: the per-photo human reference is too noisy to test ±5%; the paper must say so in the abstract, report per-annotator results, and frame the benchmark around agreement with humans rather than accuracy.
   - Anything in between: report all numbers and both readings.
4. **Noise-floor comparison.** Every method's error against one annotator is compared with the other annotator's error against that annotator (method-vs-FL_NG against FL_SA-vs-FL_NG), as already in rule 3 of the plan.

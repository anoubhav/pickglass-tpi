# Problem definition

This page states the task the code solves, how it is scored, and what it does not claim. It follows the paper (Sec. 3; supplement Secs. S1 and S10); the clause-by-clause reading of the standards is in the supplement (Sec. S1, Tab. S1).

## 1. Setting

A textile inspector checks thread density, the warp **ends** and weft **picks** per inch, by pressing a 10x **pick glass** (a folding magnifier with a nominal 1 inch x 1 inch opening in its base) onto the cloth and counting yarns with a needle. We keep that tool and photograph the fabric **through** the glass with an ordinary phone. The opening and the scales engraved on the glass's frame act as a metric fiducial in every photo.

Capture is uncontrolled: distance, tilt, focus, lighting (LED hot spots, the phone's shadow), hand shake and resolution (about 460 to 2030 px per inch in PickGlass) all vary. The question is therefore an image-quality one: which capture degradations break the measurement, and can a no-reference quality score tell which counter to trust?

## 2. Task

Input: one RGB photo taken through the pick glass. Outputs:

| Output | Type | Required |
|---|---|---|
| window corners | 4 points in image pixels (TL, TR, BR, BL) | yes for the end-to-end track; given in the oracle-window track |
| `h`, `v` | real >= 0: the number of vertical (`h`) and horizontal (`v`) yarns crossing a horizontal and a vertical line across the window | yes (primary) |
| weave | plain, twill 2/1, 3/1, 2/2, satin, other, unsure | optional, scored separately |
| warp direction | vertical, horizontal, unknown | optional; maps `h`, `v` to ends and picks; not scored (a window rarely shows a selvedge, and on PickGlass neither annotator could determine it) |
| confidence | [0, 1]; abstention allowed | optional (selective prediction) |

**Measurand.** Count the individual threads actually present (ISO 7211-2): a plied yarn counts once; two yarns woven as one count as two. Edge rule (annotation tool): a yarn counts when its centre lies inside the window.

**Units.** A homography from the four corners to an S x S crop makes the image metric, so a count across the full rectified line is threads per window side at any resolution. It equals threads per inch only if the opening is one inch at the fabric plane. The glass's own scales put this in doubt: the coarse/fine tick-pitch ratio is 2.506 [2.501, 2.510] instead of 2.54, and the opening spans about 26.0 fine divisions. A 26 mm opening would make every count about 2.4 % high. Until the glass is measured physically, report "threads per window side, i.e. TPI up to the glass's calibration" (see [open problems](open_problems.md)). Comparisons with human counts are unaffected, because humans count across the same opening.

**Ends and picks.** ends/inch = `h` if the warp runs vertically in the image, else `v`; picks/inch is the other count. A 1-inch window rarely shows a selvedge, so the warp direction is a separate output and the main scores are axis-based. On PickGlass neither annotator could determine it on any scene, so it is not scored there (supplement Tab. S4).

**Formula-derived labels.** A count obtained by a formula (the legacy sheet derives three twill warp counts as `a + 2b`) is not a count of threads present. It is flagged and results are reported with and without it.

## 3. Why one window is a screening measurement

The standards (IS 1963:1981, ISO 7211-2:2024, ASTM D3775) average five or more counts per piece over 1 to 10 cm, depending on density. Tolerances come from product specifications, applied to the mean of the readings: for example +/-5 % (IS 179), +5/-2.5 % (IS 15853), or -5 % on the mean and -15 % on any single reading (IS 745). No public precision statement for pick-glass counting exists beyond a four-fabric ASTM round robin. One window is therefore a **per-window screening estimate** whose target accuracy is +/-5 %, never a standards-conformant determination. Out of scope: piece-level conformity, knitted and nonwoven fabrics, leno and lappet weaves, pile fabrics, yarn linear density.

## 4. Metrics

Let `y` be the reference count and `y_hat` the estimate, `e = (y_hat - y) / y`. Every image contributes two counts.

| Metric | Definition |
|---|---|
| within 5 % / 10 % (primary) | share of counts with abs(y_hat - y) <= 0.05 y (0.10 y) |
| image pass `both5` / `both10` (pre-registered) | share of images or scenes with **both** counts within 5 % (10 %) |
| MAE, MAPE | MAE in threads per window side (see Units), MAPE in % |
| symmetric error (fixed in advance) | mean abs(log2(y_hat / y)), which charges a halved and a doubled count alike (MAPE charges them 50 % and 100 %) |
| bias `bias_med` (pre-registered) and spread | median signed `e`; 1.4826 MAD of `e` (averaging five windows shrinks random error but not bias) |
| selective prediction | within-tolerance rate against coverage when the method abstains below a confidence threshold |
| weave | accuracy (and macro-F1, kappa) against the human label, beside the majority class |
| capture robustness | **self-consistency**: share of scenes whose count stays within 5 % of the method's own clean count under a degradation (label-free); accuracy under degradation where labels exist |

Uncertainty: percentile bootstrap over images or scenes (2,000 to 10,000 resamples), never over counts, because the two counts of an image are correlated. Paired comparisons resample jointly and are checked with Wilcoxon signed-rank and exact McNemar tests.

`paper/tools/score150.py` implements these metrics. Its MD5 (`5879ef44...`) is disclosed in the paper and listed in [CHECKSUMS.md](../CHECKSUMS.md); `both5`, `both10` and `bias_med` were registered before any human count existed.

## 5. Benchmark protocol

| Split | Content | Role | Rules |
|---|---|---|---|
| **PickGlass** | 150 phone photos from one site, audited down to **76 scenes**; window polygons for all; two annotators' counts of the 76 scene representatives, each counted from scratch. **A2**, the textile-trained annotator, is the reference for every main result; **A1**, without textile training, is a non-expert human baseline (first row of the paper's Tab. 1). The plan fixed before labeling scored against the mean of both annotators; scoring against A2 was decided after both annotators' counts were seen and is a declared deviation (mean of both: supplement Tabs. S18 and S13; each annotator alone: Tab. S15) | test; also training data, out of fold, for the post-hoc learned rows | the post-hoc learned rows are fitted to A1's counts, only out of fold, so A2's counts are their blind test: the CNN fine-tuned with 5-fold cross-validation grouped by fabric patch (every scene is predicted by a model that never saw its patch), the CNN + GPT-6.1-sol ensemble built on that CNN, and the learned octave selector; the scene-grouped five-fold CNN rows of supplement Tabs. S13 and S15 (`cnn-cv-*`) were instead trained out of fold on the mean of both annotators' counts, so they have no blind test and are optimistic (supplement Sec. S1 and Tab. S15); label-free uses of the photos (the window detector, the octave check, a CNN variant fine-tuned on the counter's confident counts of 28 scenes) are stated where they occur; one row per scene (n = 76) or scene-clustered intervals; every split is grouped by scene; 56 scenes re-shoot 24 fabric patches (44 fabric groups, `paper/tables/fabric_groups_v2.csv`; `score150.py --groups` for fabric-level intervals) |
| **Legacy-13** | 13 hand-counted window crops from an earlier project (10 plain, 3 twill 2/1); one count is a range, three twill warps are formula-derived, four images are marked unclear, orientation not recorded | held-out test | axis-matched scores (`h` <-> warp), one crop rule for every method; no counter was fitted on these crops except two post-hoc variants, and the set informed some design choices (supplement Sec. S1); anything chosen after seeing its scores is marked post hoc |
| **Synthetic** (`cnn/synth.py`, generator v2) | unlimited, seeded, exact labels | train / dev | the main place where things are fitted or tuned (confidence model, thresholds, CNN pre-training, the Molmo2-4B LoRA and its readout, the guard); uses of the other splits are stated in their rows |
| **Degradation suite** | 8 physically scaled degradations (defocus and motion in u = 0.001 inch, downscale, JPEG, noise, glare and shadow in % of the window, corner error in % of the side) | test | byte-identical degraded inputs for every method, fixed seeds |
| **PickGlass-Stress** | deterministic phone-camera re-captures of the real photos (`camera_aug/`), with a label audit | test | renders "can only lose information"; renders below about 4 px per yarn or with shake >= half a pitch are reported separately |

**Scenes.** The 150 files are 76 distinct exposures: one representative per scene (its highest-resolution real JPEG), 47 quarter-resolution copies (often stored turned by 90 degrees), 6 byte-identical files, 10 PNG decodes and 11 crops synthesised by the preparation notebook (augmentations, never test items). The 62 full/quarter pairs of 47 scenes are a controlled resolution experiment. `data/scene_map.csv` and `annotation_tool/data/manifest.json` hold the map; `annotation_tool/propagate_scene_labels.py` copies a representative's labels to its copies (with `h` and `v` exchanged for copies stored turned by 90 or 270 degrees).

**Leakage rules.**
- The counter, the scorer, the cascade decisions, the combined counter, the 76-scene generic-IQA test and the pre-registered two-annotator analysis (`paper/tools/label_day.py`: scores every method against the mean of both annotators and against each alone, and builds the mean-of-both tables) were frozen and hashed before any human count was looked at ([CHECKSUMS.md](../CHECKSUMS.md)).
- Molmo2-4B + synthetic LoRA (ours) was tuned on synthetic crops only; its checkpoint and its P1 readout were fixed on synthetic validation before any human count was scored (supplement Sec. S8).
- The post-hoc octave check, learned octave selector and CNN + GPT-6.1-sol ensemble (with its patch-grouped CNN) were chosen after A1's counts were seen and frozen before A2's counts were opened ([CHECKSUMS.md](../CHECKSUMS.md)).
- Scoring against A2 instead of the pre-registered mean of both annotators is a declared deviation, decided after both annotators' counts were seen (paper Sec. 3); every result against the mean is reported in the supplement.
- No machine dot, count or weave guess is shown to the annotators; machine dots appear only in the four worked examples of the guide, and scores are also reported without those scenes ([annotation protocol](annotation_protocol.md)).
- VLM prompts contain no test example and are logged verbatim with model id, date and repeats (`vlm/prompts.py`, `vlm/raw/`).
- Post-hoc choices (prompts P3s/P4q, Molmo2-4B's P4s readout, CountGD cells -> lattice, the guard vs cascade comparison, the octave check, the learned octave selector, the patch-grouped CNN and the CNN + GPT-6.1-sol ensemble) are labelled as such.

## 6. Reporting checklist

For each method and split: n (scenes or images, and counts), coverage, within 5 % and 10 %, both5 / both10, MAE and MAPE with 95 % CI, bias and spread, results per resolution class (50 full-resolution / 26 quarter-only scenes) and per capture-quality flag, cost and latency per image, whether it runs offline; for legacy-13 also with and without the formula-derived labels.

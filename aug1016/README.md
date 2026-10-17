# aug1016: the preparation notebook's augmented set as labelled 1-inch crops

This folder turns the augmented set of the preparation notebook into crops that the thread-count pipeline can use. The set is the "150 → 1016" expansion in `fabric_notebooks/1_Segmentation_Dataset.ipynb`, cells 46–91. The work here does four things:

* It rectifies every augmentation exactly as `common/rectify.py` rectifies the 150 originals.
* It recovers each augmentation's 90° rotation, which was never recorded. Two independent methods are used.
* It propagates the human labels of the 150 originals onto the augmented crops (`propagate_labels.py`). The labels flow in automatically once an annotation export appears.
* It runs a label-free invariance check with the frozen counter.

`cnn/train_cv.py --extra-aug1016` uses the result as extra training data. The source folder `fabric_notebooks/` is only read, never written.

## 1. Counts: what "1016" is

| step (notebook 1) | augmentations | total with the 150 |
|---|---|---|
| cells 55–56: 6 per original, `seed = 42 + i` | 900 | 1050 |
| cells 62–67: non-convex polygons deleted | −34 → 866 | **1016** (cell 72, hence the name) |
| cells 82–88: window < 21 % of frame deleted (13 of the 23 listed) | −13 → **853 on disk** | **1003** |
| this folder: corrupt polygons (6) and extreme-noise crops held out (4) | **843 usable** | **993** |

`missing_augmentations.csv` lists the 47 missing ids and the reason for each. The cell-86 list names 23 low-area ids, but only 13 are missing. The other 10 are still on disk (window 20.2–21 % of the frame). The execution order in that notebook is not linear: the histogram counts 853, yet the list includes deleted files. Those 10 are perfectly usable, and the manifest marks them `nb_low_area_listed`. Notebook 2 confirms 853 (`len(dataset_aug)`). **"More than 1,000 images" is true only for the on-disk total of 1003, which is 150 originals plus 853 augmentations. With usable labels the set is 993, or 997 if the four noisy crops were included.** None of these are extra photographs. They are synthetic variants of the 150 originals, so any evaluation must keep them in their source's fold.

Recipe (cell 53), with `KeypointParams(format="xy")` and the 4 polygon corners as key points:
`RandomRotate90(p=1)`, `RandomBrightnessContrast(0.3, 0.2, p=.7)`, `ShiftScaleRotate(shift .05, scale .05, rotate 0, p=.7)`, `Perspective(scale .05, p=.7)` (keep_size), `GaussNoise(var_limit=(2, 2), p=.1)`, `ImageCompression(q 70–80, p=.7)`. Images were then saved with `cv2.imwrite` at the default quality of 95. Cell 58 says the seeded run is **not reproducible**, so the parameters cannot be regenerated. That is why everything below is recovered from the files.

## 2. Corner convention: which case holds

* **The stored polygon order rotates with the image.** Albumentations keeps key-point order, so stored point *m* is the same physical corner in the original and in the augmentation. If you rectify with the *stored* order (point1 → top-left), the crop comes out already de-rotated. The best rotation is 0° for 849 of 853 augmentations. The 4 exceptions are corrupt polygons (§4).
* **`common/rectify.py` does not use the stored order.** It re-orders the corners geometrically (`order_corners`: clockwise on screen, top-left = smallest x+y), because image 22 is stored TR, TL, BL, BR. So, with rectify.py's convention, which we kept, **each augmented crop is screen-aligned: it equals the source's tool crop rotated by `k_deg`, up to photometric and resampling differences.** That is why the labels need `swap_hv`.
* Stored-order rectification would also *mirror* the six augmentations of image 22. That is a further reason to keep rectify.py's convention.
* Convention used everywhere: **`k_deg` is the counter-clockwise rotation**, as in `np.rot90` and albumentations' RandomRotate90 `factor`. In other words, `aug_crop ≈ np.rot90(src_crop, k_deg // 90)`. `swap_hv` = k ∈ {90, 270}.
* Each crop is `crops/{i}_aug{n}.jpg`, where *n* is the augmentation index and not the rotation. It is made with `R.order_corners`, `R.homography` (S = round(mean side length)), bilinear interpolation and BORDER_REPLICATE, then saved as JPEG q=92 like `annotation_tool/data/crops`. The crop is S × S, and S spans 1 inch.

## 3. Recovering k two ways

**(a) Geometric.** For each stored corner, take its angle about the polygon centroid in the augmentation minus the same angle in the original. The circular mean, rounded to 90°, gives k. `geo_resid_deg` is the worst corner's deviation from that multiple of 90°. It has a median of 2.6° and a maximum of 11.7° over the usable rows; perspective and anisotropic scaling account for the deviation.

**(b) Photometric.** This is the NCC between the augmented crop and the source crop rotated by j·90°, for j = 0..3. Each hypothesis is made by re-rectifying the source with the corners rolled by j, so it sits on the same pixel grid; `np.rot90` of an S × S crop is off by 1 px. The features are CLAHE grey and gradient magnitude, low-passed (σ = 2 px at 256 px/inch) and compared at 128 px/inch over the inner 84 %. The low-pass matters because fine yarns alias at 128 px/inch and make an unfiltered NCC depend on sub-pixel offsets. In the first version, without the low-pass, the 1st-percentile best NCC was 0.37; with it, it is 0.54. `ncc_margin` is the best minus the second-best score.

**Results.**

| | value |
|---|---|
| (a) = (b) | **849 / 853**. All 4 disagreements are corrupt polygons (§4). Among the 847 polygons without corruption, the two methods agree **847 / 847**. |
| k over the 843 usable rows | 0°: 202, 90°: 207, 180°: 230, 270°: 204 (`swap_hv` 411) |
| best NCC, usable rows | median 0.992, 1 % quantile 0.73, minimum 0.54 |
| NCC margin, usable rows | median 0.98, 1 % quantile 0.39, minimum 0.18 |

The six lowest margins (0.18–0.27) all belong to source 22. That fabric is almost 180°-symmetric (NCC 0.71–0.80 at the wrong half-turn), but in all six (a) and (b) agree and the geometric residual is small. Every disagreement and every weak row is drawn in `review/contact_flagged.png`, with the columns source | aug | source rotated by k_geo | source rotated by k_ncc. `review/contact_ok_sample.png` shows a random 24 of the accepted rows (both contact sheets are photo-derived and are released with the photos after review).

## 4. A bug in the notebook's set: stale key points (6 polygons)

Albumentations drops key points that leave the frame. `update_xml_with_new_bbox_and_dimensions()` then numbers the survivors from 1, so the trailing `<xN>/<yN>` tags keep the **source's untransformed coordinates**. The polygon mixes transformed and stale corners. It can still be convex and inside the frame, so the notebook's checks passed it. The signature is exact: a stored point that is bit-identical to the source's point at the same index while the polygon as a whole differs. That happens for 44_aug5, 59_aug5, 72_aug4 and 107_aug4 (point 4), and for 132_aug3 and 144_aug4 (points 3 and 4). Four of the six are the k disagreements. 72_aug4 agreed on k and had NCC 0.59, so only this check catches it. All six are rejected. Separately, 24 augmentations have all four points identical to the source. Those are genuine identity geometry (k = 0, no shift, scale or perspective) and are fine.

## 5. Which transforms fired (recovered, `t_*` columns)

| step | how it is detected | fired | recipe p |
|---|---|---|---|
| RandomRotate90 | k ≠ 0 | 648 / 853 (76 %) | 0.75 |
| Perspective | RMS residual of an affine fit to the 4 corners after undoing k > 0.4 ‰ of the image diagonal (int key points alone give ≤ 0.3 ‰) | 590 (69 %) | 0.7 |
| ShiftScaleRotate | affine part ≠ identity; only when Perspective did not fire (otherwise confounded) | 173 / 263 (66 %) | 0.7 |
| BrightnessContrast | reduced-major-axis fit aug ≈ g·src + o on low-passed aligned crops: \|g−1\| ≥ 0.02 or \|o\| ≥ 2 | 601 / 846 (71 %) | 0.7 |
| GaussNoise | Immerkaer σ(aug) / σ(source) > 1.5 | 80 (9.4 %) | 0.1 |
| ImageCompression | 8×8 blockiness > 1.13 (< 2000 px wide) or 1.17 (≥ 2000); unknown when noisy | 553 / 773 (72 %) | 0.7 |

Each threshold sits in the clear valley of a bimodal histogram, and every detected rate matches the recipe's probability. **GaussNoise is far stronger than its nominal `var_limit=2` suggests.** The full-resolution noise σ is 27–54 grey levels, against 0.06–1.9 without it. In four crops (35_aug4, 35_aug5, 58_aug5, 94_aug4) it drives the NCC below 0.5 even though k is certain.

## 6. Usable and review

`usable` means all of the following hold: valid polygon (4 points, convex, in bounds, no stale points), (a) = (b), best NCC ≥ 0.5, margin ≥ 0.15, and geometric residual ≤ 25°. That leaves **843 / 853**. `review/review_decisions.csv` lists the 10 others with a suggested decision. The 6 corrupt polygons are marked `reject`. The 4 noisy crops are marked `accept-if-noise-ok`: k is certain and the label is valid, but the crop is very degraded. To override, fill `decision` (accept / reject) and optionally `k_deg_override`. `propagate_labels.py` and `train_cv.py` obey the decision, and rebuilds keep your entries.

## 7. `manifest.csv` (one row per augmentation on disk)

The core columns come first: `aug_id, source_id, k_deg, swap_hv, polygon` (stored order, aug pixels, `x1 y1 … x4 y4`), `rect_size, ncc_margin, agree, usable, flags, needs_review`. Then:

* **k detail:** `k_geo_deg, k_photo_deg, geo_resid_deg, geo_slot_src/aug, ncc_best/second, ncc_clahe_best, ncc_grad_best, ncc_j0..j270, stored_order_best_deg, stored_order_ncc0`
* **Geometry:** `polygon_canonical` (TL, TR, BR, BL as used for the crop), `stale_points, img_w, img_h, source_rect_size, rect_ratio, area_pct, persp_rms_px, affine_scale, aniso, shift_px`
* **Photometry:** `bc_gain, bc_offset` (at mid-grey), `bc_r, noise_sigma, noise_sigma_src, noise_ratio, jpeg_blockiness`
* **Transforms:** `t_rot90, t_persp, t_ssr, t_bc, t_noise, t_jpeg` (empty = not recoverable), `nb_low_area_listed`
* **Paths:** `crop` (relative path), `image` (the source augmented JPEG)

## 8. Label propagation (`propagate_labels.py`)

```
<venv2>/bin/python propagate_labels.py                 # annotation_tool/data/saved/*.json|csv + aug1016/inbox/*
<venv2>/bin/python propagate_labels.py A.json B.csv     # explicit exports
<venv2>/bin/python propagate_labels.py --watch 60       # re-run whenever an export appears/changes
```

* **Inputs.** The script reads tool JSON (the Download export or the server autosave) and tool CSV, v1 or v2. It also reads a train_cv-style CSV (image|id, h_count/v_count or warp/weft + warp_axis, weave, group, exclude, uncertain).
* **v2 handling.** `h_tpi`/`v_tpi` come from the geometry through an exact Python port of `core.js lineInfo()`. `tests/test_propagate.py` checks the port against core.js on 600 random lines and round-trips JSON against the core.js CSV.
* **Filters.** These match `train_cv.load_annotations`. Only `done` records are used (`--include-not-done` overrides this). Rows that are nonwoven or excluded are skipped, and h and v must be > 0. When the same annotator and image appear in several files, the newest `timestamp` wins. With several annotators, each axis is averaged, the weave is the majority label, and the image is uncertain if any annotator marked it so (confidence 1 or uncertain = 1).
* **Swap (k = 90/270).** The swap applies to `h_tpi↔v_tpi`, `h_count↔v_count` and `h_angle_deg↔v_angle_deg`. A rotation preserves the sign of the clockwise-positive line angles. `warp_dir` swaps vertical↔horizontal, and `weave` is unchanged. k = 0/180 changes nothing.
* **Scenes (default).** The 150 originals are 76 scenes (`annotation_tool/propagate_scene_labels.py`). Each scene's label (the rep's, else the best labelled member's) is first copied to every file of the scene in that file's own orientation (h/v swapped where the file is stored turned 90/270° relative to the rep), so augmentations of copies are labelled too. Every crop inherits its source's scene: **`group` = scene_id**, and new columns are `scene_id`, `source_kind`, `source_scene_rot_k` and `label_src_id`. `--no-scenes --group-by source` restores the per-source behaviour described here. `propagate()` called from code keeps `group` = source_id unless `group_by="scene"` is passed. `tests/test_propagate.py` test 5 covers this.
* **Output.** `labels_aug1016.csv` has one row per usable crop. Its columns include `image` (absolute crop path), `aug_id, source_id, group` (= scene_id by default, else source_id; for group-aware CV), `k_deg, swap_hv`, the labels, `n_annotations, annotators, source_group` (if the source row had one) and `review_decision`. `labels_aug1016_meta.json` records the inputs and counts. `--with-originals` also writes the 150 source rows.
* **In this release.** No human annotation export is included (the photos and human counts are released after review), so the released `labels_aug1016.csv` holds only its header row. Run the script again, or leave `--watch` running, once annotation exports are available.

## 9. Invariance check now, without labels (`invariance_check.py` → `invariance/`)

The frozen counter (`methods/results/counting_frozen_final.py`, "ours") was run on every augmented crop. Its output was compared with its own predictions on the originals (`methods/predictions_ours_150.csv`), with h and v swapped for k = 90/270. The relative difference is \|pred − ref\| / ref, pooled over both axes. A "flip" is a crop whose larger per-axis difference is above 10 %. That is the counter's bistable octave or stripe-versus-yarn jump. There were 2306 counter runs. Full table: `invariance/summary.md`.

| subset | n | median | p90 | within 5 % | flip |
|---|---|---|---|---|---|
| baseline: no-op re-encode (tool crops vs in-memory crops) | 150 | 0.01 % | 0.10 % | 96.7 % | 5.3 % |
| baseline: pure rot90 + re-encode | 450 | 0.04 % | 0.68 % | 97.2 % | 4.4 % |
| baseline: matched (source re-rectified at the aug's S, rotated by k) | 843 | 0.06 % | 0.79 % | 95.4 % | 6.6 % |
| **augmented, usable** | **843** | **0.12 %** | **1.98 %** | **92.9 %** | **10.7 %** |
| augmented, usable, source confidence ≥ median 0.83 | 426 | 0.11 % | 0.71 % | 98.6 % | 1.6 % |
| augmented, usable, source confidence < median | 417 | 0.13 % | 25.7 % | 87.1 % | 19.9 % |
| augmented, usable, GaussNoise yes / no | 76 / 767 | 0.36 / 0.11 % | 53 / 1.4 % | 78 / 94 % | 30 / 8.7 % |
| augmented, usable, BrightnessContrast yes / no | 591 / 245 | 0.14 / 0.07 % | 2.3 / 0.9 % | 92 / 96 % | 11.5 / 8.6 % |
| augmented, usable, Perspective yes / no | 582 / 261 | 0.12 / 0.12 % | 2.7 / 1.2 % | 92 / 95 % | 11.9 / 8.0 % |
| augmented, usable, without the 14 counter-unstable sources and without noise | 699 | 0.10 % | 0.86 % | 97.6 % | 3.9 % |
| matched baseline, without the 14 unstable sources | 768 | 0.06 % | 0.60 % | 97.7 % | 3.5 % |
| flagged (not usable) | 10 | 8.5 % | 78 % | 45 % | 60 % |

How to read the table:

* **The typical augmented crop counts within about 0.1 % of its source after the swap.** That is evidence that the rectification and the recovered k are right, and that the propagated labels mean the same physical counts.
* **The whole tail is the counter's own instability, and it is concentrated on low-confidence sources.** Even a q92 re-encode flips 5.3 % of crops: 14 sources (listed in `invariance/counter_unstable_sources.txt`) flip under a re-encode or a pure rot90 alone.
* **Changing resolution alone accounts for most of the rest.** The matched baseline flips 6.6 %, against 10.7 % for the real augmentations. Without the unstable sources and noise, the real augmentations add only about 0.4 percentage points of flips over that baseline (3.9 vs 3.5 %).
* **GaussNoise is the one damaging step:** 30 % of those crops flip.
* **As a robustness result:** on high-confidence sources, real geometric and photometric re-capture-like changes leave 98.6 % of the counts within 5 %. Rotation (k) has no systematic effect.

## 10. Training with it (`cnn/train_cv.py`, defaults unchanged)

```
VC=<venv_cnn>/bin/python
$VC train_cv.py --labels annotations.csv --extra-aug1016 [--aug1016-labels aug1016/labels_aug1016.csv] [--aug1016-weight 1.0]
```

* **What it adds.** For each fold k, the training set gains every usable augmentation whose **source is not in fold k**. Augmentations never enter grouping or testing, so an augmentation can never be trained on while its source is tested. The folds are built from the originals exactly as before, including fabric groups and auto-grouping. By default the folds are grouped by scene (`train_cv.py --scene-mode members`), and every augmentation follows its source file's scene fold.
* **Where the labels come from.** By default they are propagated on the fly from this run's merged labels, using the same `propagate()` and review decisions. With `--aug1016-labels`, they are read from a `propagate_labels.py` output; both paths gave identical items in the smoke test.
* **Loading and weights.** Crops load lazily from disk, capped at 1024 px like the originals. The sample weight is 0.5 for uncertain sources, multiplied by `--aug1016-weight`. The `constant_mean` reference still uses the originals only.
* **Run outputs.** Each run writes `extra_aug1016.csv` (aug_id, source, source_fold, h, v) and logs, per fold, how many originals and augmentations were used.
* **Smoke test.** The smoke test used the existing `cnn/data/DUMMY_labels_for_smoke_test.csv` (fake labels) with 40 images, 2 folds and 1 epoch, and wrote to a temporary folder. It gave 228 augmented crops, 113 and 115 per training fold. Source folds matched `folds.csv`, all 107 swapped labels were correct, only usable rows were used, and a run without the flag produced identical folds.csv and constant_mean predictions.

## 11. Reproduce

```
cd "aug1016"
P=<venv2>/bin/python
$P build_aug1016.py --workers 6        # ~1 min: crops/, manifest.csv, missing_augmentations.csv, review/
$P invariance_check.py --workers 6     # ~8 min: invariance/ (2306 counter runs); --summarise-only re-tabulates
$P propagate_labels.py                 # labels_aug1016.csv from whatever annotations exist
$P tests/test_propagate.py             # needs node for the core.js comparison
```

Files: `aug_common.py` (shared helpers; all geometry goes through `common/rectify.py`), `build_aug1016.py`, `invariance_check.py`, `propagate_labels.py`, `tests/`, `inbox/` (drop exports here), `crops/` (853 files, 486 MB; photo-derived, released with the photos after review), `manifest.csv`, `missing_augmentations.csv`, `review/`, `invariance/`, `labels_aug1016.csv` and `labels_aug1016_meta.json`.

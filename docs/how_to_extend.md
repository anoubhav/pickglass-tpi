# How to extend the dataset or run the pipeline on new photos

Two routes: **(A)** count new photos with the released pipeline (no labels, minutes), or **(B)** add photos to a labelled benchmark the way PickGlass was built (photo -> window polygon or ruler -> rectify -> annotate -> propagate to copies -> score). Work in a fork or a branch: the PickGlass predictions in `methods/`, `paper/tables/` and `data/predictions/` are part of the frozen record ([CHECKSUMS.md](../CHECKSUMS.md)); never overwrite them.

## 0. Capture

What the capture-quality study says (paper Secs. 5.4 to 5.6; supplement Secs. S9 and S13 to S15):

- Press the pick glass flat on the fabric and keep the **whole window and the engraved scales** in the frame (the scales give the corners and the scale).
- Resolution: at least **4 px per yarn** on the denser axis, and about 7 for flip-free counts (the paper's rule, a guide, not a guarantee: counts collapse below about 2.5 to 3.5 px per yarn, and some counts still flip the lattice hypothesis below about 7 px per yarn). The window then needs at least 4 x threads-per-inch pixels across (about 7 x for flip-free counts); a 12.5 MP photo through the glass gives about 1700 px per inch. Shoot at full sensor resolution, no digital zoom.
- Hold still and focus on the fabric: motion blur and defocus are the failures that matter most; glare and shadow matter less.
- Keep the original file. A standard messaging forward re-compresses and down-scales (to about 530 px per inch, enough only up to about 130 threads per inch, or about 75 for flip-free counts), and **never up-sample or "enhance"** a photo: generative enhancement did not reliably help and up-sampling breaks spectral counting (`methods/guard/` repairs it only partly).
- Record for each photo: mill, fabric id, glass model, phone model, date; and whether two photos show the same fabric (needed for grouped splits).

## A. Count new photos (no labels)

```bash
python3 tools/count_photos.py new_photos/*.jpg --out new_counts.csv --crops-dir new_crops/
```

For each photo: corners from the glass's engraved scales (`methods/ruler_calib.py`), rectification (`common/rectify.py` convention), the frozen counter (`methods/results/counting_frozen_final.py`), and warnings below 4 px per yarn (unreliable) or between 4 and 7 (flips possible), for low confidence and for a weak ruler fit. The ruler detector is tuned to the glass used for PickGlass (a fine scale of 26 ticks on two adjacent edges and a coarse scale of 11 ticks on the other two). For another glass, give the corners yourself:

```bash
python3 tools/count_photos.py new_photos/*.jpg --corners my_corners.csv --out new_counts.csv
# my_corners.csv: file,x1,y1,x2,y2,x3,y3,x4,y4  (original pixels, any corner order)
```

Counts are **threads per window side**; they are threads per inch only for an opening of exactly one inch at the fabric plane (measure your glass: [open problems](open_problems.md)).

## B. Add photos to a labelled benchmark

### B1. Photos

Put the photos under `$PICKGLASS_NOTEBOOKS/final_image_set/` as `{id}.jpeg`, with ids continuing after the 150 PickGlass files (150, 151, ...). The scripts loop over `range(N_IMAGES)` (`N_IMAGES` is set in `common/rectify.py`), so raise `N_IMAGES` to the new total (or keep a separate collection in its own `PICKGLASS_NOTEBOOKS` folder with its own ids from 0).

### B2. Window polygon (or ruler)

Each photo needs the four corners of the window opening in original pixels. Three ways:

1. **Ruler (automatic):** `python3 tools/count_photos.py <photos> --first-id 150 --polygons-out new_polygons.csv`, then append the rows to `data/window_polygons.csv` after checking every photo with a *weak ruler fit* note. On PickGlass the ruler finds the window in 75 of 76 scenes, a median 0.19 mm from the human corners.
2. **By hand:** draw a 4-point polygon on the inner edge of the opening (any labelling tool) and store it as `final_bbox_annotation_set/{id}.xml` in the same VOC-like layout as the PickGlass files (`<object><polygon><x1>..<y4></polygon>`), or as a row of `data/window_polygons.csv` (`x1..y4` as drawn; `common/rectify.py` orders the corners TL, TR, BR, BL itself).
3. **Check:** `python3 common/rectify.py 150 151` prints the corners and crop sizes.

### B3. Rectify and build the manifest

```bash
python3 annotation_tool/prepare.py --ids 150 151 152      # crops/, crops_enh/, thumbs/ and the manifest entries
```

`prepare.py` warps each window to an S x S crop (S = mean side length in px, so no up-sampling) and writes `annotation_tool/data/`. Scene fields (which files are copies of which exposure) come from `annotation_tool/scenes.py` and `camera_aug/originals_provenance.json` (`camera_aug/provenance_scan.py`), which are written for ids 0-149: for new photos either guarantee one file per exposure (no copies), or extend the id range in `provenance_scan.py` and run `prepare.py --scenes-only`. Never put two files of one exposure in different splits.

### B4. Annotate

Follow [annotation_protocol.md](annotation_protocol.md):

As for PickGlass, have **two** annotators count every new scene from scratch, so that their agreement is measured. In the paper, A2 (textile-trained) is the reference for every main result and A1 (no textile training) is a non-expert human baseline; the plan fixed before labeling scored against the mean of both (supplement Tab. S18), so scoring against A2 is a declared deviation. Fix your primary reference before you look at the counts. Agreement on one photo is not the noise floor: what matters is how closely one annotator repeats a count across two photos of the same fabric patch, which for the paper's reference is a median 8.9 % (paper Sec. 5.3 and Tab. 2; supplement Sec. S4). One bundle per annotator, each with its own storage key:

```bash
python3 annotation_tool/kit/build_bundle.py --data annotation_tool/data --out bundle_A1 --hide-proposals --static --storage-key A1
python3 annotation_tool/kit/build_bundle.py --data annotation_tool/data --out bundle_A2 --hide-proposals --static --storage-key A2
```

Host each bundle (static) under its own unguessable path, screen the annotators first, ask each for a Download after 10 photos, and check every export:

```bash
python3 annotation_tool/kit/check_export.py annotations_A1_*.json annotations_A1_*.csv --second annotations_A2_*.json annotations_A2_*.csv
python3 annotation_tool/kit/show_dots.py annotations_A1_*.json annotation_tool/data/crops review_A1.jpg --ids 150,151
```

### B5. Propagate labels to copies

```bash
python3 annotation_tool/propagate_scene_labels.py annotations_A1_*.json annotations_A1_*.csv annotations_A2_*.json annotations_A2_*.csv
# -> annotation_tool/propagated/labels_per_scene.csv and labels_per_file.csv (several annotators: per-axis mean)
```

A copy stored turned by 90 or 270 degrees gets `h` and `v` exchanged (and the warp direction turned). Every scorer calls the same propagation.

### B6. Predict

Run every method on the new crops **before** looking at the labels, and save the predictions with their hashes:

- PG-Lattice (frozen): `python3 methods/run_ours_150.py` (all ids up to `N_IMAGES`; writes `methods/predictions_ours_150.csv` and `methods/proposals.json`, so run it in your branch).
- VLMs: `vlm/run_vlm.py` / `vlm/openrouter_client.py` (needs `OPENROUTER_API_KEY`; prompts in `vlm/prompts.py`; raw answers are logged in `vlm/raw/`).
- CNN: `cnn/evaluate.py`; Molmo2-8B and CountGD: `gpu_results/code/`, `gpu_results/countgd/code/` (CUDA machine, `PICKGLASS_GPU_HOST`); CountGD was also run locally on Apple MPS for the legacy crops (`cnn/countgd/run_countgd.py`).
- Molmo2-4B + synthetic LoRA (ours): `ft_results/code/` (`train_lora.py` tunes Molmo2-4B on synthetic crops only, `infer_ft.py` runs the tuned model; the adapter is not in git, see [THIRD_PARTY.md](../THIRD_PARTY.md)).
- Any other method: write a wide CSV with columns `image_id, <key>__v, <key>__h` (VLM convention: `__v` = vertical yarns = our `h`), which `score150.py --extra-wide PATH::key` reads.

### B7. Score

```bash
cd paper
python3 tools/score150.py --labels ../annotation_tool/annotations_A1.csv --labels ../annotation_tool/annotations_A2.csv --out ../scratch/my_tables \
    --extra-wide ../vlm/results/rect150_P1_predictions_wide.csv::gpt-6.1-sol --fig --fig-dir ../scratch/my_tables
```

One row per scene by default (`--unit scene`), the mean of the annotators as the reference (the rule fixed before labeling; the paper reports it in supplement Tab. S18) or one annotator with `--annotator NAME` (the paper's Tab. 1 scores against A2 and is built by `paper/tools/reference/make_tab1_ref.py`), bootstrap intervals over scenes, and the pre-registered `both5`, `both10`, `bias_med`. Always pass `--out` (the default writes into `paper/tables/`). Add `--groups <csv>` to cluster the bootstrap by fabric when several scenes show one fabric.

## C. Add a new method

1. Develop and tune **only on synthetic data** (`cnn/synth.py`, seeded, exact labels; the counter's development log is in the docstring of `methods/counting.py`).
2. Freeze the code and record its hash before running it on PickGlass or legacy-13.
3. Run it on the 150 crops (or the 13 legacy crops), write a wide CSV, and score it with `score150.py --extra-wide` (PickGlass) or add it to `paper/tools/legacy_unified.py` (legacy-13, one crop rule).
4. For capture robustness, run it on the byte-identical degraded inputs of `methods/degrade.py` / `methods/ablate.py` and on PickGlass-Stress (`camera_aug/build_stress.py`).

# Thread-count annotator v2.2.2 (WVAQ 2027)

A dependency-free browser tool for counting threads per inch (TPI) on rectified 1-inch fabric crops. Plain HTML + JS, with a standard-library Python server for disk autosave.

**v2.1 (Oct 8 2026): one photo per scene.** The 150 files are only **76 distinct photos (scenes)**. Annotate the 76 photos in **Show → Scenes to annotate (76)**, which is now the default view. Every other file is a copy of one of them, and its labels are copied automatically (see *Scenes* below). If you already started on a copy, that work is kept.

**v2.2 (Oct 8 2026): "Check carefully".** Some photos show an orange **Check carefully** or a yellow **Take care** badge with a reason, such as "Check carefully: estimators disagree on H by 41%". The badge comes from machine checks only and never shows a count. Count every yarn as usual. It never appears on IAA-subset or blind images. **Show → High attention first** lists the other photos with the flagged ones first (see *Check carefully* below).

**v2.2.1 (Oct 8 2026): turn the image.** Press **,** and **.** (or use the **Image** slider in the *Guide lines* card) to turn the photo a little so the yarns line up with the guide lines. The lines stay where they are on screen and the fabric turns under them, so the count is exactly the count of turned lines (see *Turning the image* below). **R** turns the image until the selected line crosses its yarns at right angles, and **/** turns it back to 0°. Reload the page to get it; nothing else changes.

**v2.2.2 (Oct 8 2026): Examples.** The **Examples** button (or **X**) opens a panel with two tabs. **How it looks** shows example figures of four photos outside the IAA subset (the 1-inch crop with both guide lines and the machine's proposal dots, plus two 1/10-inch zooms) and six one-line tips. **My finished scenes** shows your own finished scenes as thumbnails with your dots, newest first; click one to open it. Esc closes the panel. Reload the page to get it; the record format does not change (see *Examples panel* below).

**Start:** `cd annotation_tool && python3 server.py --open` (the 150 crops will be added to this repository; `python3 server.py --demo --open` opens the 3 synthetic practice fabrics in `demo/data/`). This opens <http://127.0.0.1:8765/>. Use Chrome and always the same port. Type your name in **Annotator** first. Press **?** in the app for the illustrated how-to.

1. **What to count:** each crop is exactly the 1-inch pick-glass window. Put one dot on **every yarn that crosses the guide line**, from edge to edge. The H line counts vertical yarns and the V line counts horizontal yarns. Count an edge yarn only if its centre is inside the window. In a two-colour plain weave, a light float and the dark float next to it are **two** yarns.
2. **Align the lines (v2):** if the yarns are tilted, turn each line so it crosses its yarns at right angles. **A** auto-aligns the selected line to the estimated yarn direction (shown in the *Guide lines* card, never applied by itself; Shift-A does both lines). Drag the green ⟳ handle, or press **[** / **]** for ±0.5° (Shift ±5°), and **0** to level it again. **L** switches the selected line. The two lines are independent, so they need not stay perpendicular. A line turns about its centre and is clipped to the window. **Or turn the image instead (v2.2.1):** **,** / **.** turn the photo ∓0.5° (Shift ±5°) under the lines until the yarns cross them at right angles. **R** does it from the same estimate as **A**.
3. **Editing:** click on a line to add a dot (the click is projected onto the line), click a dot to delete it, and shift-drag a box to delete many. Drag a line's round handle to move it (its angle is kept), and press Z to undo. The starting dots are machine guesses and are often wrong, for example one dot per stripe or per float pair. A red bar marks a gap (missed yarn?) and an orange bar marks two dots too close (duplicate?).
4. **The count:** the big number is **threads/inch = dots on the line ÷ length of the line inside the window (inches)**. For a level line that length is exactly 1 inch, so it equals the dot count. The raw dot count, the line length and the angle are shown below it. If rotating a line near the window edge leaves dots outside the window, they are flagged and not counted, and **Remove** deletes them.
5. **Dense or dark fabric:** zoom (wheel or +) until the yarns are at least ~10 px apart. Pan with space-drag and work from one end of the line to the other. E switches to the contrast-enhanced view. If glare or blur hides yarns, move the line to a clearer row; the count is the same anywhere.
6. **From scratch (the protocol used for the paper, Oct 11 2026):** two annotators each count all 76 scene representatives from scratch. The deployed manifest hides every machine proposal (`proposals_hidden: true`), so no photo opens with starting dots, a count or a weave guess: the 20 **IAA subset** scenes open in blind mode (purple badge) and the other 56 open in normal mode with no dots. Machine dots appear only in the four guide examples (scenes #27, #65, #93 and #110, *Examples* below). A photo can be marked done only when it is complete (`app.js` `doneProblems`; the rule is in `docs/annotation_protocol.md`). The tool still supports the earlier plan (blind IAA subset first, then the rest with machine starting dots; every blind toggle is logged), which was not used.
7. **Each image:** set weave (keys 1–7), warp direction (W), quality flags and confidence (C). Then press Enter (done + next) or S (done). N jumps to the next unfinished image. The ⏱ timer (top right) shows active time on the image. It pauses after 60 s without input, and an idle gap longer than 60 s is not counted at all.
8. **Saving:** work autosaves to the browser and to `data/saved/annotations_<name>.json`. At the end of every session, click **Download** (JSON + CSV; allow both downloads) and send both files. **Import** restores a JSON or CSV (v1 or v2); the newer edit wins.

## Scenes (v2.1)

`prepare.py` adds the scene structure of `../camera_aug/originals_provenance.json` to every manifest image. `scenes.py` computes it deterministically from the provenance file and the crops; it never reads annotations.

| Field | Meaning |
|---|---|
| `scene_id` | Id of the scene's representative ("rep"): the highest-resolution real JPEG of the scene (ties: lowest id). The reps are exactly the photos listed in `scenes_to_annotate.txt`. |
| `is_scene_rep`, `scene_members` | Whether this file is the rep, and all file ids of the scene. |
| `kind` | `rep` (76), `quarter_res_copy` (47: the same exposure reduced exactly 4×), `byte_identical` (6), `png_copy` (10: pixel-identical PNG decodes of a JPEG), `notebook_synthetic` (11: made by the 2024 notebook with a 97 % crop, ±90° rotation and ±10 % brightness/contrast; these are augmentations, not photos). |
| `copy_of` | For byte_identical and png_copy, the file it duplicates. For notebook_synthetic, the JPEG it was made from. |
| `scene_rot_k`, `scene_swap_hv` | `crop(file) ≈ np.rot90(crop(rep), k)` (counter-clockwise, the aug1016 convention). If k is odd, H and V are swapped when labels are copied. |
| `scene_rot_ncc`, `scene_rot_margin` | NCC of the best of the 4 rotations, and the margin to the second best. |

**Copies are often rotated.** Of the 63 real copies, 41 are stored turned relative to their rep: 33 quarter-res copies at k = 3, 3 at k = 2, and 5 byte-identical copies. So for 36 of them the counts must be copied with **H and V swapped**. Of the 11 notebook files, 9 have an odd k relative to the rep. All 11 are turned ±90° relative to the file they were made from.

How k was measured: the rep is re-rectified with its corners rolled by j = 0..3. Each hypothesis is compared with the file's tool crop using the aug1016 features (CLAHE grey + gradient magnitude, low-passed, 128 px/inch, inner 84 %). The margin is at least 0.226 (median 0.987). Two independent checks agree on all files: a whole-photo NCC that does not use the window polygons (74/74 non-rep files, `data/scenes_check.csv`, written by `python scenes.py --check-images`), and the `rot90_k` of the 62 full/quarter pairs in the provenance file (62/62). The 10 PNG copies are pixel-identical to their JPEG.

**In the tool:**

* **Show → Scenes to annotate (76)** is the default filter. It lists the reps only, and the progress bar counts scenes (n / 76). **Not done** and **N** also skip copies, because a copy is never "to do".
* A copy shows a teal badge, **"same scene as #<rep> — labels copied"**. It is **read-only**: no dots, lines, weave or flags can be changed. The count card shows the rep's counts in this file's orientation, swapped when it is turned 90°, or "not counted yet". **G** or **Go to #<rep>** opens the rep.
* **Edit anyway** (in the badge, after a confirmation) unlocks one copy for the session and creates a normal record for it. The record opens blind if its scene is in the IAA subset. That record is used only as a repeat count, or as a fallback if the rep is never annotated. The rep's labels always take precedence.
* A copy that already has a record (e.g. image 0, opened before v2.1) keeps it unchanged and shows it read-only.
* Opening the tool on a copy jumps to its rep, with a notice.
* **The record format is unchanged.** It has the same fields and the same 62 CSV columns. Only `tool_version` reads 2.1.0. v2.0 and v2.1 read each other's records. With a manifest without scene fields, v2.1 behaves exactly like v2.0.

**IAA subset v2.** The 20-image subset is now **20 distinct scenes, all reps**: `[3, 7, 14, 18, 33, 34, 35, 38, 39, 49, 50, 55, 67, 76, 77, 89, 97, 99, 107, 124]`. The old subset drew 20 *files* and contained the same-scene pairs (10, 33) and (61, 69) as well as non-rep copies.

* **How it was drawn:** stratified by resolution (13 full-resolution scenes and 7 quarter-resolution-only scenes) × the frozen counter's weave guess (plain 9, twill 8, other/unknown 3). Allocation is largest-remainder proportional, with seed 2027. The strata are in `iaa_policy`.
* **The old subset:** it is kept in the manifest's `iaa_history`. It was retired when **0 records were done**: `data/saved/` held one record, image 0, not done. Had any old-subset image been done, `prepare.py` would have kept it in the new subset.
* **Records opened under the old subset:** a record that was opened but never edited follows the new membership and blind policy, and the audit trail stays honest (`proposals_seen` stays true). Edited records are never changed.
* **Scoring:** the scripts take IAA membership from the manifest, not from the per-record flag.

**Labels for scoring: `propagate_scene_labels.py`.**

* **What it does:** it reads the exports read-only (default: `data/saved/*.json|csv`) and gives every file of each labelled scene the scene's label in that file's own orientation. The label is the rep's own label. If the rep has none, it is the best labelled member's label, rotated into the rep's frame.
* **What it writes:** `propagated/labels_per_file.csv` (150 rows), `labels_per_scene.csv` (76 rows) and `labels_meta.json`. `--policy pool` averages all labelled members.
* **Who uses it:** `../paper/tools/score150.py`, `../cnn/train_cv.py` and `../aug1016/propagate_labels.py` all call it.
* **Evaluation units:** the **primary** evaluation is **one row per scene** (the rep). The **secondary** evaluation is per file over the 139 real files, with a scene-clustered CI. **notebook_synthetic files are never test items:** they are used for training only, in their scene's fold. Cross-validation folds are grouped by scene, and aug1016 crops inherit their source's scene.

## Check carefully (v2.2)

`prepare.py --attention-only` gives each of the 76 scene reps an **attention** level (low / medium / high) with generic reasons. `attention.py` computes it and runs deterministically. It reads only existing machine outputs and never reads an annotation, `data/saved/`, an export or a propagated label.

**Disagreement.** For each axis the tool compares four estimators: ours (the frozen counter, whose dots are the proposals), the log-Gabor bank, GPT-6.1-sol with prompt P1, and the CNN V5-B0 from the held-out file. The CNN is left out on the 28 rep scenes it was fine-tuned on, so it counts on 48 reps. With *m* the median of the log counts, the disagreement is **D = exp(max(|ln ours − m|, median |ln e − m|)) − 1**. D measures how far the proposals are from the consensus, or how far the estimators typically are from it, whichever is larger. A single outlying estimator does not raise D. A 2:2 split between a count and its double gives D = 41 %.

**Points.** Each check below adds points. The thresholds were declared before any human count and use the paper's units.

| Check | Points |
|---|---|
| Disagreement: largest D ≥ 33 % (the estimators disagree on the counting unit) | 2 |
| Disagreement: largest D ≥ 10 % (the paper's tolerance) | 1 |
| Px per yarn (S ÷ the consensus count of the finer axis) < 4 | 2 |
| Px per yarn 4–5 | 1 |
| Resolvability η below the frozen cascade gate's r_min | 2 |
| η below the label-free 5th percentile of the 150 clean photos (blur) | 1 |
| Near-saturated pixels (sat240) > 1 % (glare) | 1 |
| Ours' confidence below the gate's t_c = 0.469 | 1 |

A score of 2 or more is **high**, 1 is **medium** and 0 is **low**.

The thresholds come from `../paper/tables/cascade_thresholds.json`, η from `cascade_150_decisions.csv` and sat240 from `strata150.csv`. The manifest's `attention` block lists the rule, the thresholds and the md5 of each input. Re-running the script writes an identical file.

**Distribution (Oct 8 2026):**

| Reps | High | Medium | Low |
|---|---|---|---|
| All 76 | 37 | 15 | 24 |
| 56 outside the IAA subset | 26 | 12 | 18 |
| 20 IAA reps (computed but never shown) | 11 | 3 | 6 |

| Reason | Reps |
|---|---|
| Disagreement on H | 25 |
| Disagreement on V | 28 |
| Low confidence | 15 |
| Fine yarns | 6 |
| Glare | 5 |
| Blur | 3 |

**Per-image fields (reps only):**

* `attention`, `attention_score` and `attention_reasons`.
* `attention_codes`: `disagree_h`, `disagree_v`, `fine_yarns`, `low_resolvability`, `glare`, `low_confidence`.
* `attention_signals`: D per axis in %, the number of estimators, the confidence, η, sat240 and a px-per-yarn bin. There is no count, because S is shown in the tool and S ÷ px per yarn would give one.

**In the tool:**

* **High and medium:** a badge on the image shows the first reason ("Check carefully: …" or "Take care: …"). The info card lists all the reasons.
* **Low:** nothing is shown.
* **Hidden on:** IAA-subset scenes (always, also if blind was turned off), blind records (any policy) and copies.
* **High attention first:** this Show option lists the reps where the badge may be shown. They are ordered high → medium → low, then by score, then by id. ← → and N follow this order. An image that will open blind under the current blind policy is not listed.
* **Without attention fields:** with a manifest that has none, the option is hidden and the tool behaves like v2.1.
* **Record format:** unchanged (same keys and 62 CSV columns). Only `tool_version` reads 2.2.0 for records created in v2.2.

The rule above decides whether a record's annotator could have seen a badge: a non-IAA rep whose record was not blind, annotated with tool v2.2. In the protocol used for the paper (two annotators, all 76 counted from scratch, no machine dots) these are the 38 high- and medium-level non-IAA reps (*Check carefully* 26, *Take care* 12); the badge never shows a count, and the agreement of the two annotators is reported separately on badge and no-badge scenes. The 20 IAA reps also carry a level, so after labelling you can check whether attention predicts disagreement between the two annotators.

## Turning the image (v2.2.1)

**Controls.**

* **,** and **.** turn the image −0.5° and +0.5° (Shift: ±5°); + is clockwise, as for the lines.
* The **Image** slider in the *Guide lines* card sets any angle from −45° to +45°; the readout next to it shows the current turn.
* **/** (or the 0° button next to the slider) turns the image back to 0°.
* **R** (**Align image**) turns the image until the selected line crosses its yarns at right angles.
* Zoom, pan, clicks, box-delete, both handles and **F** all work on the turned image. **F** and an unzoomed view fit the whole turned crop. When you are zoomed in, the point in the middle of the view stays put while the image turns.

**What it does.** The turn is **rigid with the guide lines**. Turning the image by *d* turns the photo on screen by *d* about the crop centre and also turns both lines by −*d* on the fabric. Each line therefore keeps its angle on screen and its pivot (a fixed point of the fabric), while the fabric turns under it. A level line stays level, and you turn the fabric until its yarns are vertical and horizontal on screen. The dashed window outline turns with the image, so the tilted 1-inch edges stay visible.

**The count.** There is no new maths. Turning the image by *d* gives exactly the same record geometry as turning both lines by −*d* with **[** / **]**: the same angles, segments, dot positions and threads/inch. TPI is still dots on the line ÷ the length of the line inside the window, and the line angles in the card (and in the record) are angles on the fabric. Dots stay in image coordinates, and each keeps its parameter on its line (x for H, y for V). Turning with dots already placed therefore moves them a little along their yarns, as turning a line does; a hint says so.

**How it combines with the line controls.**

* **A** / Shift-A still set a line's angle on the fabric to the estimate, whatever the image turn is. On screen the line then shows at estimate + image turn, across the yarns.
* **R** (*Align image*) uses the same estimate but turns the image instead of the line. It picks the turn after which the selected line lies at the estimate on the fabric while keeping its angle on screen (turn = line angle − estimate). A level line stays level and the yarns end up vertical and horizontal on screen. The other line turns with it, rigidly. After **R**, **A** reports "already aligned".
* **[** **]**, the green ⟳ handle and **0** still turn one line on the fabric. **0** means parallel to the window edge, so with a turned image the line shows tilted like the window outline.
* **/** turns the image back rigidly: the lines go back to the angles they had on the fabric before the turn. This also undoes an alignment made with **R**.
* **Reset to proposals** turns the image back to 0° as well (the proposals are positions on level lines).
* Each line still stays within ±45° on the fabric, so a turn stops early when a line would pass 45° (a hint says which limit was hit).

**Unchanged.**

* **Read-only copies:** the image cannot be turned (same hint as any other edit) and the controls are disabled.
* **Blind records and IAA images:** nothing new is shown. The turn shows no count, and **R** uses only the yarn-direction estimate that **A** already shows.
* **Undo:** **Z** undoes a turn (a burst of key presses or one slider drag is one step). As in v2.0.1, the audit trail stays append-only.

**Record.**

* **`view_rot_deg`:** a new optional field. It is absent or 0 for an image that was never turned, so old records load unchanged and a new record does not get it until the image is first turned.
* **`angle_log`:** each turn adds one entry per line, `{t, line, deg, how: "view", view, src}`. `deg` is the line's new angle on the fabric, `view` the new image turn, and `src` is `key`, `slider`, `zero` or `align`. `align` entries also carry `est`, `clear` and `ref` (the line whose estimate was used). A burst of key presses or one slider drag within 3 s is one pair of entries.
* **`h_angle_src` / `v_angle_src`:** read `view` when a turn set the line's angle last.
* **Undo and reset:** an undo that changes the turn logs `{how: "undo", view}` entries, and a reset logs `{how: "reset", view: 0}`.
* **CSV:** one column, `view_rot_deg`, is appended at the very end (63 columns; the first 62 are unchanged and in the same order, and every existing row is the v2.2.0 row plus `,0`). Readers that select columns by name, and every scoring script, are unaffected. CSVs without the column import as before.
* **`tool_version`:** reads 2.2.1. It stays 2.2.x on purpose, because `paper/tools/label_day.py` finds records that could have shown the v2.2 *check carefully* badge with `tool_version.startswith("2.2")`, and 2.2.1 shows the badge exactly as 2.2.0 does.
* **Turn analysis:** `analyze.py` counts lines turned by an image turn among its "rotated guide lines".

## Examples panel (v2.2.2)

**Examples** in the top bar, or **X**, opens the panel; **Esc**, **X** or a click outside closes it. While it is open no shortcut acts on the image and the ⏱ timer pauses (reading examples is not time spent on the image).

**How it looks.** Six one-line tips (the same rules as the guide) and these figures, listed in `guide/examples/examples.json`:

* `example_27.jpg`, `example_65.jpg`, `example_110.jpg` (full resolution) and `example_93.jpg` (quarter resolution, 510 px/inch). Each shows the 1-inch crop with both guide lines and the **machine proposal dots**, and two 1/10-inch zooms, one along each line. The caption says: *machine proposals — shows the layout, not verified counts*. The figures print no count and no weave guess (the tool never shows the counter's weave guess, so a printed one could anchor the weave label).
* `photo_problems.jpg`: glare, blur, shadow, tilted yarns, low resolution and a hard pattern (crops only: no dots, no counts), picked with the rules of `bad_photos.png` from **non-IAA** reps that show no IAA patch (#43, #48, #20, #9, #126, #103).

**What is never used.** `guide/annotation_guide.png` shows proposal dots on scene 77, which is in the IAA subset. `guide/bad_photos.png` has no dots or counts, but three of its six crops are IAA scenes (#39, #124 and #21, a copy of #77) labelled with machine quality judgements, which the tool never shows on IAA scenes. `guide/howto_count.png` shows proposal dots and a count on a photo that is not in the IAA subset but shows the **same patch of fabric** as an IAA-subset scene (see below), so it would hint at a blind count. All three files are kept unchanged but are not shown in the tool.

**How the figures are chosen** (`make_tool_examples.py`, deterministic):

* Candidates are scene reps only. A file is excluded if it, or its scene, is in the manifest's `iaa_subset`.
* There are four slots: plain, twill and other (satin, pattern or unknown) at full resolution, and any weave at quarter resolution. The weave is the frozen counter's guess and the resolution split is the one used for the IAA strata.
* Within a slot the order is: attention level *low* first (the estimators agree and nothing is flagged), then the counter's confidence (highest first), then id.
* **Same patch.** The scene grouping joins only the resolution copies of one photo, but several reps are other photos (other lighting, shadow or turn) of the same patch of fabric as an IAA-subset scene; machine dots or a count on such a photo would hint at the blind count. The script therefore also excludes a rep whose 1-inch crop matches an IAA rep's crop (SIFT features + RANSAC similarity, at least 12 inliers at scale about 1; on this data the matching photos have 16 or more, every other photo at most 5). Needs OpenCV (`cv2`, in the figure venv).
* The script never opens `data/saved/`. It refuses to write an index that names an IAA scene or another photo of an IAA patch.

**The tool checks the figures again** (`C.visibleExamples`). It refuses any figure that names no scenes, or that names an unknown id, a copy or an IAA-subset scene (or a copy of one). It also refuses a path outside `guide/`. It **hides** a figure while one of its scenes is, or will open, **blind** for the current annotator: when the record is blind, or, before the scene is first opened, when the blind policy says it will open blind (*always blind* hides them all, *random 50%* about half). A note says how many figures are hidden.

**My finished scenes.**

* **Your records only.** The tab lists only the records whose `annotator` field is the current **Annotator** name (compared as stored, i.e. by the browser and disk key). It never lists another annotator's records, nor records without a name, even if they are in the same file or browser. With an empty name it shows a prompt to type one instead of a gallery. A second person who types their own name in the same browser sees none of the first person's finished scenes. While a name switch is loading, the panel shows nothing per annotator.
* **Unnamed work is taken over only on request.** Work done while the Annotator name is empty is saved without a name (in server mode also to `data/saved/annotations_anonymous.json`, which a fresh browser loads while its name is empty). Before v2.2.2 the first name typed silently took all of it over. Now the tool asks first (*Are they YOUR work?*, with the number of annotations and the time of the last edit only): **OK** files them under the new name, **Cancel** leaves them unnamed and starts the new name with only its own records. A second person must answer Cancel, or they would see, and later be credited with, the first person's dots.
* **What is listed.** Only records marked done, newest first. The header shows *n done / 76 scenes* (scene reps; an unlocked copy is listed with a *copy of #…* tag but not counted).
* **Each thumbnail** shows the crop turned by the image turn (`view_rot_deg`), the window outline, both guide lines at their angles and every dot, with the same mapping as the main view. Below it are the H and V threads per inch, the weave, the confidence and the quality flags. A red frame and a red *H empty* / *V empty* marker show a line with no dots (cleared?). Click a thumbnail to open that scene.
* **IAA-subset scenes** (from the manifest; a record's own `iaa` flag can only add one) are listed so that the count and the click-through work. They show the plain photo only: no dots, no lines and no threads per inch. The weave, confidence, flags and the *empty line* marker are shown.
* **Read-only.** Nothing in the panel changes a record. The grid is emptied whenever the panel closes.

**Regenerate** (`<venv2>` = the figure venv): `<venv2>/bin/python make_tool_examples.py` writes `guide/examples/` (`--dry-run` prints the picks only). `node tests/test_examples.js` (14) tests the gallery filter (own records only, done only, IAA handling, order, n / 76, read-only), the figure filter (IAA ids, copies of IAA scenes, unknown ids and odd paths refused; hidden when blind), the thumbnail geometry under image and line turns, and that a new name takes over unnamed work only after a confirmation. `<venv2>/bin/python tests/test_examples.py` (10) checks that the figure selection never picks an IAA scene or another photo of an IAA patch, even when doctored data would rank such photos first, that `examples.json` is up to date and that no annotation is read.

**Record format:** unchanged. `tool_version` reads 2.2.2. It stays 2.2.x because `paper/tools/label_day.py` uses `tool_version.startswith("2.2")`.

## Keyboard

| Key | Action |
|---|---|
| ← → | previous / next image (in the current filter) |
| N | next scene that is not done (in **High attention first**: the next one in that order) |
| G | on a copy: open its scene's rep (the photo to annotate) |
| S · Enter | mark done · mark done and go to next |
| L | switch the selected guide line (H ↔ V) |
| A · Shift-A | auto-align the selected line · both lines |
| [ · ] | rotate the selected line −0.5° · +0.5° (Shift: 5°); + is clockwise |
| 0 | selected line back to 0° (parallel to the window edge) |
| , · . | turn the image −0.5° · +0.5° (Shift: 5°); the guide lines stay put on screen |
| / | image back to 0° (the lines turn back with it) |
| R | turn the image so the selected line crosses its yarns at right angles (*Align image*) |
| 1–7 | weave: plain, twill 2/1, twill 3/1, twill 2/2, satin, other, unsure |
| W · C | cycle warp direction · confidence |
| B | blind mode (hide proposals) for this image; every toggle is logged |
| Z · Cmd/Ctrl-Z | undo |
| E · F · + − | contrast-enhanced view · fit · zoom |
| wheel · space-drag | zoom · pan (dragging empty space or middle-drag also pans) |
| round handle · Alt-drag on a line | move a guide line |
| green ⟳ handle | rotate the selected line |
| Shift-drag | box-delete dots |
| Image slider | set the image turn (−45° … +45°) |
| X | Examples panel (how it looks · my finished scenes) |
| ? · H · Esc | help · help · close |

## Data (v2 is backward compatible)

v1 records (level lines, no angle fields) load and display unchanged, and their counts stay the same. A record gains the v2 fields the first time it is edited in v2. Exports always include them, recomputed from the geometry.

* **Record fields (v2):** `h_angle_deg`, `v_angle_deg` (y down, positive = clockwise, |angle| ≤ 45°, missing = 0) and `angle_log` (`[{t, line, deg, how}]`, where `how` is auto, key, drag, zero, reset, undo or, since v2.2.1, view). v2.2.1 adds the optional `view_rot_deg` (the image turn, missing = 0; see *Turning the image*). Undo (Z) restores the dots, lines and fields but never the audit trail: `proposals_seen` stays true once proposals were shown, `blind_log` and `angle_log` are append-only (a blind or angle change made by an undo is logged with `undo`), and active time is kept. Each line also gets derived fields: `<line>_seg_len_px`, `<line>_seg_len_in`, `<line>_raw_count` and `<line>_tpi`. The H line passes through (S/2, `h_line_y`) and the V line through (`v_line_x`, S/2). Dots keep their v1 meaning: H dots are x positions on the line and V dots are y positions.
* **CSV:** the 47 v1 columns are unchanged and come first (`h_count` / `v_count` are still the dot counts). Appended at the end: `h_angle_deg, v_angle_deg, h_seg_len_px, v_seg_len_px, h_seg_len_in, v_seg_len_in, h_raw_count, v_raw_count, h_tpi, v_tpi, warp_tpi, weft_tpi, h_angle_src, v_angle_src, tool_version` and, since v2.2.1, `view_rot_deg` (63 columns).
* **Readers:** `analyze.py`, `../paper/tools/score150.py` (and `quality150.py` through it) and `../cnn/train_cv.py` use `h_tpi` / `v_tpi` when present and fall back to the counts otherwise. For level lines both give the same numbers.

**Auto-align estimate:** the plain crop is resampled to 512 × 512 (1 inch, so Nyquist is 256 threads/inch). A Hann-windowed 2-D power spectrum is whitened per frequency ring, and every bin with whitened power z > 6 votes for its angle within ±18° of each axis; all harmonics of a yarn family lie on one ray. A tilt prior and a joint H/V choice (a rigidly turned fabric has equal angles) suppress twill-diagonal harmonics. The peak is then measured on log-weighted votes. Estimates with an H/V difference over 8° or an ambiguous peak are marked *weak*. It runs in a Web Worker (core.js) in about 45 ms per image. On the 150 crops it agrees with the lattice method's angles within 2° for 133/147 H and 130/147 V lines, and for 200/211 of the lines it calls *clear*.

## Maintainer

`<venv2>/bin/python prepare.py --scenes-only` adds or refreshes only the scene fields and the IAA subset (v2) in the existing manifest. Proposals, crops and annotations are untouched; the run is deterministic and idempotent (a second run writes an identical file). `--saved-dir` sets which autosaves are checked for done records before an IAA re-draw. A normal `prepare.py` run also adds the scene fields; `--no-scenes` gives a v2.0 manifest. `<venv2>/bin/python prepare.py --refresh-proposals` merges `../methods/proposals.json`. Optional `h_angle_deg` / `v_angle_deg` (or the lattice method's `angle_x` / `angle_y`) are passed through and shown next to the estimate. Images opened but never edited take the new proposals; any edited image keeps its dots. `node tests/test_core.js` runs the core tests (39: geometry, normalisation, projection, v1/v2 round trips, merge, CSV order, estimator; since v2.2.1 also the image turn: rigid turn = the equivalent line rotation, limits, pointer mapping round trips at many turns / zooms / pans, clicks and box-delete on a turned image, line dragging, logging / undo / reset, records and CSVs with and without `view_rot_deg`). `node tests/test_attention.js` (9) and `<venv2>/bin/python tests/test_attention.py` (8) check the attention fields, their visibility rules, the filter order, determinism, that no count is shown and that no annotation is read. `<venv2>/bin/python prepare.py --attention-only` (or `<venv2>/bin/python attention.py` to print the table without writing) refreshes the attention fields; every normal run computes them too (`--no-attention`: none), and `--scenes-only` keeps them as they are. `node tests/test_scenes.js` (11) checks the scene fields of the manifest, the filters, progress, copied labels, the IAA re-sync and the unchanged record format. `<venv2>/bin/python tests/test_scenes.py` (7) checks the scene structure, the IAA draw, propagation and its agreement with core.js. `node tests/test_examples.js` (14) and `<venv2>/bin/python tests/test_examples.py` (10) check the Examples panel and its figure selection (see *Examples panel*); `<venv2>/bin/python make_tool_examples.py` regenerates `guide/examples/`. `python3 analyze.py A.json B.json` reports anchoring, agreement and rotation usage. If only a JSON exists (e.g. the disk autosave), `node json2csv.js data/saved/annotations_<name>.json` writes the identical CSV that the scoring scripts read. Each file's docstring has the details. The v1 files replaced by v2 are kept in `backup_v1/`, the v2.0 files (and the manifest) replaced by v2.1 are in `backup_v2/`, the v2.1 files (and the manifest) replaced by v2.2 are in `backup_v21/`, the v2.2.0 files replaced by v2.2.1 (image turn) are in `backup_v23/`, and the v2.2.1 files replaced by v2.2.2 (Examples panel) are in `backup_v24/`.

## About

Thread-count annotator v2 · Licence: MIT (see `LICENSE` at the repository root) · Please cite: see `CITATION.cff`

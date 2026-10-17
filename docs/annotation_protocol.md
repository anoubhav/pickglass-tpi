# Annotation protocol

How the human counts of PickGlass are made, checked and scored, and how to repeat this with new annotators or a new dataset. Code: `annotation_tool/` (the tool) and `annotation_tool/kit/` (bundle builder, export checker, screening scorer, dot viewer).

## 1. The tool

`annotation_tool/` is a dependency-free browser app (`index.html`, `app.js`, `core.js`, `style.css`), version 2.2.2. It can run in two ways:

- **Local, with disk autosave:** `python3 annotation_tool/server.py --open` (the 150 crops are released after review; [DATASHEET.md](../DATASHEET.md)), or `python3 annotation_tool/server.py --demo --open` for the three synthetic practice fabrics in `annotation_tool/demo/data/` (port 8766, so demo records never mix with real ones).
- **Static, browser-only** (how the paper's two annotators worked): `python3 annotation_tool/kit/build_bundle.py --data <data dir> --out <bundle> --static [--hide-proposals]`, then put the folder on any static host. Work autosaves in the browser's storage; **Download** writes a JSON and a CSV export; **Import** restores a download. Nothing is uploaded.

On every photo the annotator sees the rectified 1-inch window (the four corners come from the window polygon, so corners are never marked) and two guide lines:

1. **H line** (horizontal): one dot on every vertical yarn it crosses. **V line** (vertical): one dot on every horizontal yarn. The count is the number of dots divided by the line length in inches; for a level line that is the number of dots. A line can be turned to cross tilted yarns at right angles (A), or the image can be turned (R); the tool corrects for the tilt.
2. **Weave** (plain, twill 2/1, 3/1, 2/2, satin, other, unsure), **warp direction** (vertical, horizontal, unknown), six capture-quality **flags** (glare, blur, shadow, low resolution, perspective, partially occluded), **non-woven**, **confidence** (1 rough, 2 mostly sure, 3 sure) and **notes**.
3. The tool records the active time, every blind toggle and line rotation, the position and origin (human or machine) of every dot, and the time each record was created and last changed.

**Scene-aware.** The 150 files are 76 scenes. The default view, *Scenes to annotate (76)*, lists only the representatives; copies are read-only and show the representative's counts (with `h` and `v` exchanged for copies stored turned by 90 or 270 degrees). `annotation_tool/propagate_scene_labels.py` applies the same rule when scoring.

**Export format.** One record per photo (JSON) and a 63-column CSV (`annotation_tool/README.md`, section *Data*). Scorers read `h_tpi` / `v_tpi`. `node annotation_tool/json2csv.js export.json` turns a JSON export into the identical CSV.

## 2. Guides

- [Counting guide](counting_guide.md): the short public guide given to the annotators (synthetic pictures only).
- [Annotator guide](annotator_guide.md): the full guide that comes with the tool, edited for this release: every real-photo figure is replaced by a synthetic one.

The rules in one paragraph: zoom until neighbouring yarns are well apart; one dot on every yarn the line crosses, from one edge of the window to the other; a yarn at the edge counts only if its centre is inside the window; a light float and the dark float next to it are two yarns; diagonal twill stripes and colour blocks are not yarns; never skip a bad photo: count what you can see, tick the problem and lower the confidence; leave a line empty only if its yarns cannot be seen on any row, with the reason in Notes and confidence 1; pick *Unsure* rather than guess the weave; *Unknown* is the normal warp direction.

## 3. The completeness rule

The deployed `annotation_tool/app.js` (function `doneProblems`) refuses to mark a photo **done** unless all of the following hold. `annotation_tool/kit/check_export.py` re-applies the same rule to an export (Python port).

| Check | Rule |
|---|---|
| weave | set (*Unsure* counts) |
| confidence | set |
| empty line | allowed only with a reason in **Notes** and **confidence 1** |
| span | the dots of each line cover at least **75 %** of the line, unless *Partially occluded* is ticked or Notes is non-empty |
| density | each counted line has at least **10 dots** |
| gaps | no gap larger than **3x the median dot spacing** (shown as a red bar), unless *Partially occluded* is ticked or Notes is non-empty |

The 10-dot minimum has no exception; *Partially occluded* or a note waives only the span and gap tests. A photo that was done and is edited into an incomplete state is un-marked. Note that `<line>_tpi` divides by the full line length: a line counted over only part of its length (allowed with *Partially occluded* or a note) under-counts. `paper/tools/label_day.py` therefore leaves every line spanning less than 75 % of the window out of every score (with the other line of the same photo), and a sensitivity check adds such lines back, counted over the annotated part (`check_export.py` lists them).

## 4. Screening test and pass rule

Candidates first count **two photos with trusted reference counts** in the same tool (a two-image static bundle with its own storage key), then report the four numbers (H and V per photo).

**Pass rule:** every number on both photos within **10 %** of the reference count, where for each photo the better of the direct and the swapped H/V pairing is used (a candidate who mixes up the directions is not failed for that alone). A missing photo or an empty line fails. Score with `python3 annotation_tool/kit/score_screening.py --key <your private key.json> --reply "<the candidate's text>"`. Tell candidates only pass or fail.

For the paper, the two screening photos were the legacy crops r2 and a5 (hand counts 99 x 69 and 47 x 49). The two annotators answered 97 x 69 and 46 x 50 (A1) and 99 x 68 and 48 x 49 (A2): all eight counts within two yarns, 2.1 %, of the hand counts (supplement Tab. S4). These reference counts are printed in the paper, so **do not reuse r2 and a5** to screen anyone new; pick photos whose counts are not public.

## 5. The protocol used for the paper (two annotators, from scratch)

| Item | What was done |
|---|---|
| annotators | **two** external annotators who passed the public screening test above; each counts on their own, without seeing the other's work. Records carry pseudonymous codes (A1 and A2 in the paper). A2 has textile training; A1 has none |
| photos | each annotator counts all **76 scene representatives** (`data/scenes_to_annotate.txt`) from scratch; copies receive the representative's labels |
| machine help | **none**: the deployed manifest has `proposals_hidden: true` and `proposal: null` for every image, so no machine dot, count or weave guess is shown |
| difficulty badges | 38 of the 56 representatives outside an earlier 20-scene agreement subset show a qualitative badge, *Check carefully* (26) or *Take care* (12), with machine-derived reasons (estimators disagree, very fine yarns, low resolvability, glare, low confidence) but **never a count** (`attention` fields of `annotation_tool/data/manifest.json`); the 20 subset scenes show none. Agreement is also reported separately on badge and no-badge scenes |
| guide figures | the four worked examples of the annotator guide (also shown in the tool's *Examples* panel) show machine dots on scenes **#27, #65, #93 and #110**; scores are also reported without these four scenes, and the post-hoc proposal check leaves them out |
| completeness | enforced by the tool (section 3) |
| agreement | measured on every scene counted by both annotators: MAE per axis, Bland-Altman bias with 95 % limits of agreement, ICC(2,1) for absolute agreement, weave kappa and the median absolute difference in %, also split by badge (supplement Tab. S4); each annotator's repeatability across the 41 re-shoot pairs is in the paper's Tab. 2 and supplement Sec. S4 |
| reference | **A2**, the textile-trained annotator, for every main result; A1's counts are a **non-expert human baseline** (first row of the paper's Tab. 1). The plan fixed before labeling scored against the **mean of the two annotators' counts**; scoring against A2 was decided after both annotators' counts were seen and is a declared deviation. Results against the mean of both: supplement Tab. S18 (and Tab. S13 with MAE and weave); against each annotator alone: Tab. S15 |
| deliverable | the tool's JSON + CSV exports of each annotator (released with the photos after review) |

**Checking that no machine help was shown.** An earlier plan (a blind 20-scene agreement subset and machine starting dots on the other 56 scenes) was replaced by the protocol above; the guide the annotators received still described that plan in places ([annotator_guide.md](annotator_guide.md) is edited to the protocol as run). A1's tool link existed before the proposal-free bundle was deployed; A2's link was created after it. A1's records therefore need the check below, because a record first opened before the deployment could have shown machine starting dots. Because core.js sets `proposals_seen = true` on every non-blind record even when it inserts zero dots, that flag is not evidence of assistance; the evidence is the machine-dot fields and the time each record was created. Before scoring, run

```bash
python3 annotation_tool/kit/check_export.py annotations_A1_*.json annotations_A1_*.csv \
    --second annotations_A2_*.json annotations_A2_*.csv --hidden-since <upload time, UTC>
```

which lists every record with machine dots (`n_prop_h`, `n_prop_v`, `proposal_method`, `h_orig`, `v_orig`) and every record first opened before the proposal-free bundle was deployed; `paper/tools/label_day.py` reports the same as "proposal-free records: k of N" (`--proposals-hidden-at`).

**Scoring.** `paper/tools/score150.py --labels <exports>` (frozen, MD5 in [CHECKSUMS.md](../CHECKSUMS.md)) scores every method per scene against the mean of the annotators' counts by default, or against one annotator with `--annotator <code>`. The paper's Tab. 1 (every method, and A1's counts, scored against A2) is built by `paper/tools/reference/make_tab1_ref.py`. `paper/tools/label_day.py --protocol dual` (frozen before any human count was looked at, SHA-256 in [CHECKSUMS.md](../CHECKSUMS.md)) is the pre-registered two-annotator analysis: it checks the exports, propagates the labels to copies, computes the agreement statistics, scores every method against the mean of both annotators and against each alone, and builds the mean-of-both tables. It refuses test-looking annotator names.

## 6. How to repeat the protocol

1. **Screening.** Screen every annotator with the test and pass rule of section 4 (a separate two-photo bundle with its own storage key) and score the reply with `score_screening.py`.
2. **Main count.** Build one main bundle per annotator with `--hide-proposals --static --storage-key <code>`, each under its own unguessable URL, separate from the screening bundle (or behind an access control). Ask each annotator for a first Download after about 10 photos and run `check_export.py` and `show_dots.py` on it to give concrete feedback (missed yarns, double clicks) early.
3. **Delivery.** Run `check_export.py` on each annotator's final export, with `--second` for the other annotator (one name per export and distinct names, all representatives done, completeness rule, no machine dots, scenes done by both), keep the raw exports unchanged, and score with the frozen scorer.
4. **Two annotators on the same scenes** (as in the paper): report the agreement statistics of section 5, each annotator's re-shoot repeatability, and every method's score against each annotator and against the mean of the two, so that every method's error can be read against the human noise floor. Fix in advance which reference is primary (the paper's plan fixed the mean of both; scoring against the textile-trained annotator is its declared deviation). A third annotator, or a repeat count by the same person days later, would separate disagreement from individual repeatability ([open problems](open_problems.md)).

Names of annotators are never published: records carry a pseudonymous code, and screening exports are not part of any release.

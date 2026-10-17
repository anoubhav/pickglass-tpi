# Open problems

What the paper could not settle, in rough order of how much each would change its conclusions. Each item says what is known, what is missing and what would close it. Contributions are welcome (issues and pull requests).

## 1. Calibrate the glass

**Known.** Every count is a count across the window opening. The paper reports "threads per window side, i.e. TPI up to the glass's own calibration" because the glass's engraved scales disagree with its nominal 1-inch opening: fitted per photo over the 149 calibrated photos, the coarse/fine tick-pitch ratio is 2.506 [2.501, 2.510] (not 2.54, the ratio of 0.1 inch to 1 mm), and the opening spans 26.0 fine divisions (25.98 in the human polygons' own frame). Either the opening is about 26 mm, so every count is about 2.4 % high, or the ticks sit above the fabric plane and are imaged smaller. The loupe's pincushion distortion biases centre-line counts by about 3 % (`camera_aug/measure_distortion.py`).

**Missing.** A physical measurement. **To close:** measure the opening with calipers, and photograph a steel rule or mm graph paper laid flat under the same glass (and a printed grid for the distortion); then fix the scale in `methods/ruler_calib.py` (`GLASS`) and report absolute TPI. Comparisons between methods and humans are unaffected (both count across the same opening).

## 2. The human noise floor beyond two annotators

**Known.** Two external annotators, screened on the same public test, each counted all 76 scene representatives from scratch with no machine dots ([annotation protocol](annotation_protocol.md)). A2, with textile training, is the reference, with no second expert to check them; A1, without textile training, is a non-expert human baseline. The plan fixed before labeling scored against the mean of both (supplement Tab. S18); scoring against A2 is a declared deviation, and some rankings depend on it (supplement Tab. S15). On the same photo their counts differ by a median 32.0 % of their mean (22.8 % on the 38 scenes without a difficulty badge; ICC(2,1) 0.20 / 0.28 for h / v), above the 10 % limit fixed in advance, and their weave labels agree no better than chance (kappa -0.01; supplement Tab. S4). On the two screening photos all eight of their counts are within two yarns (at most 2.1 %) of the legacy hand counts. But 41 pairs of scenes photograph the same fabric patch again, and across them even the reference repeats a count only to a median 8.9 % (the non-expert 20.5 %; paper Tab. 2): this re-shoot spread, not the screening test, is the noise floor against which every method is read, and a +/-5 % tolerance cannot be verified against one photo count. Much of the disagreement is a factor of two: the annotators differ by more than a factor of 1.41 on 72 lines of 50 scenes (supplement Tab. S17), and no count was checked on the cloth, so which reading is right remains open. No public precision figure exists for pick-glass counting (the ASTM D3775 precision statement rests on a 1984 four-fabric study).

**Missing.** Repeatability (the same person counting the same photo again), agreement with counters from outside these two annotators, such as trained textile inspectors counting the physical fabric through the glass, and a counting convention that annotators share (what counts as one yarn when plies, paired yarns or floats are visible at 10x). **To close:** a recount of the disputed scenes on the cloth; a written convention for plies, paired yarns and floats; a repeat count of a stratified subset (by resolution class and weave) by each annotator some days later; a third annotator, ideally an inspector, on all 76 representatives with the same proposal-free bundle; report the same statistics and score methods against each reference. `paper/tools/label_day.py --protocol dual` and `annotation_tool/analyze.py` compute them.

## 3. More mills, glasses and phones

**Known.** One site, one pick-glass model, probably one phone model; 76 scenes, 26 of them only at quarter resolution; 56 of the scenes re-shoot 24 fabric patches, so the 76 scenes form 44 fabric groups (label-free audit: `paper/tools/fabric_groups_v2.py` -> `paper/tables/fabric_groups_v2.csv`). The phone-camera profiles of PickGlass-Stress (`camera_aug/camera_profiles.json`) are market-informed priors, not measured devices.

**Missing.** Any evidence that the numbers transfer. **To close:** collect photos at other mills (powerloom and handloom), with other glasses (other tick layouts break the ruler detector's assumptions; record the glass model) and current phones, with a capture protocol (see [how to extend](how_to_extend.md), section 0) and per-photo metadata (fabric id, session, phone); group every split by fabric. Calibrate each new phone's MTF and noise to turn the stress profiles into measured ones.

## 4. Defocus and other capture failures

**Known.** Motion blur, defocus, corner error and fewer than about 4 px per yarn break counts. Under 2 to 4 u of defocus the reciprocal-lattice counter locks onto residual high-frequency artefacts and over-counts by up to nearly 7x (legacy MAPE 186 to 246 %, supplement Tab. S5), and its confidence does not flag this, so a blur gate must precede any recapture prompt; VLMs degrade differently, which the quality-aware cascade exploits (its resolvability check was designed after the defocus failure on the legacy crops was known; the gate was then frozen and tested once on PickGlass, supplement Sec. S11). Generative enhancement does not reliably rescue cheap captures; the native-resolution guard repairs up-sampling but gets worse under heavy defocus.

**Missing.** A no-reference defocus / resolvability score that is calibrated on real photos, and real (not simulated) blurred captures with labels. **To close:** capture the same fabric patches at controlled focus offsets and shake (a tripod and a motorised stage, or a phone on a rig) and count them; fit and test the gate on that set; extend the cascade to request a recapture.

## 5. A multi-window protocol

**Known.** The standards average five or more counts per piece over 1 to 10 cm; one window is a screening estimate. Averaging five windows shrinks random error by about the square root of five but leaves bias, so the median signed error matters as much as the spread.

**Missing.** A protocol and a dataset where several windows per piece are photographed, with a reference count of the piece by the standard method (IS 1963 / ISO 7211-2 / ASTM D3775). **To close:** photograph at least five places per piece, count the piece by the standard method, and report piece-level pass/fail agreement against the product tolerances (+/-5 %, +5/-2.5 %, or -5 % on the mean and -15 % on any single reading).

## 6. Smaller open items

- **Weave recognition** is unsolved for every method, including the VLMs; colour-woven checks add spectral peaks; satins other than 5-end ones, basket weaves, jacquards and knits fall outside the lattice model.
- **Warp direction** is rarely visible in one window; the legacy labels never recorded it.
- **Legacy labels:** three twill warp counts of the legacy sheet are formula-derived (`a + 2b`) and conventions differ between methods (a6, r8, a1); a recount of the physical fabrics, if they still exist, would settle them.
- **Re-shoot audit:** the label-free grouping (`paper/tables/fabric_groups_v2.csv`) joins photos of the same fabric patch; photos of different patches of one fabric piece cannot be matched this way, so the 44 groups bound the number of distinct fabric patches from above, not the number of fabrics. Fabric ids and capture sessions were not recorded, so this stays open.
- **VLM results are a snapshot** of hosted models that change over time (model ids and access times are in the run logs, `vlm/raw/` and `vlm/logs/`); re-run the roster in `vlm/MODEL_ROSTER.md` and report the drift.
- **Photo release:** the 150 photos, rectified crops, stress renders and both annotators' counts with dot positions are released after review (DATASHEET.md).

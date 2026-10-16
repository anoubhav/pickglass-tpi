# Datasheet: PickGlass-TPI

Following *Datasheets for Datasets* (Gebru et al., CACM 2021). Items marked **TBD** are completed when the data are added; the data licence is a prerequisite for releasing any photo-derived file.

| Status (Oct 12 2026) | |
|---|---|
| In this repository | scene map, window polygons, manifest (no images), photo provenance (sizes, MD5, JPEG properties; no pixels), every method's predictions, the 2019 hand counts of the 13 legacy crops, synthetic demo images |
| To be added (after the review period) | the 150 photos, rectified crops, the 13 legacy crops, PickGlass-Stress renders, generative-enhancement outputs, degraded copies, the human counts with dot positions and timing logs, CNN weights fine-tuned on pseudo-labels of the real photos |
| Consent | The author took the photos with the mill's agreement. |
| Licence for the data | To be announced ([DATA_LICENSE](DATA_LICENSE) is a placeholder) |

## 1. Motivation

- **Purpose.** To study thread-density estimation (warp ends and weft picks per inch, and the weave) from smartphone photos taken through a 10x pick glass on a factory floor, as a capture-quality problem: which degradations break the count, and whether a no-reference quality score can say which counter to trust. Created for a paper at the WACV 2027 workshop on image/video quality (WVAQ).
- **Gap.** Prior image-based density measurement assumes controlled capture (scanners, backlit rigs, microscopes, loom cameras). No public dataset of pick-glass phone photos with thread counts existed.
- **Creators.** The paper's author (see [CITATION.cff](CITATION.cff)). The author took the photos by mid-2019; the 13 legacy crops and their hand counts come from a 2019 student project (Textile Quality Analysis, Computer Vision and Intelligence club, IIT Madras; see the README's acknowledgements); window polygons were drawn in 2024; the scene audit, predictions and the 2026 human counts were made for the paper.
- **Funding.** **TBD.**

## 2. Composition

- **Instances.** 150 image files of woven fabric seen through one pick-glass model, from **76 distinct scenes** (exposures):
  - 76 scene representatives (the highest-resolution real JPEG of each scene): 50 at full resolution (1838-2032 px per inch across the window) and 26 only at quarter resolution (485-521 px per inch);
  - 47 quarter-resolution copies of the same exposure;
  - 6 byte-identical duplicates, 10 PNG decodes of JPEGs (stored with a .jpeg name);
  - 11 crops synthesised by a 2024 preparation notebook (97 % crop, +/-90 degree rotation, +/-10 % brightness and contrast): training augmentations, never test items.
  36 real copies are stored turned by 90 degrees relative to their representative. The 62 full/quarter pairs of 47 scenes form a controlled resolution experiment. 56 scenes re-shoot 24 fabric patches (the same patch photographed again under other light, shadow, focus or turn), so the 76 scenes form 44 fabric groups (label-free audit: `paper/tools/fabric_groups_v2.py` -> `paper/tables/fabric_groups_v2.csv`; it supersedes the 64 candidate groups of `fabric_groups_candidate.csv`).
- **Files.** Sizes 4608x3456 (58), 864x1152 (48), 1152x864 (31), 3352x4469 (5), 1117x838 / 838x1117 (3 each), 4352x3264 (2). No file has EXIF; every JPEG was last re-saved by libjpeg at quality 75 (`data/photo_provenance.json`).
- **Labels per file.** Window polygon (four corners of the opening, drawn by hand in 2024; `data/window_polygons.csv`); scene id, kind, rotation relative to the representative (`data/scene_map.csv`, `annotation_tool/data/manifest.json`).
- **Labels per scene representative (label day, Oct 12 2026; not yet in this release).** From each of two annotators, counted from scratch and independently: the number of yarns crossing a horizontal and a vertical guide line (`h`, `v`) with every dot position, weave, warp direction, six capture-quality flags, non-woven, confidence, notes, active time and an audit log. Copies inherit the representative's labels (`h` and `v` exchanged for copies turned by 90 or 270 degrees). The reference count of a scene is the mean of the two annotators' counts.
- **Legacy-13.** 13 window crops from the 2019 project (10 plain, 3 twill 2/1) with 2019 hand counts (`data/legacy13_hand_counts.csv`): one count is a range (110-120, used as 115), three twill warp counts were derived with a formula (`a + 2b`), four images were marked unclear, orientation was not recorded. The original photos behind these crops are lost.
- **Confidential or sensitive content.** The photos show fabric on a working factory floor. Woven designs can be a client's property. Frames may show hands, faces or labels. No personal data of annotators are released: records carry a pseudonymous code.
- **Splits.** PickGlass is the test set; legacy-13 is a held-out test. The CNN is pre-trained on synthetic data (`cnn/synth.py`), and most methods use no PickGlass counts. Some methods also learn from PickGlass, always out of fold: a CNN fine-tuned on the spectral counter's confident counts of 28 scenes (no human counts; scored only on the other 48), and, post hoc, CNNs fine-tuned on the human counts with five-fold cross-validation (each scene is predicted by a model that never trained on it; with folds grouped by fabric, never on its fabric), the CNN + GPT-6.1-sol ensemble (ens-v1) and a learned octave selector. The post hoc methods were frozen before the second annotator's counts existed. Every statistic is per scene (n = 76) or scene-clustered; every split is grouped by scene.
- **Errors, noise.** Two annotators from one recruitment round each count every scene; their agreement on the same photo (MAE, Bland-Altman limits, ICC(2,1), weave kappa) is to be added. On two screening photos all eight of their counts are within two yarns (at most 2.1 %) of the 2019 hand counts. The paper's noise floor is the re-shoot spread: the first annotator's counts of two photos of the same fabric patch differ so much (row "Annotator, another photo" of the paper's Tab. 1) that a +/-5 % tolerance cannot be verified against one photo count. The glass's nominal 1-inch opening is unverified: it spans 26.0 fine divisions of the engraved scales, +2.4 % wider than 1 inch if these are millimetres and +4 % if they are 1/25 inch, and the loupe's pincushion distortion adds about +2.7 % on centre lines; only a physical measurement of the glass can settle it, so counts are threads per window side.

## 3. Collection process

- **Acquisition.** The author took the photos by mid-2019 on the working factory floor of one mill, through a commodity 10x pick glass pressed onto the fabric, with uncontrolled distance, tilt, focus and lighting; probably one phone. No capture records exist beyond the photos and their file metadata: the number of capture sessions and the phone model are not recorded. The label-free scene audit finds 44 distinct fabric patches (section 2).
- **Sampling.** Convenience sample of fabrics on the floor at the time; no selection rule was recorded.
- **Consent.** The author took the photos with the mill's agreement.
- **Timeframe.** Photos by mid-2019; window polygons 2024; scene audit, predictions and annotation October 2026.
- **Ethical review.** **TBD.**

## 4. Preprocessing, cleaning, labelling

- **Scene audit** (label-free): `camera_aug/provenance_scan.py` (container, JPEG tables, sizes, MD5, coarse and fine image correlation) and `annotation_tool/scenes.py` (crop NCC over the four 90-degree rotations, confirmed by a polygon-free whole-photo check) -> `data/scene_map.csv`, `data/scenes_check.csv`.
- **Rectification.** `common/rectify.py`: the window polygon is warped to an S x S crop, S = mean side length in px (no up-sampling), bilinear.
- **Human counts.** [docs/annotation_protocol.md](docs/annotation_protocol.md): the scene-aware tool in `annotation_tool/`, a screening test, the completeness rule, two annotators each counting all 76 representatives from scratch with no machine dots (machine dots appear only in four worked examples of the guide, scenes #27, #65, #93 and #110; 38 scenes show a count-free difficulty badge); delivered as the tool's JSON and CSV exports. **TBD:** each annotator's total time and pay (record whether it met or exceeded a fair local rate).
- **Raw data.** The original files are kept unchanged; every derived file records how it was made.

## 5. Uses

- **Intended.** Benchmarking per-window thread-count estimators on real pick-glass phone photos; capture-quality studies (resolution pairs, degradations, phone re-capture); testing quality gates and cascades; teaching.
- **Not intended.** Certifying fabric, rejecting a piece or cutting a weaver's pay on one estimate; monitoring individual weavers; any use against the mill's interests. One window is a screening measurement, not a standards-conformant determination (IS 1963, ISO 7211-2, ASTM D3775 average five or more places per piece).
- **Known limits.** One site, one glass, probably one phone; 76 scenes, 26 only at quarter resolution; colour-woven checks, satins, basket weaves, jacquards and knits are poorly covered; see [docs/open_problems.md](docs/open_problems.md).

## 6. Distribution

- **Code:** MIT ([LICENSE](LICENSE)), public on GitHub.
- **Data:** to be added, on an archive with a DOI (Zenodo or Hugging Face), under the licence to be announced in [DATA_LICENSE](DATA_LICENSE). The DOI will be added to the README and the paper.
- **Export controls / third-party IP.** Woven designs may be a client's property.

## 7. Maintenance

- **Contact:** GitHub issues of this repository.
- **Errata and versions:** corrections to labels or metadata are released as new versions with a changelog; earlier versions stay available; the frozen predictions and scorer are never changed ([CHECKSUMS.md](CHECKSUMS.md)).
- **Contributions:** new photos, further annotators' counts (or repeat counts) and calibrations of the glass are welcome ([docs/how_to_extend.md](docs/how_to_extend.md)).

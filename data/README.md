# data/

Dataset-level files of PickGlass (150 files, 76 scenes) that contain **no pixels**. Code and per-experiment results keep the original
project layout (`methods/`, `cnn/`, `vlm/`, ..., `paper/tables/`); the files here are the entry points.

| File | Rows | What |
|---|---|---|
| `manifest.json` | 150 images | the annotation tool's manifest: file name, size, window polygon, rectified size `rect_size`, scene fields (`scene_id`, `is_scene_rep`, `kind`, `scene_rot_k`, `scene_swap_hv`), the machine proposals of the frozen counter (`proposal`: dot positions on the two guide lines; never shown to the annotator), the qualitative `attention` badge fields, and the 20-scene agreement subset `iaa_subset` (drawn under an earlier plan; it has no special role in the paper). Identical to `annotation_tool/data/manifest.json`, which the scripts read |
| `scene_map.csv` | 150 | file id -> scene id, representative flag, kind, rotation / swap relative to the representative, NCC evidence (`cnn/scene_map.py`) |
| `scenes_to_annotate.txt` | 76 | the scene representatives and the files of each scene |
| `scenes_check.csv` | 74 | the polygon-free whole-photo check of every copy's rotation (`annotation_tool/scenes.py --check-images`) |
| `window_polygons.csv` | 150 | the hand-drawn window polygons (`x1..y4` exactly as stored in the original XML files, original pixels) and the canonical TL, TR, BR, BL corners used by `common/rectify.py`, the crop side `side_px`, and whether they equal the manifest polygon (150/150). `common/rectify.py` reads this file when the XML files are absent |
| `photo_provenance.json` | 150 | per file: container, size, bytes, MD5, JPEG tables and sampling, window side in px, px per inch; scene groups and full/quarter pairs (`camera_aug/provenance_scan.py`). No EXIF exists in any file. The MD5 values let anyone match the photos once they are released |
| `legacy13_hand_counts.csv` | 26 | the pick-glass hand counts of the 13 legacy crops (one row per count; `formula_derived` marks the three twill warps computed as a + 2b) |
| `predictions/` | | copies of every method's predictions on PickGlass (`pickglass150/`) and on legacy-13 (`legacy13/`); `predictions/INDEX.csv` lists the source path in this repo, what each file holds, its row count and SHA-256 |

Wide prediction files use the VLM convention of `paper/tools/score150.py --extra-wide`: one row per `image_id`,
columns `<method>__v` (vertical yarns = our `h`) and `<method>__h` (horizontal yarns = our `v`).
`ours_pg-lattice_and_spectral.csv` uses `h` / `v` directly (see its header).

## Not in this repository yet (released after review; see `DATASHEET.md`)

The 150 photos (`$PICKGLASS_NOTEBOOKS/final_image_set/`), the window-polygon XML files, the rectified crops
(`annotation_tool/data/crops*/`, `thumbs/`), the 13 legacy crops and the legacy hand-count sheet
(`$PICKGLASS_LEGACY_ROOT`), the preparation-notebook augmentations, all degraded copies and PickGlass-Stress renders
(`camera_aug/stress/`), the generative-enhancement outputs (`genai_sr/outputs/`), both annotators' counts with dot positions
and timing logs, and CNN weights fine-tuned on pseudo-labels of the real photos. They will be published on an
archive with a DOI under the licence in `DATA_LICENSE`; see `DATASHEET.md`. `common/paths.py` shows where the code
expects them (all locations are environment variables with repo-relative defaults).

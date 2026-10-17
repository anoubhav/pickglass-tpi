"""Where the code looks for data that is NOT in this repository (photos, legacy material, scratch space).

Every location is an environment variable with a default relative to the repository root, so a fresh clone
needs no editing.  The scripts read these variables directly (``os.environ.get(...) or <default>``); this
module only documents them in one place and offers ``python3 common/paths.py`` to print what is in effect.

Variable                  Default (repo-relative)     What it must contain
------------------------  --------------------------  ---------------------------------------------------------
PICKGLASS_NOTEBOOKS       data/photos                 final_image_set/{0..149}.jpeg (the 150 photos),
                                                      final_bbox_annotation_set/{0..149}.xml (window polygons),
                                                      augmented_images/, augmented_xml/ (augmentations of
                                                      the dataset notebook) and 2_Segmentation_Result_classical_CV.ipynb.
                                                      Released after review (DATASHEET.md).
                                                      Without the XML files, common/rectify.py reads the same
                                                      corners from data/window_polygons.csv.
PICKGLASS_LEGACY_ROOT     data/legacy2019             the earlier project's folder: "FR-final attempt/images/"
                                                      (the 20 legacy window crops a1..a10, r1..r10),
                                                      "Warp-Weft count.xlsx" (legacy hand counts) and TQA/
                                                      (the original legacy scripts). The 13 hand counts used in
                                                      the paper are in data/legacy13_hand_counts.csv.
PICKGLASS_SCRATCH         scratch                     large intermediate outputs (degraded images, synthetic
                                                      dev sets); created on demand, git-ignored.
PICKGLASS_GPU_HOST        gpu-server                  ssh alias of a CUDA machine (Molmo2, CountGD, LoRA, the
                                                      generative restorers, CNN cross-validation).
PICKGLASS_REMOTE_ROOT     pickglass                   project folder on that machine, relative to its $HOME.
PICKGLASS_PYTHON          python3 on PATH             interpreter for the shell wrappers (label_day.sh, the
                                                      camera_aug chains); needs requirements.txt.
OPENROUTER_API_KEY        (unset)                     API key for the VLM baselines (vlm/openrouter_client.py,
                                                      genai_sr/ai_enhance_openrouter.py). Read from the
                                                      environment only; never commit it.

The annotation tool's own data live in annotation_tool/data/ (manifest.json for the 150 photos; crops/,
crops_enh/, thumbs/ are written by annotation_tool/prepare.py once the photos are available) and
annotation_tool/demo/data/ (three synthetic practice fabrics, in the repository).
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(var: str, *default_parts: str) -> str:
    v = os.environ.get(var)
    return os.path.abspath(os.path.expanduser(v)) if v else os.path.join(REPO, *default_parts)


NOTEBOOKS = _p("PICKGLASS_NOTEBOOKS", "data", "photos")
PHOTOS_DIR = os.path.join(NOTEBOOKS, "final_image_set")
WINDOWS_DIR = os.path.join(NOTEBOOKS, "final_bbox_annotation_set")
LEGACY_ROOT = _p("PICKGLASS_LEGACY_ROOT", "data", "legacy2019")
LEGACY_CROPS = os.path.join(LEGACY_ROOT, "FR-final attempt", "images")
LEGACY_XLSX = os.path.join(LEGACY_ROOT, "Warp-Weft count.xlsx")
SCRATCH = _p("PICKGLASS_SCRATCH", "scratch")
GPU_HOST = os.environ.get("PICKGLASS_GPU_HOST", "gpu-server")
REMOTE_ROOT = os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass")
WINDOW_POLYGONS_CSV = os.path.join(REPO, "data", "window_polygons.csv")
MANIFEST = os.path.join(REPO, "annotation_tool", "data", "manifest.json")


def report() -> str:
    rows = [("PICKGLASS_NOTEBOOKS", NOTEBOOKS, os.path.isdir(PHOTOS_DIR)),
            ("PICKGLASS_LEGACY_ROOT", LEGACY_ROOT, os.path.isdir(LEGACY_CROPS)),
            ("PICKGLASS_SCRATCH", SCRATCH, os.path.isdir(SCRATCH)),
            ("PICKGLASS_GPU_HOST", GPU_HOST, None),
            ("PICKGLASS_REMOTE_ROOT", REMOTE_ROOT, None),
            ("OPENROUTER_API_KEY", "(set)" if os.environ.get("OPENROUTER_API_KEY") else "(unset)", None)]
    out = []
    for var, val, ok in rows:
        flag = "" if ok is None else ("  [found]" if ok else "  [missing: released after review / not created yet]")
        out.append(f"{var:22s} {val}{flag}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())

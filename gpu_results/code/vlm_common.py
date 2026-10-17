"""Shared helpers for the GPU (bf16, transformers) Molmo2-8B run.

Nothing in here re-implements the Mac harness: the prompt set, the output parsers,
the image-view helpers and the legacy GT parser are all imported or extracted
VERBATIM from the Mac files copied to code/mac_vlm/ (prompts.py, parse_output.py,
data_vlm.py, gt_legacy.py) and code/run_pipeline.py (the legacy harness' load_gt):

  * prompts.PROMPTS / prompts.get           -> imported as-is
  * parse_output.parse_counts/parse_points  -> imported as-is
  * data_vlm.{quarter_patch, guide_line_view, resize_square, resize_max}
        -> function definitions extracted with `ast` (data_vlm imports the Mac-only
           common/rectify.py at module level, so it cannot be imported directly)
  * gt_legacy.legacy_gt -> imported as-is; only its two path constants are pointed at
        the local copies of run_pipeline.py and Warp-Weft count.xlsx
"""
from __future__ import annotations

import ast
import os
import sys

import cv2
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)                     # gpu_results (on the GPU server: the copy that holds code/)
MAC = os.path.join(CODE, "mac_vlm")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, MAC)

import prompts as PR                              # noqa: E402  (verbatim Mac prompt set)
from parse_output import parse_counts, parse_points  # noqa: E402  (verbatim Mac parsers)

LEGACY_NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
LEGACY_CROP_DIR = os.path.join(DATA, "legacy_crops")   # = Mac vlm/crops/legacy/{name}.png (frame-trimmed)
CROPS150_DIR = os.path.join(DATA, "crops150")          # = Mac annotation_tool/data/crops/{i}.jpg
MANIFEST = os.path.join(DATA, "manifest.json")
XLSX = os.path.join(DATA, "Warp-Weft count.xlsx")
RUN_PIPELINE = os.path.join(CODE, "run_pipeline.py")


def _extract(path, names, ns):
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names), f"{path}: missing {set(names) - {n.name for n in keep}}"
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


_V = _extract(os.path.join(MAC, "data_vlm.py"),
              ["quarter_patch", "guide_line_view", "resize_square", "resize_max"],
              {"cv2": cv2, "np": np, "Optional": None})
quarter_patch = _V["quarter_patch"]
guide_line_view = _V["guide_line_view"]
resize_square = _V["resize_square"]
resize_max = _V["resize_max"]


def legacy_gt(usable_only: bool = True) -> dict:
    import gt_legacy as G                          # verbatim Mac module
    G.PIPELINE = RUN_PIPELINE
    G.GT_XLSX = XLSX
    G._NS = None
    return G.legacy_gt(usable_only=usable_only)

"""View helpers shared by training and inference: extracted VERBATIM (via ast) from the project's
vlm/data_vlm.py, exactly as gpu_results/code/vlm_common.py does for the Molmo2-8B bf16 run, so the
fine-tuned model sees views built by the same code as every other Molmo2 row in the paper."""
import ast
import os

import cv2
import numpy as np

_CANDIDATES = [os.environ.get("DATA_VLM_PY", ""),
               os.path.expanduser(("~/" + os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass") + "/code/mac_vlm/data_vlm.py")),
               os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "vlm", "data_vlm.py")]
DATA_VLM = next(p for p in _CANDIDATES if p and os.path.exists(p))
_src = open(DATA_VLM).read()
_keep = [n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef)
         and n.name in ("quarter_patch", "guide_line_view", "resize_square")]
assert len(_keep) == 3
_ns = {"cv2": cv2, "np": np, "Optional": None}
exec(compile(ast.Module(body=_keep, type_ignores=[]), DATA_VLM, "exec"), _ns)
quarter_patch = _ns["quarter_patch"]
guide_line_view = _ns["guide_line_view"]
resize_square = _ns["resize_square"]
SIDE = 644


def view_image(crop, rect, guide=None, side=SIDE):
    """rect=(x0,y0,w,h) slice of the crop -> side x side view (+ red guide line 'h' or 'v')."""
    x0, y0, w, h = rect
    v = resize_square(crop[y0:y0 + h, x0:x0 + w], side)
    return guide_line_view(v, guide) if guide else v

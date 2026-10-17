"""Legacy ground truth (hand counts) for the VLM evaluation.

The parser is NOT rewritten here: `load_gt`, `_parse_count` and `_parse_weave` are
taken verbatim from the faithful legacy harness (pipeline/run_pipeline.py) by
extracting just those three function definitions with `ast`, so that the harness'
heavy top-level imports (scipy.ndimage.interpolation, seg_final, ...) are not needed.

    from gt_legacy import legacy_gt
    gt = legacy_gt()     # {'a1': {'weft':..,'warp':..,'weave':'1/1',...}, ...}  usable ones only

Conventions (as in the harness): text formulas 'a + b*2 = N' -> N; ranges '110-120'
-> midpoint (weft_lo/weft_hi kept, flagged as is_range); weave Excel dates
YYYY-01-01 -> '1/1' (plain), YYYY-02-01 -> '2/1' (twill); a trailing '*' on the image
name -> unclear=True.
"""
from __future__ import annotations

import ast
import datetime as _dt
import os
import re

import openpyxl

PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods", "legacy_2019", "run_pipeline.py")
GT_XLSX = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "Warp-Weft count.xlsx")
WEAVE_NAME = {"1/1": "plain", "2/1": "twill", "1/2": "twill"}

_NS = None


def _harness_ns():
    global _NS
    if _NS is None:
        src = open(PIPELINE).read()
        tree = ast.parse(src)
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_parse_count", "_parse_weave", "load_gt")]
        assert len(keep) == 3, "run_pipeline.py changed: GT functions not found"
        mod = ast.Module(body=keep, type_ignores=[])
        ns = {"re": re, "os": os, "openpyxl": openpyxl, "_dt": _dt}
        exec(compile(mod, PIPELINE, "exec"), ns)
        _NS = ns
    return _NS


def legacy_gt(usable_only: bool = True) -> dict:
    gt, _ = _harness_ns()["load_gt"](GT_XLSX)
    out = {}
    for k, g in gt.items():
        if usable_only and not g["usable"]:
            continue
        g = dict(g)
        g["weave_name"] = WEAVE_NAME.get(g["weave"], g["weave"] or None)
        g["is_range"] = (g["weft_lo"] != g["weft_hi"]) or (g["warp_lo"] != g["warp_hi"])
        out[k] = g
    return out


if __name__ == "__main__":
    gt = legacy_gt(usable_only=False)
    for k, g in gt.items():
        print(k, {x: g[x] for x in ("usable", "warp", "weft", "weave", "weave_name", "unclear", "is_range",
                                    "warp_raw", "weft_raw")})
    print(sum(g["usable"] for g in gt.values()), "usable of", len(gt))

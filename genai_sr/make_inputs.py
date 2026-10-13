"""make_inputs.py -- reference copies + degraded LR inputs (README section 2-3), built once on the Mac.

    <venv2>/bin/python make_inputs.py

Reference images are BYTE copies (no re-encode) of annotation_tool/data/crops/{i}.jpg (R-HR) and
vlm/crops/legacy/{name}.png (R-LEG).  LR inputs use methods/degrade.py unchanged:
D4 = downscale x4 (INTER_AREA), D8 = downscale x8, D4J = jpeg(downscale x4, q50); saved as PNG.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import cv2
import pandas as pd

import gsr_common as G

sys.path.insert(0, os.path.join(G.WVAQ, "methods"))
import degrade as D  # noqa: E402

SRC_HR = os.path.join(G.WVAQ, "annotation_tool", "data", "crops")
SRC_LEG = os.path.join(G.WVAQ, "vlm", "crops", "legacy")


def degrade(img, cond):
    if cond == "D4":
        return D.downscale(img, 4)
    if cond == "D8":
        return D.downscale(img, 8)
    if cond == "D4J":
        return D.jpeg(D.downscale(img, 4), 50)
    raise ValueError(cond)


def main():
    rows = []
    items = [("R-HR", i, os.path.join(SRC_HR, f"{i}.jpg")) for i in G.HR_IDS] + \
            [("R-LEG", n, os.path.join(SRC_LEG, f"{n}.png")) for n in G.LEG_NAMES]
    for set_name, ident, src in items:
        key = G.key_of(set_name, ident)
        rp = G.ref_path(key)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        if not os.path.exists(rp):
            shutil.copyfile(src, rp)
        assert G.md5(rp) == G.md5(src)
        img = cv2.imread(rp, cv2.IMREAD_COLOR)
        S = img.shape[0]
        assert img.shape[0] == img.shape[1], (key, img.shape)
        for cond in G.CONDS:
            lp = G.lr_path(cond, key)
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            lr = degrade(img, cond)
            cv2.imwrite(lp, lr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            rows.append(dict(key=key, set=set_name, ident=str(ident), S=S, cond=cond, lr_w=lr.shape[1], lr_h=lr.shape[0],
                             lr_path=os.path.relpath(lp, G.GSR), lr_md5=G.md5(lp),
                             ref_path=os.path.relpath(rp, G.GSR), ref_md5=G.md5(rp), src=os.path.relpath(src, G.WVAQ)))
    df = pd.DataFrame(rows)
    df.to_csv(G.MANIFEST, index=False)
    G.write_json(os.path.join(G.INPUTS, "meta.json"), dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"), degrade_py_md5=G.md5(D.__file__), cv2=cv2.__version__,
        n_refs=len(items), n_lr=len(df), conds=G.CONDS, hr_ids=G.HR_IDS, leg=G.LEG_NAMES))
    print(df.groupby(["set", "cond"]).agg(n=("key", "size"), lr_w_min=("lr_w", "min"), lr_w_max=("lr_w", "max")))


if __name__ == "__main__":
    main()

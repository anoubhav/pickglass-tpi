"""finalize_raw.py -- resize a restorer's raw outputs (outputs/_<name>_raw/<cond>/<key>.png) to S x S
(INTER_CUBIC if up, INTER_AREA if down) into outputs/<name>/<cond>/<key>.png; logs raw sizes.

    python code/finalize_raw.py invsr
"""
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402

name = sys.argv[1]
man = pd.read_csv(G.MANIFEST)
rows = []
for r in man.itertuples():
    raw = os.path.join(G.OUT_DIR, f"_{name}_raw", r.cond, r.key + ".png")
    if not os.path.exists(raw):
        rows.append(dict(key=r.key, cond=r.cond, missing=True))
        continue
    img = cv2.imread(raw, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    out = img if (h, w) == (r.S, r.S) else cv2.resize(img, (r.S, r.S), interpolation=cv2.INTER_CUBIC if h < r.S else cv2.INTER_AREA)
    op = G.out_path(name, r.cond, r.key)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    cv2.imwrite(op, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    rows.append(dict(key=r.key, cond=r.cond, raw_w=w, raw_h=h, S=r.S, missing=False))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(G.GSR, "logs", f"{name}_raw_sizes.csv"), index=False)
print(name, "finalized", int((~df.missing).sum()), "missing", int(df.missing.sum()))

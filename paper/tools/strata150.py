"""Label-free capture-quality descriptors for the 150 PickGlass photos (Job 5; moved into paper/tools by Job 6).
Reads only the manifest polygons and the rectified crops; writes paper/tables/strata150.csv.
Columns: S (mean polygon side, px per inch), side_ratio, max_angle_dev (deg from 90), sat240/sat250 (share of
window pixels with all channels >= 240 / 250), dark20 (grey <= 20). These define the capture-quality strata that
were fixed before labelling (supplement, Task definition). Run: <venv2>/bin/python tools/strata150.py"""
import json, math, os, sys
import numpy as np, cv2, pandas as pd
ROOT = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "annotation_tool")
m = json.load(open(os.path.join(ROOT, "data/manifest.json")))
items = m["images"] if "images" in m else None
if items is None:
    for k, v in m.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "polygon" in v[0]:
            items = v; break
rows = []
for it in items:
    P = np.array(it["polygon"], float)
    sides = [np.linalg.norm(P[(i+1) % 4] - P[i]) for i in range(4)]
    angs = []
    for i in range(4):
        a = P[i-1] - P[i]; b = P[(i+1) % 4] - P[i]
        angs.append(math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    S = float(np.mean(sides))
    img = cv2.imread(os.path.join(ROOT, it["crop"]))
    mn = img.min(axis=2)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows.append(dict(id=it["id"], S=S, side_ratio=max(sides)/min(sides),
                     max_angle_dev=max(abs(a-90) for a in angs),
                     sat250=float((mn >= 250).mean()), sat240=float((mn >= 240).mean()),
                     dark20=float((lum <= 20).mean()), crop_side=img.shape[0],
                     orig_w=it.get("orig_w"), orig_h=it.get("orig_h")))
df = pd.DataFrame(rows).sort_values("id")
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables", "strata150.csv"),
          index=False)
print(df.describe().T.to_string())
print("S classes:", (df.S < 1000).sum(), (df.S >= 1000).sum())
for t in [0.001, 0.01, 0.05, 0.10, 0.20]:
    print(f"sat250 > {t:.3f}: {(df.sat250 > t).sum()}   sat240 > {t:.3f}: {(df.sat240 > t).sum()}")
for t in [1.02, 1.05, 1.08, 1.10]:
    print(f"side_ratio > {t}: {(df.side_ratio > t).sum()}")
for t in [1, 2, 3, 5]:
    print(f"max_angle_dev > {t} deg: {(df.max_angle_dev > t).sum()}")

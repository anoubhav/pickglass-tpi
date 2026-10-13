"""make_viewing.py -- 768-px JPEG (q92, INTER_AREA) viewing copies of every restored output + md5 manifest of the
full-resolution PNGs (which stay on gpu-server).  -> viewing/<restorer>/<cond>/<key>.jpg, results/outputs_manifest.csv"""
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402

rows = []
for rest in [r for r in G.RESTORERS if r != "none"]:
    for key, s, cond, r, path, S in G.items(rest):
        vp = os.path.join(G.GSR, "viewing", rest, cond, key + ".jpg")
        if not os.path.exists(vp):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            os.makedirs(os.path.dirname(vp), exist_ok=True)
            cv2.imwrite(vp, cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(dict(restorer=rest, cond=cond, key=key, set=s, S=S, png=os.path.relpath(path, G.GSR),
                         png_md5=G.md5(path), png_bytes=os.path.getsize(path)))
pd.DataFrame(rows).to_csv(os.path.join(G.GSR, "results", "outputs_manifest.csv"), index=False)
print(len(rows))

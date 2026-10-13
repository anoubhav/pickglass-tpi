"""Check 1c: large side-by-side for the samples still ambiguous by eye: AUG | candidate A | candidate B
(CLAHE grey, low-passed sigma 1.5 at 330 px so glare/defect layout and weave are both visible)."""
import os, sys, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); AUG = os.path.dirname(HERE); ROOT = os.path.dirname(AUG)
TOOL = os.path.join(ROOT, "annotation_tool", "data", "crops"); CROPS = os.path.join(AUG, "crops")
T = 326
def prep(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (T, T), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(3.0, (4, 4)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def lab(im, s):
    im = im.copy(); cv2.putText(im, s, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3); cv2.putText(im, s, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1); return im
rows = []
for spec in sys.argv[2:]:
    aid, ja, jb = spec.split(":")
    src = cv2.imread(os.path.join(TOOL, aid.split("_aug")[0] + ".jpg")); aug = cv2.imread(os.path.join(CROPS, aid + ".jpg"))
    cells = [lab(prep(aug), aid + " AUG")] + [lab(prep(np.ascontiguousarray(np.rot90(src, int(j) // 90))), "rot" + j) for j in (ja, jb)]
    rows.append(np.hstack([np.pad(c, ((3, 3), (3, 3), (0, 0)), constant_values=255) for c in cells]))
cv2.imwrite(os.path.join(HERE, "sheets", sys.argv[1]), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

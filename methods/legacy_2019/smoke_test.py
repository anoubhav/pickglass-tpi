"""Smoke test of seg_final.segment() on 3 annotated images (final_image_set) and
3 FR-final-attempt images; writes visualisations + crops to seg_smoke/."""
import contextlib, io, json, os, sys, math
import numpy as np, cv2
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seg_final
OUT = os.path.join(HERE, "seg_smoke"); os.makedirs(OUT, exist_ok=True)
nb = json.load(open(os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "2_Segmentation_Result_classical_CV.ipynb")))
ns = dict(np=np, cv2=cv2, math=math, os=os, Polygon=Polygon)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; ns["plt"] = plt
for idx in (28, 29): exec("".join(nb["cells"][idx]["source"]), ns)
NBR = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "")
FR = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "legacy2019")), "FR-final attempt", "images", "")
cases = [(NBR + "final_image_set/63.jpeg", NBR + "final_bbox_annotation_set/63.xml"),
         (NBR + "final_image_set/89.jpeg", NBR + "final_bbox_annotation_set/89.xml"),
         (NBR + "final_image_set/70.jpeg", NBR + "final_bbox_annotation_set/70.xml"),
         (FR + "a1.jpg", None), (FR + "r6 .jpg", None), (FR + "r10.jpg", None)]
summary = []
for img_path, xml in cases:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        crop, box, fb = seg_final.segment(img_path, cv_seed=0)
    gt = ns["parse_annotation"](xml)[1] if xml else None
    iou = float(ns["get_IOU"](gt, box)) if gt is not None else None
    stem = os.path.splitext(os.path.basename(img_path))[0].strip().replace(" ", "_")
    tag = ("nb_" if xml else "fr_") + stem
    seg_final._visualise(img_path, box, fb, crop, os.path.join(OUT, tag + "_vis.jpg"), gt_box=gt)
    if crop is not None: cv2.imwrite(os.path.join(OUT, tag + "_crop.png"), crop)
    h, w = cv2.imread(img_path).shape[:2]
    rec = dict(image=img_path, image_hw=[h, w], used_fallback=bool(fb), box=box.tolist(),
               crop_hw=None if crop is None else list(crop.shape[:2]),
               box_area_frac=round(float(cv2.contourArea(box.astype(np.float32))) / (h * w), 3),
               iou_vs_gt=None if iou is None else round(iou, 4), stdout=buf.getvalue().strip())
    summary.append(rec); print(json.dumps(rec))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

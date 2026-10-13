"""Repeat final_algorithm on the 4 images whose IoU differed from the notebook run,
in one process (cv2 RNG advances between calls), to show k-means run-to-run variation."""
import contextlib, io, json, os, sys, math
import numpy as np, cv2
from shapely.geometry import Polygon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_final
nb = json.load(open(os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "2_Segmentation_Result_classical_CV.ipynb")))
ns = dict(np=np, cv2=cv2, math=math, os=os, Polygon=Polygon)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; ns["plt"] = plt
for idx in (28, 29): exec("".join(nb["cells"][idx]["source"]), ns)
root = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "")
for name in ["89", "148", "15", "90"]:
    gt = ns["parse_annotation"](root + f"final_bbox_annotation_set/{name}.xml")[1]
    res = []
    for r in range(8):
        with contextlib.redirect_stdout(io.StringIO()):
            _, box, st = seg_final.final_algorithm(root + f"final_image_set/{name}.jpeg")
        res.append((round(ns["get_IOU"](gt, box), 4), st[0]))
    print(name, res)

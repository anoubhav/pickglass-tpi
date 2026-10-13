"""Re-run the notebook's 150-set evaluation (cells[39]) with seg_final.final_algorithm and
the notebook's own evaluation code (cells[28], cells[29], exec'd from the .ipynb JSON),
then compare per-image IoUs with the IoUs stored in cells[39]'s output."""
import ast, contextlib, io, json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, cv2, math
from shapely.geometry import Polygon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_final

NB = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "2_Segmentation_Result_classical_CV.ipynb")
nb = json.load(open(NB))
ns = dict(np=np, cv2=cv2, math=math, os=os, plt=plt, Polygon=Polygon)
for idx in (28, 29):
    exec("".join(nb["cells"][idx]["source"]), ns)
stored = ast.literal_eval("".join(nb["cells"][39]["outputs"][1]["data"]["text/plain"]))

os.chdir((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")))  # same relative paths as the notebook (read-only use)
dataset, annotations = ns["create_dataset"]("final_image_set/", "final_bbox_annotation_set/")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
alg = seg_final.final_algorithm
if seed is not None:
    def alg(p, _s=seed):
        cv2.setRNGSeed(_s); return seg_final.final_algorithm(p)
t = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m = ns["evaluate_algorithm"](dataset[:150], annotations[:150], alg, display=False)
print("cv2", cv2.__version__, "numpy", np.__version__, "cv_seed", seed, "time %.1fs" % (time.time() - t))
print("fallback messages printed:", buf.getvalue().count("Fallback box used") + buf.getvalue().count("Using fallback box"))
a, b = np.array(m["IoU Scores"]), np.array(stored["IoU Scores"])
print("reproduced Mean IoU %.6f  (notebook %.6f)" % (m["Mean IoU"], stored["Mean IoU"]))
print("reproduced Fallback %d Valid %d Invalid %d (notebook Fallback %d Valid %d Invalid %d)" % (
    m["Fallback Count"], m["Valid Count"], m["Invalid Count"], stored["Fallback Count"], stored["Valid Count"], stored["Invalid Count"]))
print("Mean BF1 %.6f (notebook %.6f)" % (m["Mean BF1"], stored["Mean BF1"]))
print("acc@thr", list(m.values())[4], "notebook", list(stored.values())[4])
if len(a) == len(b):
    d = np.abs(a - b)
    print("per-image |dIoU|: exact(<1e-9)=%d, <1e-3=%d, max=%.4f" % ((d < 1e-9).sum(), (d < 1e-3).sum(), d.max()))
    for i in np.where(d >= 1e-3)[0]:
        print("  differs:", dataset[i], "repro %.4f notebook %.4f" % (a[i], b[i]))

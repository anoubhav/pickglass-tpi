import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "cnn"))
import numpy as np, torch
import cnn_common as C, evaluate as E
dev = C.get_device("auto")
model = C.load_ckpt(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "cnn", "runs", "synth_r18_v2mix_aniso", "model.pt"), dev)
gts = E.load_legacy_gt()
# warm-up
C.predict_crop(model, E.legacy_crop(gts[0], "vlm"), dev, Ps=C.P_TEST, flips=True)
ts = []
for g in gts:
    t0 = time.perf_counter()
    crop = E.legacy_crop(g, "vlm")  # includes PNG read + cap to 1024
    o = C.predict_crop(model, crop, dev, Ps=C.P_TEST, flips=True)
    if dev.type == "mps": torch.mps.synchronize()
    ts.append(time.perf_counter() - t0)
res = dict(hw=f"M2 Max {dev.type.upper()} (PyTorch), TTA 3 scales x 3 flips, median", median_s=float(np.median(ts)),
           min_s=float(np.min(ts)), max_s=float(np.max(ts)), n=len(ts), device=str(dev))
print(res)
json.dump(res, open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "cnn", "runs", "synth_r18_v2mix_aniso", "eval_vlmcrops", "latency.json"), "w"), indent=1)

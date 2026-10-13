"""set C (seeds 992000..): plain / 2-1 twill with strongly unbalanced densities (poplin-like, ratio 1.6-2.4)."""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."), "cnn"))
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
import synth
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthC")
def work(seed):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    p = synth.sample_params(rng, seed)
    p.weave = str(rng.choice(["plain", "plain", "plain", "twill21", "twill22"]))
    lo = float(np.exp(rng.uniform(np.log(35), np.log(90))))
    hi = lo * float(rng.uniform(1.6, 2.4))
    p.epi, p.ppi = (hi, lo) if rng.random() < 0.5 else (lo, hi)
    p.r_eff = float(max(2.6 * max(p.epi, p.ppi), p.r_eff))
    p.r_sensor = max(p.r_sensor, p.r_eff)
    _, crop, lab = synth.generate(seed, params=p, frame_margin=0.1)
    fn = f"c{seed:07d}.jpg"; cv2.imwrite(os.path.join(OUT, fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lab["file"] = fn; lab["set"] = "C"; lab["corners"] = None
    return lab
if __name__ == "__main__":
    n = int(sys.argv[1])
    with Pool(9) as pool: labs = pool.map(work, [992000 + k for k in range(n)])
    pd.DataFrame(labs).to_csv(os.path.join(OUT, "labels.csv"), index=False); print("done", n)

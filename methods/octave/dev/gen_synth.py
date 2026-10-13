"""Fresh synthetic renders for the octave study (seeds >= 990000; never used before).
set A: seeds 990000.. default v2 generator (WEAVE_PROBS as shipped)
set B: seeds 991000.. extended weaves: + 'oxford' (ends work in pairs), 'oxfordw' (picks in pairs),
       more basket22 and plain with stripes/checks  -> structures with a strong 2-yarn period.
"""
import os, sys, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."), "cnn"))
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
import synth

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth")
_orig_wm = synth.weave_matrix


def wm(name, rng):
    if name == "oxford":
        rng.choice([-1, 1])
        i, j = np.mgrid[0:2, 0:4]
        return (((i + j // 2) % 2) == 0).astype(np.uint8)
    if name == "oxfordw":
        rng.choice([-1, 1])
        i, j = np.mgrid[0:4, 0:2]
        return (((i // 2 + j) % 2) == 0).astype(np.uint8)
    return _orig_wm(name, rng)


EXT = {"plain": 0.30, "oxford": 0.17, "oxfordw": 0.13, "basket22": 0.14, "twill21": 0.06, "twill22": 0.06,
       "twill31": 0.04, "satin5": 0.04, "herringbone": 0.03, "sateen5": 0.03}


def _work(a):
    seed, setname = a
    cv2.setNumThreads(1)
    if setname == "B":
        synth.WEAVE_PROBS = dict(EXT)
        synth.COARSE.update({"oxford": "other", "oxfordw": "other"})
        synth.weave_matrix = wm
    _, crop, lab = synth.generate(seed, frame_margin=0.1)
    fn = f"s{seed:07d}.jpg"
    cv2.imwrite(os.path.join(OUT, fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lab["file"] = fn
    lab["set"] = setname
    lab["corners"] = None
    return lab


if __name__ == "__main__":
    nA, nB = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(990000 + k, "A") for k in range(nA)] + [(991000 + k, "B") for k in range(nB)]
    t0 = time.time()
    with Pool(8) as p:
        labs = p.map(_work, jobs, chunksize=4)
    pd.DataFrame(labs).to_csv(os.path.join(OUT, "labels.csv"), index=False)
    print("done", len(labs), time.time() - t0)

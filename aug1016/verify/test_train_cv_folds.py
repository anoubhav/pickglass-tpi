"""Verification check 4: cnn/train_cv.py --extra-aug1016 never trains on an augmentation in the fold
that tests its source (or any image of its source's group).

Runs the REAL train_cv.main() (folds, grouping, load_aug1016_items, the per-fold training-set
assembly) on a dummy labels CSV, with only the expensive parts stubbed:
    finetune()      -> records the exact train_items list it is given, returns None
    evaluate_fold() -> records the test recs, returns the ground truth as prediction
Scenarios
    A  150 images, groups = 'g{i % 37}' (multi-image groups spread over the index range), 5 folds,
       labels propagated on the fly
    B  same, but labels from a propagate_labels.py output CSV (--aug1016-labels) built from a
       DIFFERENT dummy labelling (to prove those labels, not the run's, are used, and still no leak)
    C  --max-images 40, 3 folds, no group column -> auto-grouping (ImageNet embeddings)
    D  no flag -> no aug crop anywhere (defaults unchanged)
Assertions per fold k
    1  every aug item in train(k): fold[source] != k and group[source] not in test groups of k
    2  test set of fold k contains no aug crop (only originals)
    3  the set of aug crops in train(k) == {usable aug of a labelled source with fold[source] != k}
    4  aug labels == source labels with h/v swapped iff manifest k in {90, 270}
Run with the torch venv:  <venv_cnn>/bin/python test_train_cv_folds.py      exit 0 = all PASS
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
CNN = os.path.join(ROOT, "cnn")
OUT = os.path.join(HERE, "cv_test")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CNN)
os.chdir(CNN)
import train_cv as TC  # noqa: E402

FAILS = []
MAN = pd.read_csv(os.path.join(AUG, "manifest.csv"))
USABLE = MAN[MAN.usable]
CROP2ID = {os.path.join(AUG, c): a for c, a in zip(USABLE.crop, USABLE.aug_id)}
KDEG = dict(zip(MAN.aug_id, MAN.k_deg))


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def dummy_labels(path, seed, groups=True, n=150):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"image": range(n), "h_count": rng.uniform(20, 140, n).round(1),
                       "v_count": rng.uniform(15, 120, n).round(1),
                       "weave": rng.choice(["plain", "twill", "satin", ""], n)})
    if groups:
        df["group"] = [f"g{i % 37}" for i in range(n)]
    df["uncertain"] = (rng.uniform(0, 1, n) < 0.1).astype(int)
    df.to_csv(path, index=False)
    return df


def run(tag, argv):
    calls = {"train": [], "test": []}

    def fake_finetune(init, backbone, train_items, *a, **k):
        calls["train"].append(list(train_items))
        return None

    def fake_eval(model, te_recs, te_crops, *a, **k):
        calls["test"].append(list(te_recs))
        return [dict(pred_h=r["h"], pred_v=r["v"], h_iqr=0, v_iqr=0, pred_weave=0) for r in te_recs]

    TC.finetune, TC.evaluate_fold = fake_finetune, fake_eval
    out = os.path.join(OUT, tag)
    sys.argv = ["train_cv.py", "--experiments", "imagenet_ft", "--backbones", "resnet18", "--workers", "0",
                "--device", "cpu", "--out", out] + argv
    TC.main()
    folds = pd.read_csv(os.path.join(out, "folds.csv"))
    return calls, folds, out


def verify(tag, calls, folds, label_src, expect_aug=True):
    """label_src: DataFrame image(int)->h_count, v_count of the labels the aug crops must carry."""
    fold_of = {int(i): int(f) for i, f in zip(folds.image, folds.fold)}
    group_of = {int(i): str(g) for i, g in zip(folds.image, folds.group)}
    nf = int(folds.fold.max()) + 1
    lab = {int(r.image): (float(r.h_count), float(r.v_count)) for r in label_src.itertuples()}
    ok1 = ok2 = ok3 = ok4 = True
    n_aug_total = 0
    detail = []
    for k in range(nf):
        tr, te = calls["train"][k], calls["test"][k]
        te_ids = {int(r["idx"]) for r in te}
        te_groups = {group_of[i] for i in te_ids}
        ok2 &= all(r["idx"] is not None and r["path"] not in CROP2ID for r in te)
        aug_items = [it for it in tr if it.path is not None and it.path in CROP2ID]
        n_aug_total += len(aug_items)
        got = set()
        for it in aug_items:
            aid = CROP2ID[it.path]
            src = int(aid.split("_aug")[0])
            got.add(aid)
            if fold_of[src] == k or group_of[src] in te_groups or src in te_ids:
                ok1 = False
                detail.append(f"LEAK fold {k}: {aid}")
            h, v = lab[src]
            eh, ev = (v, h) if KDEG[aid] in (90, 270) else (h, v)
            if abs(it.h - eh) > 1e-6 or abs(it.v - ev) > 1e-6:
                ok4 = False
                detail.append(f"label {aid}: {it.h},{it.v} vs {eh},{ev}")
        exp = {a for a, s in zip(USABLE.aug_id, USABLE.source_id) if s in fold_of and fold_of[s] != k}
        if got != exp:
            ok3 = False
            detail.append(f"fold {k}: got {len(got)} expected {len(exp)}")
        print(f"   [{tag}] fold {k}: test {len(te)} originals ({len(te_groups)} groups); train "
              f"{len(tr) - len(aug_items)} originals + {len(aug_items)} aug crops")
    if not expect_aug:
        check(f"{tag}: no aug crops without the flag", n_aug_total == 0, f"(n={n_aug_total})")
        return
    check(f"{tag}: no aug crop trained in its source's / source group's test fold", ok1, "; ".join(detail[:5]))
    check(f"{tag}: test folds contain only originals", ok2)
    check(f"{tag}: train(k) aug set == usable augs of labelled sources outside fold k", ok3, "; ".join(detail[:5]))
    check(f"{tag}: aug labels == source labels, h/v swapped iff k in 90/270", ok4, "; ".join(detail[:5]))


def main():
    # A: on-the-fly propagation, provided groups
    pA = os.path.join(OUT, "dummy_A.csv")
    dA = dummy_labels(pA, 1)
    calls, folds, _ = run("A", ["--labels", pA, "--folds", "5", "--extra-aug1016"])
    verify("A", calls, folds, dA)

    # B: precomputed propagate_labels.py output from a DIFFERENT labelling
    import importlib.util
    spec = importlib.util.spec_from_file_location("pl", os.path.join(AUG, "propagate_labels.py"))
    PL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PL)
    pB = os.path.join(OUT, "dummy_B_other.csv")
    dB = dummy_labels(pB, 2)
    labB = os.path.join(OUT, "labels_aug1016_B.csv")
    PL.run([pB], out=labB, log=lambda *x: None)
    calls, folds, _ = run("B", ["--labels", pA, "--folds", "5", "--extra-aug1016", "--aug1016-labels", labB])
    verify("B", calls, folds, dB)

    # C: subsample + auto-grouping
    pC = os.path.join(OUT, "dummy_C.csv")
    dC = dummy_labels(pC, 3, groups=False)
    calls, folds, _ = run("C", ["--labels", pC, "--folds", "3", "--max-images", "40", "--extra-aug1016"])
    verify("C", calls, folds, dC.iloc[:40])

    # D: flag off
    calls, folds, _ = run("D", ["--labels", pA, "--folds", "5"])
    verify("D", calls, folds, dA, expect_aug=False)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

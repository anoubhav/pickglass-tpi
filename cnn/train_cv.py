#!/usr/bin/env python
"""Full 5-fold cross-validation of the learned thread-density baselines.

ONE COMMAND once the author's annotations exist:

    python train_cv.py --labels annotations.csv

Experiments (each for --backbones, default resnet18 and efficientnet_b0)
    constant_mean  : predicts the training-fold mean count (reference)
    imagenet_ft    : ImageNet-pretrained backbone fine-tuned on the real
                     training folds (tile regression + weave head)
    synth_ft       : ImageNet -> synthetic pre-training -> real fine-tuning
                     (sim-to-real)
    synth_zeroshot : synthetic-only model, no real labels (evaluated on the
                     same test folds for a like-for-like comparison)

The CSV exported by the annotation tool (annotation_tool/, "Download" ->
annotations_<name>_<time>.csv: columns id, h_count, v_count, weave, done,
nonwoven, confidence, ...) is accepted AS IS; several files / annotators can
be concatenated (counts are averaged per image, see below).

Annotation CSV schema (one row per image; extra columns are ignored)
    image|id   REQUIRED  index 0..149 of final_image_set/{i}.jpeg, or a file
                         name/path of an image ('12.jpeg', '/abs/x.jpg')
    h_count, v_count     threads per inch crossed walking horizontally /
                         vertically across the RECTIFIED window
                         (= vertical yarns / inch, horizontal yarns / inch);
                         h_tpi, v_tpi (tool v2: normalised by the length of a
                         rotated guide line) are used instead when present
      -- OR --
    warp, weft           ends / picks per inch, plus optional
    warp_axis            'vertical' (default: warp yarns run top-bottom in
                         the photo, so h_count = warp) or 'horizontal'
    weave      optional  plain / twill / satin / other (or 1/1, 2/1, ...)
    group      optional  fabric id; images of the same fabric are kept in
                         the same fold (strongly recommended - several
                         photos per fabric exist). If absent, groups are
                         inferred by clustering ImageNet embeddings of the
                         crops (see --auto-group-threshold); check the
                         printed clusters.
    exclude    optional  1 -> skip the image   (nonwoven=1 is treated the same)
    done       optional  if the column exists, only rows with done=1 are used
    uncertain  optional  1 -> sample weight 0.5 in training, flagged in eval
                         (confidence=1, the tool's "low", is treated the same)
    several rows for one image (e.g. two annotators on the IAA subset): the
    per-axis counts are averaged, the weave is the majority label, and the
    image is 'uncertain' if any row is.
    corners    optional  'x1 y1 x2 y2 x3 y3 x4 y4' window corners for images
                         not in final_bbox_annotation_set
Write an empty template with:   python train_cv.py --write-template annotations.csv

SCENES (default since Oct 8 2026, --scene-mode members): the 150 files are 76 scenes (annotation_tool
manifest fields scene_id / is_scene_rep / kind / scene_rot_k). Labels are propagated from each labelled
file to every file of its scene with annotation_tool/propagate_scene_labels.py (the rep's label wins;
h <-> v for files stored turned 90 / 270 deg). Then
    folds    GROUPED BY SCENE (GroupKFold-style, balanced on the number of scenes; a provided `group`
             column is merged in, so files that share a scene OR a group stay together)
    test     PRIMARY = one row per scene (its rep) -> summary.csv / summary.md;
             SECONDARY = every real file of the test scenes -> summary_perfile.csv / .md
             (predictions.csv has all rows, with scene / is_rep / kind columns)
    train    files of the training scenes: rep, quarter_res_copy and the notebook_synthetic files
             (augmentations, never test items); byte_identical and png_copy files are exact pixel
             duplicates of another member and are left out (--train-duplicates keeps them)
    aug1016  crops inherit their source file's scene (and therefore its fold)
--scene-mode reps trains and tests on the reps only; --scene-mode off is the pre-Oct-8 behaviour
(labelled images as given, provided groups or auto-grouping by embeddings).

Optional (default off): --extra-aug1016 adds the usable crops of the 2024 augmented set
(../aug1016, see its README) to the TRAINING folds only, labels propagated from their source
image (h/v swapped for 90/270-degree rotations); an augmentation is never in the training set
of its source's test fold.

Augmentation (--augment, default 'existing' = the original recipe: flips, anisotropic px/inch jitter,
cnn_common.photometric_aug).  '--augment whitelist' (tier A of augment_whitelist.json) or
'--augment whitelist_ext' (tier A+B, rotate +-5 deg) switch every trained model of the run -- real
fine-tuning AND a synthetic model trained inside this script -- to aug_pipeline.AugTileDataset, the
count-preserving pipeline of Job 2 (runs_aug/, RESULTS_AUG.md): exact label maps for rotations /
perspective (human guide-line counts are crossing counts; the yarn tilt is taken from ours'
angle_x / angle_y in methods/predictions_ours_150.csv), rot90 with h <-> v swap, whitelist-only
photometric / sensor / illumination ranges.  '--augment whitelist_optics' adds the label-exact optics /
phone-capture / fabric-geometry tier of augment_optics_whitelist.json (cnn/RESULTS_AUG.md, Oct 8).
--augment-drop removes families (ablation).  Use the matching synthetic checkpoint, e.g.
    python train_cv.py --labels annotations.csv --augment whitelist_ext \\
        --synth-ckpt resnet18=runs_aug/V2E_r18/model.pt --synth-ckpt efficientnet_b0=runs_aug/V2E_b0/model.pt

Outputs (in --out, default runs/cv_<timestamp>/)
    predictions.csv  per image x experiment x backbone (fold, gt, pred, APE)
    summary.csv / summary.md  mean +- std over folds of MAE, MAPE, acc@5/10%,
                              weave accuracy
    config.json, log.txt, checkpoints only with --save-ckpts
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

import cnn_common as C
import fabric_data as fd

if sys.platform.startswith("linux"):
    # Linux DataLoader workers are forked: a live OpenCV thread pool in the parent deadlocks them
    # (seen with the crops pre-loaded by load_crop); macOS spawns workers and is unaffected.
    cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SYNTH = os.path.join(HERE, "data", "synth_train") + "," + os.path.join(HERE, "data", "synth_train_v2")
RECT_CACHE = os.path.join(HERE, "data", "rectified")
TOOL_CROPS = os.path.join(os.path.dirname(HERE), "annotation_tool", "data", "crops")
TOOL_MANIFEST = os.path.join(os.path.dirname(HERE), "annotation_tool", "data", "manifest.json")
TRAIN_KINDS = ("rep", "quarter_res_copy", "notebook_synthetic")   # byte_identical / png_copy = pixel duplicates


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------
def write_template(path):
    df = pd.DataFrame({"image": range(150), "h_count": "", "v_count": "", "warp": "", "weft": "",
                       "warp_axis": "", "weave": "", "group": "", "exclude": "", "uncertain": "",
                       "notes": ""})
    df.to_csv(path, index=False)
    print("wrote template", path)


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def resolve_image(img_field):
    s = str(img_field).strip()
    if s.isdigit() or (s.replace(".0", "").isdigit()):
        i = int(float(s))
        return i, os.path.join(fd.IMG_DIR, f"{i}.jpeg")
    if os.path.isabs(s):
        return None, s
    base = os.path.splitext(os.path.basename(s))[0]
    if base.isdigit():
        return int(base), os.path.join(fd.IMG_DIR, f"{int(base)}.jpeg")
    return None, os.path.join(fd.IMG_DIR, s)


def _truthy(x):
    return str(x).strip().lower() in ("1", "1.0", "true", "yes", "y")


def load_annotations(path, log=print):
    """path: one CSV or a comma-separated list of CSVs (rows are pooled)."""
    df = pd.concat([pd.read_csv(p.strip(), dtype=str, keep_default_na=False) for p in str(path).split(",")],
                   ignore_index=True)
    cols = {c.lower().strip(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in cols.items()})
    if "image" not in df and "id" in df:          # annotation-tool export
        df["image"] = df["id"]
    if "image" not in df:
        sys.exit("annotation CSV needs an 'image' (or 'id') column")
    n_raw = len(df)
    if "done" in df:
        df = df[[_truthy(x) for x in df["done"]]]
    recs = []
    for r in df.to_dict("records"):
        if _num(r.get("exclude")) == 1 or _truthy(r.get("nonwoven", "")):
            continue
        # annotation tool v2: h_tpi / v_tpi = dots / length of the (possibly rotated) guide line
        # inside the window (== h_count / v_count for unrotated lines); else the raw counts
        h, v = _num(r.get("h_tpi")), _num(r.get("v_tpi"))
        h = h if h is not None else _num(r.get("h_count"))
        v = v if v is not None else _num(r.get("v_count"))
        if h is None or v is None:
            wa, we = _num(r.get("warp")), _num(r.get("weft"))
            if wa is None or we is None:
                continue
            axis = str(r.get("warp_axis", "vertical") or "vertical").strip().lower()
            h, v = (wa, we) if axis.startswith("v") or axis in ("nan", "") else (we, wa)
        idx, ipath = resolve_image(r["image"])
        corners = None
        if isinstance(r.get("corners"), str) and r["corners"].strip():
            corners = np.array([float(t) for t in r["corners"].replace(",", " ").split()], np.float32).reshape(4, 2)
        if h <= 0 or v <= 0:
            continue
        recs.append(dict(image=str(r["image"]).strip(), idx=idx, path=ipath, h=h, v=v,
                         weave=C.weave_to_idx(r.get("weave")), weave_raw=r.get("weave"),
                         group=r.get("group") if str(r.get("group", "")) not in ("", "nan") else None,
                         uncertain=_num(r.get("uncertain")) == 1 or _num(r.get("confidence")) == 1,
                         corners=corners))
    # merge several annotations of the same image (IAA subset / several files)
    merged = {}
    for r in recs:
        key = r["idx"] if r["idx"] is not None else r["path"]
        merged.setdefault(key, []).append(r)
    out = []
    for key, rs in merged.items():
        m = dict(rs[0])
        if len(rs) > 1:
            m["h"] = float(np.mean([x["h"] for x in rs]))
            m["v"] = float(np.mean([x["v"] for x in rs]))
            ws = [x["weave"] for x in rs if x["weave"] >= 0]
            m["weave"] = max(set(ws), key=ws.count) if ws else -100
            m["uncertain"] = any(x["uncertain"] for x in rs)
            m["n_annotations"] = len(rs)
        out.append(m)
    log(f"loaded {len(out)} labelled images ({len(recs)} usable rows of {n_raw}) from {path}")
    return out


def load_crop(rec, max_side=1024):
    """Rectified 1-inch crop.  For the 150 dataset images this is the exact
    crop the annotators counted on (annotation_tool/data/crops/{i}.jpg, made by
    common/rectify.py), so the h/v axes of labels and pixels agree."""
    if rec["corners"] is not None:
        img = cv2.imread(rec["path"])
        crop = fd.rectify(img, rec["corners"])
    elif rec["idx"] is not None:
        p = os.path.join(TOOL_CROPS, f"{rec['idx']}.jpg")
        if not os.path.exists(p):
            p = os.path.join(RECT_CACHE, f"{rec['idx']}.png")
        crop = cv2.imread(p) if os.path.exists(p) else fd.load_rectified(rec["idx"])
    else:
        raise ValueError(f"no corners for {rec['image']}")
    return C.cap_side(crop, max_side)


# ---------------------------------------------------------------------------
# scenes: one entry per FILE of every labelled scene (labels propagated from the scene's label)
# ---------------------------------------------------------------------------
def scene_module(manifest_path=TOOL_MANIFEST):
    """propagate_scene_labels.py of the annotation tool that owns the manifest (<tool>/data/manifest.json)."""
    import importlib.util
    tool = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    spec = importlib.util.spec_from_file_location("propagate_scene_labels", os.path.join(tool, "propagate_scene_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scene_expand(recs, mode="members", manifest_path=TOOL_MANIFEST, train_duplicates=False, log=print):
    """recs (load_annotations) -> (entries, info). Every file of a labelled scene becomes an entry with
    the scene's label in its own orientation, plus scene / is_rep / kind / rot_k / train / test keys.
    Images outside the 150 (corners / paths) are kept as their own one-file scene. info is None (and
    recs are returned unchanged) when the manifest has no scene fields."""
    try:
        PS = scene_module(manifest_path)
        scenes = PS.load_scenes(manifest_path)
    except (OSError, ValueError, KeyError) as e:
        log(f"scene mode unavailable ({e}); using the labelled images as given (--scene-mode off)")
        return recs, None
    ds = [r for r in recs if r["idx"] is not None and r["corners"] is None and r["idx"] in scenes.index]
    ext = [r for r in recs if not (r["idx"] is not None and r["corners"] is None and r["idx"] in scenes.index)]
    own = pd.DataFrame([dict(id=r["idx"], h=r["h"], v=r["v"], weave=r["weave"], weave_raw=r.get("weave_raw"),
                             uncertain=bool(r["uncertain"]), group=r["group"] or "",
                             n_annotations=r.get("n_annotations", 1)) for r in ds],
                       columns=["id", "h", "v", "weave", "weave_raw", "uncertain", "group", "n_annotations"]).set_index("id")
    per_file, per_scene = PS.propagate(own, scenes)
    entries = []
    for i, row in per_file.iterrows():
        if mode == "reps" and not row.is_scene_rep:
            continue
        kind = row.kind
        entries.append(dict(
            image=str(int(i)), idx=int(i), path=os.path.join(fd.IMG_DIR, f"{int(i)}.jpeg"), h=float(row.h), v=float(row.v),
            weave=int(row.weave), weave_raw=row.weave_raw, group=row.group or None, uncertain=bool(row.uncertain),
            corners=None, scene=f"scene{int(row.scene_id)}", is_rep=bool(row.is_scene_rep), kind=kind,
            rot_k=int(row.scene_rot_k), label_src=int(per_scene.at[row.scene_id, "label_src_id"]),
            train=kind in TRAIN_KINDS or (train_duplicates and kind in ("byte_identical", "png_copy")),
            test=kind != "notebook_synthetic"))
    for r in ext:
        entries.append(dict(r, scene=f"ext:{r['image']}", is_rep=True, kind="external", rot_k=0, label_src=None,
                            train=True, test=True))
    info = dict(n_scenes=int(per_scene.shape[0]), n_entries=len(entries), n_external=len(ext),
                n_train_files=sum(e["train"] for e in entries), n_test_files=sum(e["test"] for e in entries),
                n_swapped=sum(e["rot_k"] % 2 for e in entries),
                from_non_rep=sorted(int(s) for s in per_scene.index[per_scene.label_src_id != per_scene.index]))
    log(f"scenes ({mode}): {len(ds)} labelled dataset files -> {info['n_scenes']} scenes -> {len(entries)} files "
        f"({info['n_test_files']} test items, {info['n_train_files']} training files, {info['n_swapped']} with h/v "
        f"swapped); {len(ext)} external images")
    return entries, info


def scene_groups(entries):
    """Union of scene and provided group: files sharing either end up in one CV group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in entries:
        a = find(("s", e["scene"]))
        if e.get("group"):
            b = find(("g", str(e["group"])))
            if a != b:
                parent[b] = a
    roots = [find(("s", e["scene"])) for e in entries]
    label = {}
    for r, e in zip(roots, entries):
        label.setdefault(r, str(e["scene"]))
    return [label[r] for r in roots]


# ---------------------------------------------------------------------------
# optional: the 2024 augmented set (../aug1016) as extra TRAINING data
# ---------------------------------------------------------------------------
AUG1016_DIR = os.path.join(os.path.dirname(HERE), "aug1016")


def load_aug1016_items(recs, aug_dir, labels_csv=None, weight=1.0, log=print):
    """[(index into recs of the SOURCE image, Item, aug_id)] for every usable augmented crop whose
    source is labelled in this run.  Labels: by default propagated on the fly from this run's
    own (merged) source labels with aug1016/propagate_labels.py (h <-> v swapped for k = 90/270,
    weave unchanged); with labels_csv, read from a propagate_labels.py output instead.
    Crops are loaded lazily from disk (Item.img=None), capped to 1024 px like the originals."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aug1016_propagate", os.path.join(aug_dir, "propagate_labels.py"))
    PL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PL)
    src_of = {r["idx"]: n for n, r in enumerate(recs) if r["idx"] is not None and r["corners"] is None}
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        lab = lab[lab["aug_id"].fillna("").astype(str) != ""]
        lab["weave_idx"] = [C.weave_to_idx(w) for w in lab["weave"]]
        lab["uncertain"] = lab["uncertain"].map(_truthy) if "uncertain" in lab else False
    else:
        src = pd.DataFrame({"source_id": list(src_of), "h_tpi": [recs[n]["h"] for n in src_of.values()],
                            "v_tpi": [recs[n]["v"] for n in src_of.values()],
                            "weave_idx": [recs[n]["weave"] for n in src_of.values()],
                            "uncertain": [recs[n]["uncertain"] for n in src_of.values()]})
        lab = PL.propagate(src, PL.load_manifest())
    out = []
    for r in lab.itertuples():
        n = src_of.get(int(r.source_id))
        if n is None or not os.path.exists(r.image):
            continue
        w = (0.5 if bool(r.uncertain) else 1.0) * weight
        out.append((n, C.Item(None, r.image, float(r.h_tpi), float(r.v_tpi), int(r.weave_idx), w), r.aug_id))
    log(f"aug1016: {len(out)} augmented crops of {len({o[0] for o in out})} labelled sources "
        f"({'from ' + labels_csv if labels_csv else 'labels propagated from this run'}); training folds only")
    return out


# ---------------------------------------------------------------------------
# grouping (avoid leakage: same fabric photographed several times)
# ---------------------------------------------------------------------------
@torch.no_grad()
def auto_groups(crops, device, threshold=0.92, log=print):
    """Agglomerative clustering of ImageNet (resnet18) embeddings of the
    centre 448 px of each crop at 768 px/inch; cosine-similarity threshold."""
    import timm
    from scipy.cluster.hierarchy import fcluster, linkage
    m = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    feats = []
    for c in crops:
        im = C.resize_to(c, 768)[160:608, 160:608]
        x = C.to_tensor(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA))[None].to(device)
        f = m(x).float().cpu().numpy()[0]
        feats.append(f / (np.linalg.norm(f) + 1e-8))
    F = np.stack(feats)
    Z = linkage(F, method="average", metric="cosine")
    g = fcluster(Z, t=1 - threshold, criterion="distance")
    n = len(set(g))
    log(f"auto-grouping: {len(crops)} images -> {n} groups (cos-sim threshold {threshold})")
    return g


def make_folds(groups, n_folds=5, seed=0, weights=None):
    """Group-aware folds, balanced by size (greedy on shuffled groups). weights (optional, one per
    entry): balance on their sum per group instead of the entry count (scene mode: 1 per rep)."""
    rng = np.random.default_rng(seed)
    ug = list(dict.fromkeys(groups))
    rng.shuffle(ug)
    w = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=float)
    sizes = {g: float(w[np.asarray(groups) == g].sum()) for g in ug}
    ug.sort(key=lambda g: -sizes[g])
    load = [0] * n_folds
    assign = {}
    for g in ug:
        k = int(np.argmin(load))
        assign[g] = k
        load[k] += sizes[g]
    return np.array([assign[g] for g in groups])


# ---------------------------------------------------------------------------
# one experiment on one fold
# ---------------------------------------------------------------------------
def worker_init(_):
    cv2.setNumThreads(1)
    np.random.seed(torch.initial_seed() % 2 ** 32)


def finetune(init, backbone, train_items, device, epochs, tiles_per_image, batch, lr, workers, log,
             body_lr_mult=1.0, seed=0, aniso=0.5, lam_ratio=1.0, augment="existing", augment_drop=()):
    C.seed_all(seed)
    model = C.build_model(backbone, pretrained=(init == "imagenet"))
    if init != "imagenet":
        sd = torch.load(init, map_location="cpu", weights_only=False)["state"]
        model.load_state_dict(sd)
    if augment == "existing":
        ds = C.TileDataset(train_items, tiles_per_image=tiles_per_image, train=True, rot90=False, seed=seed,
                           aniso=aniso)
    else:   # count-preserving whitelist pipeline (aug_pipeline.py); labels are guide-line crossing counts
        import aug_pipeline as AP
        aitems = [AP.AugItem(it.img, it.path, it.h, it.v, it.weave, it.w, "crossing",
                             tuple(getattr(it, "angles", (0.0, 0.0))), "real") for it in train_items]
        ds = AP.AugTileDataset(aitems, tiles_per_image=tiles_per_image, policy=augment, drop=augment_drop,
                               aniso=aniso, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                                     persistent_workers=workers > 0, worker_init_fn=worker_init)
    C.train_loop(model, dl, device, epochs=epochs, lr=lr, log=log, log_every=max(10, len(dl)),
                 warmup=min(100, len(dl) * 2), body_lr_mult=body_lr_mult, lam_ratio=lam_ratio)
    return model


def evaluate_fold(model, test_recs, test_crops, device, Ps, flips):
    out = []
    for rec, crop in zip(test_recs, test_crops):
        o = C.predict_crop(model, crop, device, Ps=Ps, flips=flips)
        out.append(dict(pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                        pred_weave=int(np.argmax(o["weave_probs"]))))
    return out


def summarise(pred: pd.DataFrame):
    rows = []
    for (bb, exp), g in pred.groupby(["backbone", "experiment"]):
        per_fold = []
        for f, gf in g.groupby("fold"):
            m = C.count_metrics(gf.pred_h, gf.pred_v, gf.h_count, gf.v_count)
            wv = gf[gf.weave >= 0]
            m["weave_acc"] = float(100 * (wv.pred_weave == wv.weave).mean()) if len(wv) else float("nan")
            per_fold.append(m)
        pf = pd.DataFrame(per_fold)
        pooled = C.count_metrics(g.pred_h, g.pred_v, g.h_count, g.v_count)
        row = {"backbone": bb, "experiment": exp, "n_images": len(g)}
        for k in ["MAE", "MAE_h", "MAE_v", "MAPE", "MdAPE", "acc@5%", "acc@10%", "bias_pct", "weave_acc"]:
            row[f"{k}_mean"] = round(float(pf[k].mean()), 2)
            row[f"{k}_std"] = round(float(pf[k].std(ddof=0)), 2)
            if k in pooled:
                row[f"{k}_pooled"] = round(float(pooled[k]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(summ: pd.DataFrame):
    lines = ["| backbone | experiment | n | MAE (thr/in) | MAPE % | MdAPE % | acc@5% | acc@10% | weave acc % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, d in summ.iterrows():
        f = lambda k: f"{d[k + '_mean']:.1f} ± {d[k + '_std']:.1f}"
        lines.append(f"| {d['backbone']} | {d['experiment']} | {d['n_images']} | {f('MAE')} | {f('MAPE')} | "
                     f"{f('MdAPE')} | {f('acc@5%')} | {f('acc@10%')} | {f('weave_acc')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="annotation CSV (see schema above)")
    ap.add_argument("--write-template", help="write an empty annotation CSV for the 150 images and exit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backbones", default="resnet18,efficientnet_b0")
    ap.add_argument("--experiments", default="constant_mean,imagenet_ft,synth_ft,synth_zeroshot")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25, help="fine-tuning epochs per fold")
    ap.add_argument("--tiles-per-image", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="ImageNet init fine-tuning LR")
    ap.add_argument("--lr-synth-ft", type=float, default=1e-4, help="LR when starting from the synthetic model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aniso", type=float, default=0.5, help="anisotropic px/inch jitter (log-range)")
    ap.add_argument("--lam-ratio", type=float, default=1.0, help="weight of the log(h/v) loss")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--synth-dir", default=DEFAULT_SYNTH)
    ap.add_argument("--synth-ckpt", action="append", default=[],
                    help="backbone=path of an existing synthetic checkpoint (else trained here)")
    ap.add_argument("--synth-minutes", type=float, default=20.0)
    ap.add_argument("--auto-group-threshold", type=float, default=0.92)
    ap.add_argument("--fast-eval", action="store_true", help="single scale/no flips at test time")
    ap.add_argument("--max-images", type=int, default=None,
                    help="debug: subsample the labelled set (scene mode: the first N scenes with all their files)")
    ap.add_argument("--save-ckpts", action="store_true")
    ap.add_argument("--extra-aug1016", action="store_true",
                    help="add the usable 2024 augmented crops (../aug1016, labels propagated from their source "
                         "image) to the TRAINING folds only; an augmentation is never trained on in its "
                         "source's test fold (default off)")
    ap.add_argument("--aug1016-dir", default=AUG1016_DIR)
    ap.add_argument("--aug1016-labels", default=None,
                    help="use this propagate_labels.py output instead of propagating this run's labels")
    ap.add_argument("--aug1016-weight", type=float, default=1.0, help="sample-weight factor for the aug crops")
    ap.add_argument("--scene-mode", default="members", choices=("members", "reps", "off"),
                    help="members (default): labels propagated to every file of a labelled scene, folds grouped "
                         "by scene, primary metrics on the reps, secondary per file, notebook_synthetic files "
                         "training-only; reps: reps only; off: pre-Oct-8 behaviour (labelled images as given)")
    ap.add_argument("--manifest", default=TOOL_MANIFEST, help="annotation-tool manifest with the scene fields")
    ap.add_argument("--train-duplicates", action="store_true",
                    help="scene mode: also train on byte_identical / png_copy files (exact pixel duplicates)")
    ap.add_argument("--augment", default="existing",
                    choices=("existing", "whitelist", "whitelist_ext", "whitelist_optics"),
                    help="training augmentation: 'existing' (original recipe, default), 'whitelist' (verified "
                         "tier A), 'whitelist_ext' (tier A+B) or 'whitelist_optics' (A+B + label-exact optics / "
                         "phone capture, augment_optics_whitelist.json) via aug_pipeline.AugTileDataset (see above)")
    ap.add_argument("--augment-drop", default="",
                    help="comma list of augmentation families to drop with --augment whitelist*: geometry, "
                         "photometric, compression_blur, illumination, zoom, tiling (+ optics, phone with "
                         "whitelist_optics)")
    a = ap.parse_args()

    if a.write_template:
        write_template(a.write_template)
        return
    if not a.labels:
        ap.error("--labels is required")
    out = a.out or os.path.join(HERE, "runs", "cv_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    dev = C.get_device(a.device)
    log(f"device {dev}; out {out}")

    recs = load_annotations(a.labels, log)
    scene_info = None
    if a.scene_mode != "off":
        recs, scene_info = scene_expand(recs, a.scene_mode, a.manifest, a.train_duplicates, log)
        if scene_info is not None:
            with open(os.path.join(out, "scenes.json"), "w") as f:
                json.dump(scene_info, f, indent=1)
    if a.max_images:   # scene mode: the first N scenes, all of their files
        if scene_info is not None:
            keep = set(list(dict.fromkeys(r["scene"] for r in recs))[: a.max_images])
            recs = [r for r in recs if r["scene"] in keep]
        else:
            recs = recs[: a.max_images]
    crops = [load_crop(r) for r in recs]
    if scene_info is not None:
        groups = scene_groups(recs)
        log(f"grouping by scene: {len(set(r['scene'] for r in recs))} scenes -> {len(set(groups))} CV groups "
            f"(merged with provided groups: {sum(1 for r in recs if r.get('group'))} files had one)")
        fold = make_folds(groups, a.folds, a.seed, weights=[1.0 if r["is_rep"] else 0.0 for r in recs])
    else:
        if all(r["group"] is not None for r in recs):
            groups = [str(r["group"]) for r in recs]
            log(f"using provided groups: {len(set(groups))} fabrics")
        else:
            g = auto_groups(crops, dev, a.auto_group_threshold, log)
            groups = [str(r["group"]) if r["group"] is not None else f"auto{gi}" for r, gi in zip(recs, g)]
        fold = make_folds(groups, a.folds, a.seed)
    fdf = pd.DataFrame({"image": [r["image"] for r in recs], "group": groups, "fold": fold})
    if scene_info is not None:
        for k in ("scene", "is_rep", "kind", "rot_k", "train", "test", "label_src"):
            fdf[k] = [r[k] for r in recs]
    fdf.to_csv(os.path.join(out, "folds.csv"), index=False)
    log("fold sizes:", np.bincount(fold, minlength=a.folds).tolist())
    if scene_info is not None:
        log("scenes (reps) per fold:", np.bincount(fold[[r["is_rep"] for r in recs]], minlength=a.folds).tolist())

    items = [C.Item(c, None, r["h"], r["v"], r["weave"], 0.5 if r["uncertain"] else 1.0)
             for r, c in zip(recs, crops)]
    aug_drop = tuple(x for x in a.augment_drop.split(",") if x)
    if a.augment != "existing":
        # yarn tilt of each crop for the exact crossing-count maps of rotate / perspective: ours' lattice
        # angles on the same crops (0 deg when unavailable, e.g. images given by corners / aug1016 crops)
        ang = {}
        p_ours = os.path.join(os.path.dirname(HERE), "methods", "predictions_ours_150.csv")
        if os.path.exists(p_ours):
            po = pd.read_csv(p_ours)
            ang = {int(r.id): (float(np.nan_to_num(r.angle_x)), float(np.nan_to_num(r.angle_y))) for r in po.itertuples()}
        for r, it in zip(recs, items):
            it.angles = ang.get(r["idx"], (0.0, 0.0)) if (r["idx"] is not None and r["corners"] is None) else (0.0, 0.0)
        log(f"augment={a.augment} drop={list(aug_drop)}; yarn tilt from ours for "
            f"{sum(1 for it in items if it.angles != (0.0, 0.0))}/{len(items)} images")
    extra = []   # [(source rec index, Item, aug_id)] -- only with --extra-aug1016
    if a.extra_aug1016:
        extra = load_aug1016_items(recs, a.aug1016_dir, a.aug1016_labels, a.aug1016_weight, log)
        pd.DataFrame({"aug_id": [e[2] for e in extra], "source": [recs[e[0]]["image"] for e in extra],
                      "source_fold": [int(fold[e[0]]) for e in extra], "h": [e[1].h for e in extra],
                      "v": [e[1].v for e in extra]}).to_csv(os.path.join(out, "extra_aug1016.csv"), index=False)
    exps = [e.strip() for e in a.experiments.split(",") if e.strip()]
    bbs = [b.strip() for b in a.backbones.split(",") if b.strip()]
    Ps, flips = ((768,), False) if a.fast_eval else (C.P_TEST, True)

    # synthetic checkpoints (needed for synth_ft / synth_zeroshot)
    synth_ck = dict(s.split("=", 1) for s in a.synth_ckpt)
    if any(e.startswith("synth") for e in exps):
        for bb in bbs:
            if bb not in synth_ck:
                from train_synth import train_synth
                p = os.path.join(out, f"synth_{bb}")
                log(f"training synthetic model for {bb} ({a.synth_minutes} min) -> {p}")
                train_synth(a.synth_dir, bb, minutes=a.synth_minutes, workers=a.workers, device=a.device,
                            out=p, log=log, aniso=a.aniso, lam_ratio=a.lam_ratio, augment=a.augment,
                            augment_drop=aug_drop)
                synth_ck[bb] = os.path.join(p, "model.pt")
    rows = []
    t_all = time.time()
    for bb in bbs:
        zs_model = C.load_ckpt(synth_ck[bb], dev) if "synth_zeroshot" in exps else None
        for k in range(a.folds):
            # scene mode: train on the training scenes' training files, test on the fold's real files
            # (rows of reps = primary, all = secondary); notebook_synthetic files are never test items
            tr = [i for i in range(len(recs)) if fold[i] != k and recs[i].get("train", True)]
            te = [i for i in range(len(recs)) if fold[i] == k and recs[i].get("test", True)]
            tr_ref = [i for i in tr if recs[i].get("is_rep", True)]   # one value per scene for constant_mean
            tr_items = [items[i] for i in tr]
            if extra:   # augmentations of TRAINING-fold sources only (never of a test-fold source)
                tr_items += [it for n, it, _ in extra if fold[n] != k]
                log(f"[fold {k}] training on {len(tr)} originals + {len(tr_items) - len(tr)} aug1016 crops")
            te_recs, te_crops = [recs[i] for i in te], [crops[i] for i in te]
            for exp in exps:
                t0 = time.time()
                if exp == "constant_mean":
                    mh = float(np.mean([items[i].h for i in tr_ref]))
                    mv = float(np.mean([items[i].v for i in tr_ref]))
                    kw = [items[i].weave for i in tr_ref if items[i].weave >= 0]   # majority weave of the fold
                    mw = max(set(kw), key=kw.count) if kw else -1
                    preds = [dict(pred_h=mh, pred_v=mv, h_iqr=0, v_iqr=0, pred_weave=mw) for _ in te]
                else:
                    if exp == "imagenet_ft":
                        model = finetune("imagenet", bb, tr_items, dev, a.epochs, a.tiles_per_image, a.batch,
                                         a.lr, a.workers, log, seed=a.seed + k, aniso=a.aniso,
                                         lam_ratio=a.lam_ratio, augment=a.augment, augment_drop=aug_drop)
                    elif exp == "synth_ft":
                        model = finetune(synth_ck[bb], bb, tr_items, dev, a.epochs, a.tiles_per_image, a.batch,
                                         a.lr_synth_ft, a.workers, log, seed=a.seed + k, aniso=a.aniso,
                                         lam_ratio=a.lam_ratio, augment=a.augment, augment_drop=aug_drop)
                    elif exp == "synth_zeroshot":
                        model = zs_model
                    else:
                        raise ValueError(exp)
                    preds = evaluate_fold(model, te_recs, te_crops, dev, Ps, flips)
                    if a.save_ckpts and exp != "synth_zeroshot":
                        C.save_ckpt(model, os.path.join(out, "ckpts", f"{bb}_{exp}_fold{k}.pt"))
                for r, p in zip(te_recs, preds):
                    row = dict(backbone=bb, experiment=exp, fold=k, image=r["image"], h_count=r["h"],
                               v_count=r["v"], weave=r["weave"], uncertain=r["uncertain"], **p)
                    if scene_info is not None:
                        row.update(scene=r["scene"], is_rep=r["is_rep"], kind=r["kind"], rot_k=r["rot_k"])
                    rows.append(row)
                prim = [n for n, r in enumerate(te_recs) if r.get("is_rep", True)]   # one row per scene
                fm = C.count_metrics([preds[n]["pred_h"] for n in prim], [preds[n]["pred_v"] for n in prim],
                                     [te_recs[n]["h"] for n in prim], [te_recs[n]["v"] for n in prim])
                log(f"[{bb} | {exp} | fold {k}] n={len(prim)}{' scenes (+%d other files)' % (len(te) - len(prim)) if scene_info else ''} "
                    f"MAE {fm['MAE']:.2f} MAPE {fm['MAPE']:.1f}% "
                    f"acc@5% {fm['acc@5%']:.0f}  ({time.time() - t0:.0f}s, total {(time.time() - t_all) / 60:.1f} min)")
                pd.DataFrame(rows).to_csv(os.path.join(out, "predictions.csv"), index=False)
    pred = pd.DataFrame(rows)
    pred["ape_h"] = 100 * (pred.pred_h - pred.h_count).abs() / pred.h_count
    pred["ape_v"] = 100 * (pred.pred_v - pred.v_count).abs() / pred.v_count
    pred.to_csv(os.path.join(out, "predictions.csv"), index=False)
    if scene_info is not None:   # PRIMARY: one row per scene (the rep); SECONDARY: every real test file
        summ = summarise(pred[pred.is_rep.astype(bool)])
        summ_f = summarise(pred)
        summ_f.to_csv(os.path.join(out, "summary_perfile.csv"), index=False)
        with open(os.path.join(out, "summary_perfile.md"), "w") as f:
            f.write(f"# {a.folds}-fold CV, SECONDARY per-file ({int(pred.groupby(['backbone', 'experiment']).size().max())} "
                    f"real files of {pred.scene.nunique()} scenes; scene-grouped folds)\n\n{to_markdown(summ_f)}\n")
        n_sc = int(pred[pred.is_rep.astype(bool)].groupby(["backbone", "experiment"]).size().max())
        title = f"# {a.folds}-fold CV, PRIMARY one row per scene ({n_sc} scenes, {len(set(groups))} scene groups)"
    else:
        summ = summarise(pred)
        title = f"# {a.folds}-fold CV ({len(recs)} images, {len(set(groups))} groups)"
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    md = to_markdown(summ)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"{title}\n\n{md}\n")
    log("\n" + md)
    log(f"done in {(time.time() - t_all) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()

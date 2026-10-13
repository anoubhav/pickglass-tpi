"""Example figures for the annotator's Examples panel (v2.2.2): guide/examples/.

Blind protocol: every figure is made from NON-IAA scene representatives only. The IAA subset is read
from data/manifest.json (`iaa_subset`, the authoritative list); a file is excluded when it or its scene
is in the subset, and only scene reps are used (copies are never shown). No annotation is read
(data/saved/ is never opened), so no annotator's dots or counts can appear.
Same patch (v2.2.2 fix): the scene grouping joins only resolution copies of ONE photo, but several reps are
other photos of the same patch of fabric as an IAA-subset scene (other lighting / shadow / turn; e.g. the
howto figure's #8). Machine dots or a count on such a photo would hint at that blind count, so a rep is also
excluded when its 1-inch crop matches an IAA rep's crop (SameSpot: SIFT + RANSAC similarity, >= 12 inliers
at scale ~1; unrelated fabrics give <= 5). Needs OpenCV (cv2, in the figure venv).

Writes (deterministic; re-running gives the same selection and the same examples.json):
  guide/examples/example_<id>.jpg  for ~4 scenes (plain / twill / pattern at full resolution and one
                                   quarter-resolution scene): the 1-inch crop with both guide lines and
                                   the machine proposal dots, and two 1/10-inch zooms (H and V line).
                                   Caption: "machine proposals - shows the layout, not verified counts".
                                   No count, weave guess or attention level is printed (the tool never
                                   shows a weave guess; a printed one would anchor the weave label).
  guide/examples/photo_problems.jpg  crops (no dots, no counts) of common photo problems, picked from the
                                   counter's quality measures among non-IAA reps (replaces, in the tool,
                                   guide/bad_photos.png, three of whose six crops are IAA-subset scenes).
  guide/examples/examples.json     the index the tool reads: per figure its src, title, caption and the
                                   scene ids it shows (the tool re-checks them against the IAA subset
                                   and hides a figure whose scene the annotator counts blind).
guide/howto_count.png (make_doc_examples.py; scenes 8 and 20, both non-IAA) is listed in the index only if it
passes the same checks; it does NOT (#8 is another photo of the same patch as an IAA-subset scene), so it is
not shown. guide/annotation_guide.png is NOT used: it shows proposal dots on scene 77 (IAA).

Selection ("the counter is confident"): candidates are non-IAA reps of the slot's weave class (the frozen
counter's weave guess: plain / twill / other = satin, other, unknown, as the IAA strata) and resolution
(full: the rep's photo is >= 3000 px on its long side, else quarter); ranked by attention level 'low' first
(the estimators agree and nothing is flagged), then the counter's confidence (high first), then id.

Usage (venv with matplotlib, numpy, pillow):
  <venv2>/bin/python make_tool_examples.py            # write the figures + examples.json
  <venv2>/bin/python make_tool_examples.py --dry-run  # print the selection only
Tests: <venv2>/bin/python tests/test_examples.py
"""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "guide", "examples")
PRED = os.path.join(HERE, "..", "methods", "predictions_ours_150.csv")
CAPTION = "machine proposals — shows the layout, not verified counts"
FULL_PX = 3000          # same split as the IAA strata (iaa_policy)

# (counter weave class, resolution, slot name); weave class None = any. The slot name is for the log only:
# no weave word is printed on a figure or in the index (the tool never shows the counter's weave guess).
SLOTS = [
    ("plain", "full", "plain, full resolution"),
    ("twill", "full", "twill, full resolution"),
    ("other", "full", "satin / pattern / other, full resolution"),
    (None, "quarter", "any weave, quarter resolution"),
]
HOWTO = {"src": "guide/howto_count.png", "scenes": [8, 20], "proposals": True,
         "title": "Where the dots go",
         "caption": "One dot per yarn on the line (machine proposals on #8, not verified counts); "
                    "the diagonal stripes of a twill (#20) are not yarns."}

# tool colours (style.css): guide lines, H dots (vertical yarns), V dots (horizontal yarns)
COL = {"line": "#f5c542", "h": "#22d3ee", "v": "#f472b6", "bg": "#1a1d22", "text": "#e6e8eb", "muted": "#8b939e"}


# ---------------------------------------------------------------- selection (pure, no plotting imports)
def load_manifest(path=None):
    with open(path or os.path.join(HERE, "data", "manifest.json")) as f:
        return json.load(f)


def load_preds(path=PRED):
    """Counter predictions keyed by int id (strings kept; numbers parsed on use)."""
    with open(path, newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def num(v, default=float("nan")):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


def iaa_ids(manifest):
    """The IAA subset, from the manifest (authoritative)."""
    return {int(i) for i in (manifest.get("iaa_subset") or [])}


def scene_of(img):
    s = img.get("scene_id")
    return int(img["id"] if s is None else s)


def excluded(img, iaa):
    """True for anything an example may not show: a copy (not a scene rep), or an IAA-subset file or scene."""
    if img.get("is_scene_rep") is False:
        return True
    return int(img["id"]) in iaa or scene_of(img) in iaa


SAME_SPOT_MIN = 12      # RANSAC inliers (scale ~1) that make two crops "the same patch"; on the v2 data the same-patch
                        # pairs have 16-1979 (aligned high-pass NCC 0.54-0.93), every other non-IAA x IAA pair <= 5
SAME_SPOT_PX = 500      # crops are compared at 500 px per inch, central 70% (the window frame is left out)


class SameSpot:
    """Callable: True when a rep's 1-inch crop shows the same patch of fabric as an IAA-subset rep (another photo of
    that spot, which the scene grouping does not join). SIFT features of the central 70% of each crop (500 px,
    CLAHE), ratio test 0.75, RANSAC similarity (3 px); same patch = >= SAME_SPOT_MIN inliers with scale 0.8-1.25
    (any turn). Deterministic; results are cached. `matches(img)` -> [(inliers, iaa_id)] above the threshold."""

    def __init__(self, images, iaa, root=HERE, min_inliers=SAME_SPOT_MIN):
        self.root, self.min = root, int(min_inliers)
        self.by_id = {int(e["id"]): e for e in images}
        self.iaa_reps = sorted(i for i in iaa if i in self.by_id and self.by_id[i].get("is_scene_rep") is not False)
        self._feat, self._res = {}, {}

    def _features(self, i):
        if i not in self._feat:
            import cv2
            import numpy as np
            im = cv2.imread(os.path.join(self.root, self.by_id[i]["crop"]), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(self.by_id[i]["crop"])
            n = SAME_SPOT_PX
            im = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA))
            mask = np.zeros_like(im)
            mask[int(n * 0.15):int(n * 0.85), int(n * 0.15):int(n * 0.85)] = 255
            self._feat[i] = cv2.SIFT_create(nfeatures=4000).detectAndCompute(im, mask)
        return self._feat[i]

    def inliers(self, a, b):
        import cv2
        import numpy as np
        ka, da = self._features(a)
        kb, db = self._features(b)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return 0
        good = [m[0] for m in cv2.BFMatcher().knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            return 0
        pa = np.float32([ka[g.queryIdx].pt for g in good])
        pb = np.float32([kb[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is None or inl is None:
            return 0
        scale = float(np.hypot(A[0, 0], A[1, 0]))
        return int(inl.sum()) if 0.8 <= scale <= 1.25 else 0

    def matches(self, img):
        i = int(img["id"])
        if i not in self._res:
            self._res[i] = sorted(((n, j) for j in self.iaa_reps if j != i
                                   for n in [self.inliers(i, j)] if n >= self.min), reverse=True)
        return self._res[i]

    def __call__(self, img):
        return bool(self.matches(img))


def weave_class(w):
    w = str(w or "")
    if w == "plain":
        return "plain"
    if w.startswith("twill"):
        return "twill"
    return "other"


def resolution(img):
    return "full" if max(int(img.get("orig_w") or 0), int(img.get("orig_h") or 0)) >= FULL_PX else "quarter"


def _rank(img, preds):
    p = preds.get(int(img["id"]), {})
    conf = num(p.get("confidence"), 0.0)
    if not math.isfinite(conf):
        conf = 0.0
    return (img.get("attention") != "low", -conf, int(img["id"]))


def pick_examples(images, preds, iaa, slots=SLOTS, same_spot=None):
    """[(id, title)] one per slot (skipped when a slot has no candidate). Never an IAA file or scene, never a copy,
    and (with same_spot, a SameSpot) never another photo of the same patch as an IAA scene."""
    out, used = [], set()
    for wc, res, title in slots:
        cands = [im for im in images
                 if not excluded(im, iaa) and int(im["id"]) not in used and int(im["id"]) in preds
                 and (wc is None or weave_class(preds[int(im["id"])].get("weave")) == wc) and resolution(im) == res]
        best = next((im for im in sorted(cands, key=lambda im: _rank(im, preds))
                     if same_spot is None or not same_spot(im)), None)
        if best is None:
            continue
        used.add(int(best["id"]))
        out.append((int(best["id"]), title))
    return out


PROBLEMS = [  # (label, advice, measure, ascending, filter, minimum) -- the categories of make_doc_examples.bad_photos
    ("Glare", "Move the line to a clear row.\nTick Glare.", "glare", False, None, 0.005),
    ("Blur", "If yarns are visible when zoomed,\ncount; tick Blur, confidence 2.", "sharpness", True, "hires", None),
    ("Dark / shadow", "Press E for the enhanced view.\nTick Shadow.", "dark_frac", False, None, 0.02),
    ("Tilted yarns", "Press A (auto-align) or R (turn the\nimage), check, then count.", "tilt", False, None, None),
    ("Low resolution", "Zoom in; if yarns blur together,\ncount anyway, confidence 1.", "px_per_yarn", True, None, None),
    ("Hard pattern", "Count yarns, not colour blocks.\nWeave: Unsure is fine.", "confidence", True, "satin", None),
]


def _measure(img, p, key):
    if key == "glare":   # the counter's glare fraction or, if larger, the near-saturated share (attention sat240)
        sig = img.get("attention_signals") or {}
        a = [num(p.get("glare_frac")), num(sig.get("sat240"))]
        a = [x for x in a if math.isfinite(x)]
        return max(a) if a else float("nan")
    if key == "tilt":
        a = [abs(num(p.get("angle_x"))), abs(num(p.get("angle_y")))]
        a = [x for x in a if math.isfinite(x)]
        return max(a) if a else float("nan")
    if key == "px_per_yarn":
        m = max(num(p.get("h_count"), 0.0), num(p.get("v_count"), 0.0))
        return num(p.get("S")) / m if m > 0 else float("nan")
    return num(p.get(key))


def pick_problem_photos(images, preds, iaa, same_spot=None):
    """[(id, label, advice)] for the photo-problems figure: non-IAA reps only (with same_spot: never another photo of
    the same patch as an IAA scene), each photo at most once; a category whose best value does not reach its
    minimum (e.g. no glare at all outside the IAA subset) is left out."""
    out = []
    pool = [im for im in images if not excluded(im, iaa) and int(im["id"]) in preds]
    by_id = {int(im["id"]): im for im in pool}
    for label, advice, key, asc, filt, vmin in PROBLEMS:
        cands = []
        for im in pool:
            p = preds[int(im["id"])]
            if filt == "hires" and not num(p.get("S"), 0.0) > 1000:
                continue
            if filt == "satin" and not any(s in str(p.get("weave")) for s in ("satin", "unsure")):
                continue
            v = _measure(im, p, key)
            if not math.isfinite(v) or (vmin is not None and v < vmin) or int(im["id"]) in [o[0] for o in out]:
                continue
            cands.append(((v if asc else -v), int(im["id"])))
        best = next((i for _, i in sorted(cands) if same_spot is None or not same_spot(by_id[i])), None)
        if best is not None:
            out.append((best, label, advice))
    return out


def build_index(manifest, examples, problems, howto=HOWTO, same_spot=None):
    """examples.json content; every figure names the scenes it shows. Raises if any of them is IAA (or, with
    same_spot, another photo of the same patch as an IAA scene); the howto figure is left out in that case."""
    iaa = iaa_ids(manifest)
    by_id = {int(e["id"]): e for e in manifest["images"]}

    def refused(e):   # excluded(), or another photo of the same patch as an IAA scene
        return excluded(e, iaa) or (same_spot is not None and same_spot(e))

    figs = []
    for i, _slot in examples:
        e = by_id[i]
        figs.append({"src": "guide/examples/example_%d.jpg" % i, "scenes": [i], "proposals": True,
                     "title": "#%d · %s resolution, %d px per inch" % (i, resolution(e), int(e["rect_size"][0])),
                     "caption": CAPTION,
                     "resolution": resolution(e), "px_per_inch": int(e["rect_size"][0])})
    if howto and all(int(s) in by_id and not refused(by_id[int(s)]) for s in howto["scenes"]):
        figs.append(dict(howto))
    if problems:
        figs.append({"src": "guide/examples/photo_problems.jpg", "scenes": [p[0] for p in problems], "proposals": False,
                     "title": "Difficult photos",
                     "caption": "Common photo problems and what to do (crops only, no dots or counts)."})
    for f in figs:
        bad = [s for s in f["scenes"] if int(s) not in by_id or refused(by_id[int(s)])]
        if bad:
            raise ValueError("figure %s would show IAA / same-patch / copy / unknown scenes %s" % (f["src"], bad))
    out = {"version": 1, "tool": "wvaq2027-thread-count-annotator", "made_by": "make_tool_examples.py",
           "iaa_version": manifest.get("iaa_version"), "iaa_subset_excluded": sorted(iaa),
           "rule": "non-IAA scene reps only (IAA subset from data/manifest.json); proposals only, no annotation is read; "
                   "slots full plain / twill / other and quarter, ranked attention low first, then counter confidence, then id",
           "figures": figs}
    if same_spot is not None:
        out["same_patch_rule"] = ("no figure shows another photo of the same patch of fabric as an IAA-subset scene "
                                  "(SIFT + RANSAC similarity of the 1-inch crops, >= %d inliers)" % same_spot.min)
    return out


def add_sizes(index, root=HERE):
    """Pixel size [w, h] of each figure file (the tool reserves the space before an image loads)."""
    from PIL import Image
    for f in index["figures"]:
        p = os.path.join(root, f["src"])
        if os.path.exists(p):
            with Image.open(p) as im:
                f["px"] = [int(im.size[0]), int(im.size[1])]
    return index


# ---------------------------------------------------------------- drawing
def _zoom(lo_c, size, span):
    lo = int(round(lo_c - span / 2))
    lo = max(0, min(lo, size - int(span)))
    return lo, lo + int(span)


def draw_example(e, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from PIL import Image

    crop = np.asarray(Image.open(os.path.join(HERE, e["crop"])).convert("RGB"))
    S = crop.shape[1]
    p = e["proposal"]
    hy, vx = float(p["h_line_y"]), float(p["v_line_x"])
    hx, vy = sorted(p["h_positions"]), sorted(p["v_positions"])
    span = S / 10.0
    zx0, zx1 = _zoom(S * 0.35, S, span)          # H-line zoom: x range
    zy0, zy1 = _zoom(S * 0.35, S, span)          # V-line zoom: y range
    hy0, hy1 = _zoom(hy, S, span)
    vx0, vx1 = _zoom(vx, S, span)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.9), facecolor=COL["bg"])
    ms = 6 if S > 1000 else 9
    ax[0].imshow(crop)
    ax[0].axhline(hy, color=COL["line"], lw=0.9)
    ax[0].axvline(vx, color=COL["line"], lw=0.9)
    ax[0].scatter(hx, [hy] * len(hx), s=ms, c=COL["h"], edgecolors="k", linewidths=0.3, zorder=3)
    ax[0].scatter([vx] * len(vy), vy, s=ms, c=COL["v"], edgecolors="k", linewidths=0.3, zorder=3)
    ax[0].add_patch(Rectangle((zx0, hy0), span, span, fill=False, ec=COL["h"], lw=1.2))
    ax[0].add_patch(Rectangle((vx0, zy0), span, span, fill=False, ec=COL["v"], lw=1.2))
    ax[0].set_xlim(0, S); ax[0].set_ylim(S, 0)
    ax[0].set_title("1 inch (the window):\nboth guide lines, boxes = zooms", fontsize=10, color=COL["text"])

    ax[1].imshow(crop[hy0:hy1, zx0:zx1], extent=(zx0, zx1, hy1, hy0), interpolation="lanczos")
    xs = [x for x in hx if zx0 <= x <= zx1]
    ax[1].axhline(hy, color=COL["line"], lw=1.2)
    ax[1].scatter(xs, [hy] * len(xs), s=95, c=COL["h"], edgecolors="k", linewidths=0.8, zorder=3)
    ax[1].set_xlim(zx0, zx1); ax[1].set_ylim(hy1, hy0)
    ax[1].set_title("1/10 inch along the H line:\na dot on each vertical yarn", fontsize=10, color=COL["h"])

    ax[2].imshow(crop[zy0:zy1, vx0:vx1], extent=(vx0, vx1, zy1, zy0), interpolation="lanczos")
    ys = [y for y in vy if zy0 <= y <= zy1]
    ax[2].axvline(vx, color=COL["line"], lw=1.2)
    ax[2].scatter([vx] * len(ys), ys, s=95, c=COL["v"], edgecolors="k", linewidths=0.8, zorder=3)
    ax[2].set_xlim(vx0, vx1); ax[2].set_ylim(zy1, zy0)
    ax[2].set_title("1/10 inch along the V line:\na dot on each horizontal yarn", fontsize=10, color=COL["v"])
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#3a404a")
    fig.text(0.5, 0.02, "#%d · %d px per inch · %s" % (e["id"], S, CAPTION), ha="center", fontsize=10,
             color=COL["muted"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.08, wspace=0.04)
    fig.savefig(path, dpi=120, facecolor=COL["bg"], pil_kwargs={"quality": 88})
    plt.close(fig)


def draw_problems(by_id, picks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(picks)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows), facecolor=COL["bg"], squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, (i, label, advice) in zip(axes.ravel(), picks):
        a.axis("on")
        im = Image.open(os.path.join(HERE, by_id[i]["crop"])).convert("RGB")
        im.thumbnail((900, 900))
        a.imshow(im)
        a.set_title("%s  (photo #%d)" % (label, i), fontsize=11, fontweight="bold", color=COL["text"])
        a.set_xlabel(advice, fontsize=9.5, color=COL["text"])
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#3a404a")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=100, facecolor=COL["bg"], pil_kwargs={"quality": 86})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the selection, write nothing")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred", default=PRED, help="counter predictions CSV (default ../methods/predictions_ours_150.csv)")
    a = ap.parse_args()
    man = load_manifest()
    preds = load_preds(a.pred)
    iaa = iaa_ids(man)
    images = sorted(man["images"], key=lambda e: int(e["id"]))
    by_id = {int(e["id"]): e for e in images}
    ss = SameSpot(images, iaa)
    ex = pick_examples(images, preds, iaa, same_spot=ss)
    pr = pick_problem_photos(images, preds, iaa, same_spot=ss)
    index = build_index(man, ex, pr, same_spot=ss)
    print("examples:", ex)
    print("photo problems:", [(i, l) for i, l, _ in pr])
    print("IAA subset excluded:", sorted(iaa))
    print("same patch as an IAA scene (checked candidates, left out):", sorted(i for i, m in ss._res.items() if m))
    print("howto_count.png:", "shown" if any(f["src"] == HOWTO["src"] for f in index["figures"]) else "not shown")
    if a.dry_run:
        return 0
    os.makedirs(a.out, exist_ok=True)
    for i, _ in ex:
        draw_example(by_id[i], os.path.join(a.out, "example_%d.jpg" % i))
    if pr:
        draw_problems(by_id, pr, os.path.join(a.out, "photo_problems.jpg"))
    add_sizes(index)
    with open(os.path.join(a.out, "examples.json"), "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    print("wrote", len(index["figures"]), "figures to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

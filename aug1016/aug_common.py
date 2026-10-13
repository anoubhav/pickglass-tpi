"""Shared helpers for the 2024 augmented set ("1016" = 150 originals + 866 augmentations,
853 of which survive on disk) -- see README.md in this folder.

Everything geometric goes through ../common/rectify.py so the augmented crops use exactly the
same corner convention, output-size rule, interpolation and border mode as the 150 crops the
annotators count on (annotation_tool/data/crops/{i}.jpg).

Conventions used throughout aug1016/
  k_deg     rotation (degrees, COUNTER-clockwise on screen, = numpy.rot90 / albumentations
            RandomRotate90 'factor' * 90) that maps the SOURCE rectified crop onto the
            AUGMENTED rectified crop:  aug_crop ~= np.rot90(src_crop, k_deg // 90)
  swap_hv   k_deg in (90, 270): the aug crop's H line crosses the yarns that the source crop's
            V line crosses, so h <-> v (and warp_dir vertical <-> horizontal) are swapped.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # .../wvaq2027
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402  (shared contract; never re-implemented here)

NB_DIR = (os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "photos"))            # READ-ONLY
AUG_IMG_DIR = os.path.join(NB_DIR, "augmented_images")
AUG_XML_DIR = os.path.join(NB_DIR, "augmented_xml")
TOOL_CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
CROPS_DIR = os.path.join(HERE, "crops")
MANIFEST = os.path.join(HERE, "manifest.csv")
JPEG_Q = 92            # same quality as annotation_tool/prepare.py uses for the 150 crops
N_AUG_PER_SOURCE = 6   # notebook cell 55: 6 augmentations per original (aug0..aug5)


def aug_id(i: int, n: int) -> str:
    return f"{int(i)}_aug{int(n)}"


def parse_aug_id(s: str):
    a, b = str(s).split("_aug")
    return int(a), int(b)


def aug_image_path(i, n):
    return os.path.join(AUG_IMG_DIR, f"{aug_id(i, n)}.jpeg")


def aug_xml_path(i, n):
    return os.path.join(AUG_XML_DIR, f"{aug_id(i, n)}.xml")


def crop_path(i, n):
    return os.path.join(CROPS_DIR, f"{aug_id(i, n)}.jpg")


def list_aug_ids():
    """All (i, n) with BOTH an image and an XML on disk, sorted."""
    out = []
    for f in os.listdir(AUG_IMG_DIR):
        if not f.endswith(".jpeg") or "_aug" not in f:
            continue
        i, n = parse_aug_id(f[:-5])
        if os.path.exists(aug_xml_path(i, n)):
            out.append((i, n))
    return sorted(out)


# ---------------------------------------------------------------------------- polygons
def read_voc_polygon(path):
    """Stored polygon (4, 2) float in STORED order (point1..point4) and (width, height)."""
    root = ET.parse(path).getroot()
    poly = root.find("object/polygon")
    pts = np.array([[float(poly.find(f"x{k}").text), float(poly.find(f"y{k}").text)] for k in range(1, 5)])
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    return pts, (W, H)


def is_convex(pts) -> bool:
    """Same criterion as the 2024 notebook (cell 62): all edge cross products share a sign."""
    p = np.asarray(pts, float)
    cps = []
    for k in range(4):
        a, b, c = p[k], p[(k + 1) % 4], p[(k + 2) % 4]
        v1, v2 = b - a, c - b
        cps.append(v1[0] * v2[1] - v1[1] * v2[0])
    return all(c > 0 for c in cps) or all(c < 0 for c in cps)


def polygon_problems(pts, size, img_shape=None):
    """List of reasons why a polygon is not usable ([] = fine)."""
    W, H = size
    out = []
    p = np.asarray(pts, float)
    if p.shape != (4, 2):
        return ["not_4_points"]
    if (p[:, 0] < 0).any() or (p[:, 0] > W).any() or (p[:, 1] < 0).any() or (p[:, 1] > H).any():
        out.append("out_of_bounds")
    if not is_convex(p):
        out.append("non_convex")
    if img_shape is not None and (img_shape[1], img_shape[0]) != (W, H):
        out.append("xml_size_mismatch")
    return out


def shoelace(pts) -> float:
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------- rectification
def rectify_canonical(img, pts, out_size=None):
    """EXACTLY common/rectify.py: corners re-ordered geometrically (TL,TR,BR,BL on screen,
    TL = smallest x+y), S = round(mean side), bilinear, BORDER_REPLICATE.  The result is
    screen-aligned, i.e. it carries the image's 90-degree rotation."""
    c = R.order_corners(pts)
    H, S = R.homography(None, out_size=out_size, corners=c)
    crop = cv2.warpPerspective(img, H, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, c, S


def rectify_stored_order(img, pts, out_size=None):
    """Alternative convention (NOT rectify.py's): stored point1..4 -> (0,0),(S,0),(S,S),(0,S).
    Used only as evidence for which convention de-rotates (see README)."""
    p = np.asarray(pts, np.float32)
    S = int(out_size) if out_size is not None else R.default_out_size(p)
    dst = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    Hm = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(img, Hm, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------- k recovery
def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def k_geometric(src_pts, aug_pts):
    """(a) Geometric: angle of each STORED point about the polygon centroid (y down, so the
    angle grows clockwise on screen), aug minus source.  Albumentations keeps keypoint order,
    so stored point m is the same physical corner in both.  Returns
      k_deg      CCW rotation (np.rot90 convention) source -> aug, from the circular mean
      resid_deg  max |per-point difference - nearest multiple of 90| (consistency, small if OK)
      idx_src, idx_aug  canonical (TL=0,TR=1,BR=2,BL=3) slot of stored point 1."""
    s = np.asarray(src_pts, float)
    a = np.asarray(aug_pts, float)
    ts = np.degrees(np.arctan2(s[:, 1] - s[:, 1].mean(), s[:, 0] - s[:, 0].mean()))
    ta = np.degrees(np.arctan2(a[:, 1] - a[:, 1].mean(), a[:, 0] - a[:, 0].mean()))
    d = _wrap180(ta - ts)                                # clockwise-positive on screen
    mean = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(d)))))
    j_cw = int(np.round(mean / 90.0)) % 4
    resid = float(np.max(np.abs(_wrap180(d - 90.0 * j_cw))))
    k_deg = (-90 * j_cw) % 360

    def slot(pts, m=0):
        c = R.order_corners(pts)
        return int(np.argmin(np.linalg.norm(c - np.asarray(pts, float)[m], axis=1)))
    return dict(k_deg=int(k_deg), resid_deg=resid, idx_src=slot(s), idx_aug=slot(a))


FEAT_N = 128          # NCC working resolution (per 1-inch crop side)
FEAT_MARGIN = 0.08    # ignore this fraction at each border (resampling / border replicate)
FEAT_SIGMA = 2.0      # Gaussian low-pass at 256 px/inch before the 128 px/inch features
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def ncc_features(crop_bgr, sigma=FEAT_SIGMA):
    """Two standardisable feature maps (FEAT_N x FEAT_N) of a rectified crop: CLAHE grey and
    gradient magnitude, both low-passed (sigma px at 256 px/inch) so that the thread texture,
    which aliases at 128 px/inch for fine fabrics and makes the NCC sensitive to sub-pixel
    offsets, does not dominate; what is matched is the illumination / glare / fold / defect
    layout, which is what tells the four rotations apart."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA if g.shape[0] > 256 else cv2.INTER_CUBIC)
    c = cv2.GaussianBlur(_CLAHE.apply(g).astype(np.float32), (0, 0), sigma)
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    gm = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)),
                          (0, 0), sigma)
    m = int(round(FEAT_N * FEAT_MARGIN))
    return [cv2.resize(f, (FEAT_N, FEAT_N), interpolation=cv2.INTER_AREA) for f in (c, gm)], m


def source_rotations(img, corners_canonical, S=None):
    """rot90^j (CCW, j = 0..3) of the source crop, made by re-rectifying the source image
    with the canonical corners rolled by j (corner j -> output top-left) instead of
    np.rot90 of the pixel array, so all four hypotheses are on exactly the same pixel grid
    as a crop rectified by rectify.py (np.rot90 of an S x S crop that spans 0..S is off by
    one pixel for j != 0).  j = 0 is identical to rectify.py's crop."""
    out = []
    for j in range(4):
        H, S_ = R.homography(None, out_size=S, corners=np.roll(np.asarray(corners_canonical), -j, axis=0))
        out.append(cv2.warpPerspective(img, H, (S_, S_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def _std(f, m):
    f = f[m:FEAT_N - m, m:FEAT_N - m].astype(np.float64)
    return (f - f.mean()) / (f.std() + 1e-9)


def ncc_rot(src_rot_feats, aug_feats, m):
    """NCC between the aug crop and each rotation hypothesis j (rows), per feature
    [clahe, gradmag] and their mean (cols)."""
    res = np.zeros((4, 3))
    a = [_std(f, m) for f in aug_feats]
    for j in range(4):
        for q in range(2):
            res[j, q] = float(np.mean(_std(src_rot_feats[j][q], m) * a[q]))
        res[j, 2] = res[j, :2].mean()
    return res


# ---------------------------------------------------------------------------- transform recovery
def rot90_points(pts, k_ccw, size):
    """Albumentations / numpy rot90 keypoint map for an image of size (W, H) (pixel centres,
    0-based):  factor 1: (x, y) -> (y, W-1-x); 2: (W-1-x, H-1-y); 3: (H-1-y, x)."""
    W, H = size
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    f = (k_ccw // 90) % 4
    if f == 0:
        return p.copy()
    if f == 1:
        return np.stack([y, W - 1 - x], 1)
    if f == 2:
        return np.stack([W - 1 - x, H - 1 - y], 1)
    return np.stack([H - 1 - y, x], 1)


def residual_transform(src_pts, src_size, aug_pts, k_deg):
    """Geometry left after undoing the 90-degree rotation: rotated-source -> aug corner map.
    Fits an affine map to the 4 correspondences (least squares):
      affine_scale  sqrt|det A|,  aniso  ratio of singular values of A,
      shift_px      displacement of the window centroid (aug px), persp_rms_px  RMS residual
      of the affine fit
      (> ~1.5 px means a projective component, i.e. A.Perspective fired; the int truncation
      of the stored key points alone gives < ~0.7 px)."""
    p = rot90_points(src_pts, k_deg, src_size)
    q = np.asarray(aug_pts, float)
    X = np.hstack([p, np.ones((4, 1))])
    M, *_ = np.linalg.lstsq(X, q, rcond=None)       # 3x2
    A = M[:2].T
    t = M[2]
    res = X @ M - q
    sv = np.linalg.svd(A, compute_uv=False)
    return dict(affine_scale=float(np.sqrt(abs(np.linalg.det(A)))), aniso=float(sv[0] / sv[1]),
                shift_px=float(np.linalg.norm(q.mean(0) - p.mean(0))),
                persp_rms_px=float(np.sqrt(np.mean(np.sum(res ** 2, 1)))))


def photometric_fit(src_aligned, aug_crop, n=256, sigma=3.0):
    """aug_grey ~= gain * src_grey + offset, src_aligned = the source crop already rotated by
    the recovered k (source_rotations()[k // 90]).  Fitted on low-passed grey images (sigma px
    at 256 px/inch) so the residual thread misalignment does not bias the slope, as the
    geometric-mean (reduced major axis) slope sign(r) * sd(aug) / sd(src) over the inner
    84 % of unsaturated pixels.  RandomBrightnessContrast(0.3, 0.2) gives gain in [0.8, 1.2]
    and a shift of up to +-0.3 * 255; 'not applied' shows up as gain ~1 and offset ~0.
    offset is reported at mid-grey (128) so it is not entangled with the gain."""
    def grey(c):
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA if g.shape[0] > n else cv2.INTER_CUBIC)
        return g.astype(np.float64)
    gs, ga = grey(src_aligned), grey(aug_crop)
    sat = (gs <= 3) | (gs >= 252) | (ga <= 3) | (ga >= 252)
    sat = cv2.dilate(sat.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bs, ba = cv2.GaussianBlur(gs, (0, 0), sigma), cv2.GaussianBlur(ga, (0, 0), sigma)
    m = int(0.08 * n)
    ok = ~sat[m:-m, m:-m]
    x, y = bs[m:-m, m:-m][ok], ba[m:-m, m:-m][ok]
    if x.size < 500 or x.std() < 1e-6:
        return dict(gain=np.nan, offset=np.nan, r=np.nan)
    r = float(np.corrcoef(x, y)[0, 1])
    g = float(np.sign(r) * y.std() / x.std())
    o = float(y.mean() - g * x.mean())
    return dict(gain=g, offset=g * 128 + o - 128, r=r)


def noise_sigma(grey) -> float:
    """Immerkaer (1996) fast noise-sigma estimate (grey levels) of a full image."""
    g = np.asarray(grey, np.float64)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    conv = cv2.filter2D(g, -1, M)[1:-1, 1:-1]
    H, W = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (W - 2) * (H - 2)))


def jpeg_blockiness(grey) -> float:
    """Mean |pixel step| across the 8x8 JPEG block boundaries / elsewhere (1.0 = no grid)."""
    g = np.asarray(grey, np.float64)
    dx = np.abs(np.diff(g, axis=1)).mean(0)
    dy = np.abs(np.diff(g, axis=0)).mean(1)
    bx = dx[7::8].mean() / np.delete(dx, np.s_[7::8]).mean()
    by = dy[7::8].mean() / np.delete(dy, np.s_[7::8]).mean()
    return float((bx + by) / 2)


def stale_points(src_pts, aug_pts):
    """Indices (1-based) of stored aug points that are bit-identical to the SOURCE's point at
    the same index while the polygon as a whole is not identical.  This is the signature of
    the 2024 notebook's key-point-drop bug: albumentations removes key points that leave the
    frame, update_xml_with_new_bbox_and_dimensions() enumerates the survivors from 1, and the
    trailing <xN>/<yN> tags keep the SOURCE values -> the polygon mixes transformed and
    untransformed corners (convex and in bounds, so the 2024 checks let it through)."""
    s, a = np.asarray(src_pts, float), np.asarray(aug_pts, float)
    eq = [m + 1 for m in range(4) if np.all(s[m] == a[m])]
    return [] if len(eq) == 4 else eq

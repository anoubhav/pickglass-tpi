"""degrade.py -- controlled capture-quality degradations for the WVAQ ablations.

All severities are defined PHYSICALLY, relative to the 1-inch window, so that a level means the
same thing for a 480 px and a 1950 px crop:  "u" = 1/1000 inch (one milli-inch).  Typical yarn
periods are 5-25 u (40-200 threads per inch).

    apply(crop_bgr, kind, level, seed=0) -> degraded crop (same size unless kind == 'down')

kind        level meaning                                        default levels
---------   -------------------------------------------------    -----------------------
gblur       Gaussian defocus sigma, in u                           1, 2, 3, 4, 6
motion      linear motion-blur length, in u (direction 30 deg)     5, 10, 20, 30
down        down-scale factor (INTER_AREA), crop stays smaller     2, 4, 8
jpeg        JPEG quality (re-encode of the crop)                   90, 50, 20
noise       additive Gaussian noise sigma, 8-bit grey levels       5, 10, 20, 40
glare       area fraction of a synthetic saturated LED hot spot    0.05, 0.10, 0.20, 0.30
shadow      area fraction covered by a hard cast shadow (x0.3)     0.25, 0.50
corners     window-corner perturbation, fraction of side length    0.02, 0.05, 0.10
            (needs the original photo + corners: see perturb_corners / rectify_with_corners;
             for a crop without a photo, the crop's own corners are perturbed)

The random parts (glare centre, shadow edge, noise, corner offsets) are seeded per image so
that every estimator sees exactly the same degraded pixels.
"""
from __future__ import annotations

import math
import zlib

import cv2
import numpy as np

LEVELS = {
    "gblur": [1, 2, 3, 4, 6],
    "motion": [5, 10, 20, 30],
    "down": [2, 4, 8],
    "jpeg": [90, 50, 20],
    "noise": [5, 10, 20, 40],
    "glare": [0.05, 0.10, 0.20, 0.30],
    "shadow": [0.25, 0.50],
    "corners": [0.02, 0.05, 0.10],
}
UNITS = {"gblur": "sigma [1e-3 in]", "motion": "length [1e-3 in]", "down": "factor", "jpeg": "quality",
         "noise": "sigma [grey levels]", "glare": "saturated area fraction", "shadow": "shadowed area fraction",
         "corners": "corner error [fraction of side]"}
# order of increasing severity for plotting (jpeg quality decreases)
SEVERITY_SIGN = {"jpeg": -1}


def _rng(seed, kind):
    # deterministic across processes (built-in hash() of str is salted per process)
    return np.random.default_rng(zlib.crc32(f"{int(seed)}:{kind}".encode()))


def gaussian_blur(img, sigma_u):
    S = img.shape[1]
    s = sigma_u * S / 1000.0
    return cv2.GaussianBlur(img, (0, 0), s) if s > 0.05 else img.copy()


def motion_blur(img, length_u, angle_deg=30.0):
    S = img.shape[1]
    L = max(1.0, length_u * S / 1000.0)
    n = int(math.ceil(L)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    t = np.linspace(-L / 2, L / 2, max(2, int(4 * L)))
    xs = c + t * math.cos(math.radians(angle_deg))
    ys = c - t * math.sin(math.radians(angle_deg))
    for x, y in zip(xs, ys):  # bilinear splat of the line
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < n and 0 <= y0 + dy < n:
                k[y0 + dy, x0 + dx] += w
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def downscale(img, factor):
    H, W = img.shape[:2]
    return cv2.resize(img, (max(8, int(round(W / factor))), max(8, int(round(H / factor)))),
                      interpolation=cv2.INTER_AREA)


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def noise(img, sigma, seed=0):
    r = _rng(seed, "noise")
    n = r.normal(0, sigma, img.shape[:2])[..., None]  # luminance-like noise (same on all channels)
    n = n + r.normal(0, 0.3 * sigma, img.shape)        # plus some chroma noise
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def glare(img, area_frac, seed=0):
    """LED-like specular hot spot: a saturated elliptical core of the requested area with a
    smooth additive halo (twice the core radius), centre random in the central 60 %."""
    H, W = img.shape[:2]
    r = _rng(seed, "glare")
    cx, cy = W * r.uniform(0.2, 0.8), H * r.uniform(0.2, 0.8)
    ecc = r.uniform(0.7, 1.0)
    a = math.sqrt(area_frac * W * H / (math.pi * ecc))  # core semi-axes a, a*ecc
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / (a * ecc)) ** 2)
    add = np.where(d <= 1.0, 255.0, 255.0 * np.exp(-((d - 1.0) ** 2) / (2 * 0.5 ** 2)))
    out = img.astype(np.float32) + add[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def shadow(img, area_frac, seed=0, factor=0.3):
    """Hard cast shadow (phone / hand): a half-plane with a slightly soft edge (S/150) covering
    the requested area fraction, at a random orientation; intensity x factor inside."""
    H, W = img.shape[:2]
    r = _rng(seed, "shadow")
    th = r.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    proj = (xx - W / 2) * math.cos(th) + (yy - H / 2) * math.sin(th)
    thr = np.quantile(proj, 1 - area_frac)
    soft = max(1.0, W / 150.0)
    m = 1.0 / (1.0 + np.exp(np.clip(-(proj - thr) / soft, -60, 60)))  # 1 inside shadow
    gain = 1.0 - (1.0 - factor) * m
    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def perturb_corners(corners, frac, seed=0):
    """Each corner moved by a random vector of length frac * mean side length."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    side = float(np.linalg.norm(c - np.roll(c, -1, 0), axis=1).mean())
    r = _rng(seed, "corners")
    ang = r.uniform(0, 2 * math.pi, 4)
    return c + frac * side * np.stack([np.cos(ang), np.sin(ang)], 1)


def rectify_with_corners(photo, corners, S):
    """Same convention as common/rectify.py (TL,TR,BR,BL -> S x S, bilinear, replicate)."""
    c = np.asarray(corners, np.float32)
    dst = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    Hm = cv2.getPerspectiveTransform(c, dst)
    return cv2.warpPerspective(photo, Hm, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def corners_on_crop(crop, frac, seed=0):
    """Corner perturbation when only the crop is available (legacy images): the crop's own
    corners are perturbed and the crop re-warped to its original size."""
    H, W = crop.shape[:2]
    c = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    pc = perturb_corners(c, frac, seed)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    Hm = cv2.getPerspectiveTransform(pc.astype(np.float32), dst)
    return cv2.warpPerspective(crop, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply(img, kind, level, seed=0, photo=None, corners=None):
    """Degrade a rectified crop.  For kind == 'corners' with (photo, corners) given, the window
    is re-rectified from the original photo with perturbed corners (output size unchanged)."""
    if kind in (None, "", "clean", "none"):
        return img
    if kind == "gblur":
        return gaussian_blur(img, level)
    if kind == "motion":
        return motion_blur(img, level)
    if kind == "down":
        return downscale(img, level)
    if kind == "jpeg":
        return jpeg(img, level)
    if kind == "noise":
        return noise(img, level, seed)
    if kind == "glare":
        return glare(img, level, seed)
    if kind == "shadow":
        return shadow(img, level, seed)
    if kind == "corners":
        if photo is not None and corners is not None:
            return rectify_with_corners(photo, perturb_corners(corners, level, seed), img.shape[1])
        return corners_on_crop(img, level, seed)
    raise ValueError(kind)


def all_conditions(kinds=None):
    """[(kind, level)] including ('clean', 0)."""
    out = [("clean", 0)]
    for k in (kinds or LEVELS):
        out += [(k, v) for v in LEVELS[k]]
    return out


if __name__ == "__main__":  # contact sheet of all degradations on one crop
    import os
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotation_tool", "data", "crops", "110.jpg")
    out = sys.argv[2] if len(sys.argv) > 2 else "degrade_sheet.jpg"
    im = cv2.imread(p)
    tiles = []
    for k, v in all_conditions():
        d = apply(im, k, v, seed=1)
        S = d.shape[0]
        z = d[S // 2 - S // 8:S // 2 + S // 8, S // 2 - S // 8:S // 2 + S // 8]
        t = np.hstack([cv2.resize(d, (240, 240), interpolation=cv2.INTER_AREA),
                       cv2.resize(z, (240, 240), interpolation=cv2.INTER_NEAREST)])
        cv2.rectangle(t, (0, 0), (200, 18), (0, 0, 0), -1)
        cv2.putText(t, f"{k} {v}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print("wrote", out)

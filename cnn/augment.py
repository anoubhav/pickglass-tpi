"""augment.py -- label-aware augmentation library for 1-inch pick-glass crops (WVAQ 2027).

Every transform has the signature

    img_out, lt = T(img, *, extent=(1.0, 1.0), rng=None, **strength)

``img`` is an HxWx3 uint8 BGR crop that covers ``extent`` = (wx, wy) inches of fabric (a rectified
pick-glass crop covers (1, 1)).  ``lt`` is a :class:`LabelTransform` that maps the label
(h_tpi, v_tpi, weave) of the INPUT image to the label of the OUTPUT image.  Transforms compose with
:func:`compose` (or :class:`Pipeline`), which threads the physical extent through.

Label conventions
-----------------
h_tpi  vertical yarns per inch, counted walking horizontally (CNN ``h``, ours ``h_count``, annotation
       tool ``h_positions``);  v_tpi  horizontal yarns per inch, counted walking vertically.
Two flavours of a per-inch label exist and the library returns both:

* density   intrinsic yarns per inch measured perpendicular to the yarns (textile EPI / PPI; the
            synthetic generator's labels).  ``lt(label)`` == ``lt.density(label)``.
* crossing  yarns crossing a line parallel to the image axis, per inch of that line (what the
            annotation tool's guide line and ours' ``h_count`` measure) = density * cos(yarn tilt).
            ``lt.crossing(label, angles=(ax, ay))``; the yarn tilt of the input image is needed for
            the exact value (e.g. ours' lattice ``angle_x, angle_y``); without it, 0 deg is assumed.

Per-crop counts (what a counter run on the output image returns) = per-inch x physical extent:
``lt.counts(label)``.  Per-pixel yarn frequency = per-inch / pixels-per-inch (``lt.ppi_out``).

Yarn-tilt angles use ours' convention (degrees, image y axis pointing down):
angle_x: the h family's reciprocal vector k_h ~ (cos ax, sin ax);  angle_y: the v family's k_v ~
(sin ay, cos ay).

How labels are transformed (one engine for every geometric transform)
--------------------------------------------------------------------
Each LabelTransform stores the exact inverse pixel map ``inv`` (output px -> input px).  Straight,
uniformly spaced yarns are the level sets of a phase  phi(p) = k . p.  The number of yarns crossing a
monotone curve equals |phi(end) - phi(start)|, so the crossing count of every output row / column is
computed exactly from the preimages of its two end points (rotation, flips, rot90, crop, zoom, tiling
and perspective alike).  The intrinsic density follows from the Jacobian of ``inv`` (k_out = J^T k_in).
The output axis assignment (does h become v?) is the one with the larger crossing counts.

Families (verified empirically by verify_augment.py; safe strength ranges in augment_whitelist.json)
------------------------------------------------------------------------------------------------
(a) hflip, vflip                  labels unchanged; the twill diagonal is mirrored (Z <-> S as seen),
                                  the weave CLASS is unchanged (``lt.diag_flip``)
(b) rot90, rot270 (and rot180)    h <-> v swapped (rot180: unchanged); rot90/270 also mirror the
                                  diagonal; warp-faced <-> weft-faced twill/satin, same class
(c) rotate(deg, mode)             +-15 deg about the centre, physical scale kept: 'inner' crops the
                                  largest axis-aligned rectangle inside the rotated image (output
                                  extent = f * input, f = 1/(|cos|+|sin|) for a square, no
                                  resampling of scale); 'reflect' keeps the frame and fills the
                                  corners by reflection (approximate: mirrored corners).  Density is
                                  unchanged; crossing counts become density * cos(tilt + deg)
(d) photometric                   brightness, contrast, gamma, hue, saturation, greyscale, CLAHE,
                                  white balance -- pixel-wise colour maps, labels unchanged
(e) sensor / compression          Gaussian noise, JPEG q20-95, mild Gaussian blur (sigma in input px,
                                  or as a fraction of the yarn pitch), short motion blur, unsharp mask
(f) illumination                  specular glare blobs, soft / deep shadows, vignetting
(g) zoom(s) / zoom_xy(sx, sy)     s >= 1: crop a window of known physical extent 1/s (in input
                                  extent units) and resize it to the input pixel size; s < 1:
                                  down-sample the whole crop.  Density per inch unchanged; per-crop
                                  count scales with the physical width (x 1/s for s >= 1, x 1 for
                                  s < 1); per-pixel frequency scales by 1/s; ppi_out = s * ppi_in
(h) tile(size_in)                 physical-size tiles (0.25 / 0.5 inch): density unchanged, per-tile
                                  count = density x tile width
(i) perspective(eps)              corner jitter <= eps (<= 3 %) of the side: a mis-rectified window.
                                  The exact per-row / per-column crossing change (~ +-eps) is
                                  computed from the homography

Verification (verify_augment.py, 150 real + 100 synthetic crops, frozen classical counter + CNN):
tier A (within the counter's clean-run noise, = DEFAULT_POLICY): flips, rot90/270, brightness x0.6,
contrast x0.6, noise sigma <= 0.01, blur sigma <= 0.5 px, unsharp amount <= 1, mild glare, soft shadow,
vignette <= 0.6.  Tier B (label map exact and unbiased on both instruments, but the counter's lattice
search is not stable under the resampling): rotate +-10 deg (start at +-5), zoom 0.5-1.25 with >= 6 px
per yarn, perspective <= 3 %, hue / saturation / greyscale / white balance, gamma, JPEG q >= 20.
See augment_whitelist.json and augment_verification.csv.

Explicitly EXCLUDED (``EXCLUDED`` below gives the reasons; verify_augment.py runs them as negative
controls): elastic / grid warps, mixup and cutmix across fabrics, blur strong enough to erase the
yarns, label-agnostic RandomResizedCrop / aspect-ratio jitter, rotations beyond the axis-count
definition range.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHITELIST_PATH = os.path.join(HERE, "augment_whitelist.json")

EXCLUDED = {
    "elastic": "Elastic / grid-distortion warps move yarns by a spatially varying field: local yarn spacing "
               "(the quantity being labelled) changes non-uniformly, so a per-tile density label no longer "
               "exists, and yarn curvature that is not in the fabric is introduced.  Only the whole-row "
               "integral of the crossing count survives, which is useless for tile-level training.",
    "mixup": "Mixup across fabrics superimposes two yarn lattices: the image has two (h, v) pairs and two "
             "weaves, and a convex combination of the labels is not the count of anything a counter or a "
             "person would measure.",
    "cutmix": "CutMix pastes a second fabric: the count along a guide line becomes a mixture weighted by "
              "where the line runs, tile labels are undefined near the seam, and the weave label is "
              "ambiguous.",
    "strong_blur": "Blur with sigma comparable to the yarn pitch (MTF at the yarn frequency "
                   "exp(-2 pi^2 sigma^2 / pitch^2) < ~0.1 for sigma > 0.35 pitch) erases the yarns: the "
                   "label is unchanged in principle but no longer observable, so it becomes label noise.",
    "label_agnostic_resized_crop": "torchvision-style RandomResizedCrop / aspect-ratio jitter WITHOUT updating "
                                   "the physical extent silently rescales the per-inch density (a 0.7-inch "
                                   "window resized to the full frame reads as 0.7x the true TPI).  Use zoom() "
                                   "/ zoom_xy(), which carry the extent.",
    "large_rotation": "Rotations beyond ~15 deg (on top of the fabric's own tilt) make the axis-count definition "
                      "ambiguous (crossing count = density cos(tilt) drifts by > 3 %) and leave the "
                      "near-axis search range of the counters; 90-degree multiples are handled exactly by "
                      "rot90/rot270.",
}


# =============================================================================================
# label transform
# =============================================================================================
def _ident(p):
    return p


def _rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


@dataclass
class LabelTransform:
    """Label bookkeeping for one (possibly composed) transform.  See the module docstring."""

    name: str
    in_size: tuple                      # (W, H) px of the input image
    out_size: tuple                     # (W, H) px of the output image
    ppi_in: tuple                       # (px per inch along x, along y) of the input
    ppi_out: tuple                      # ... of the output
    inv: Callable = _ident              # (N, 2) output px -> input px (pixel centres at integers)
    diag_flip: bool = False             # twill diagonal mirrored in image space (class unchanged)
    exact: bool = True                  # exact for straight, uniformly spaced yarns
    status: str = "exact"               # 'exact' | 'approx' | 'excluded'
    weave_defined: bool = True          # False for mixup / cutmix
    params: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- geometry helpers
    @property
    def extent_in(self):
        return (self.in_size[0] / self.ppi_in[0], self.in_size[1] / self.ppi_in[1])

    @property
    def extent_out(self):
        return (self.out_size[0] / self.ppi_out[0], self.out_size[1] / self.ppi_out[1])

    def _k_in(self, label, angles, kind):
        """Reciprocal vectors (cycles / input px) of the two yarn families."""
        h, v = float(label[0]), float(label[1])
        ax, ay = math.radians(angles[0]), math.radians(angles[1])
        if kind == "crossing":           # label = k_x of the h family, k_y of the v family
            kh = np.array([h, h * math.tan(ax)])
            kv = np.array([v * math.tan(ay), v])
        elif kind == "density":          # label = |k|
            kh = h * np.array([math.cos(ax), math.sin(ax)])
            kv = v * np.array([math.sin(ay), math.cos(ay)])
        else:
            raise ValueError(kind)
        s = np.array(self.ppi_in, float)
        return kh / s, kv / s

    def _crossings(self, k_px, n=33):
        """Mean crossings per output inch along output rows (x) and columns (y) for reciprocal k_px."""
        W, H = self.out_size
        ys = np.linspace(0, H - 1, n)
        xs = np.linspace(0, W - 1, n)
        r0 = self.inv(np.stack([np.full(n, -0.5), ys], 1))
        r1 = self.inv(np.stack([np.full(n, W - 0.5), ys], 1))
        c0 = self.inv(np.stack([xs, np.full(n, -0.5)], 1))
        c1 = self.inv(np.stack([xs, np.full(n, H - 0.5)], 1))
        cx = np.abs((r1 - r0) @ k_px).mean() / self.extent_out[0]
        cy = np.abs((c1 - c0) @ k_px).mean() / self.extent_out[1]
        return float(cx), float(cy)

    def jacobian(self):
        """d(input inch) / d(output inch) at the output centre (numerical)."""
        W, H = self.out_size
        c = np.array([[(W - 1) / 2, (H - 1) / 2]])
        e = 0.5
        P = np.concatenate([c, c + [e, 0], c + [0, e]])
        Q = self.inv(P) / np.array(self.ppi_in, float)
        dx = (Q[1] - Q[0]) / (e / self.ppi_out[0])
        dy = (Q[2] - Q[0]) / (e / self.ppi_out[1])
        return np.stack([dx, dy], 1)          # columns: d in / d out_x, d in / d out_y

    def _swap(self, kh_px, kv_px):
        hx, hy = self._crossings(kh_px)
        vx, vy = self._crossings(kv_px)
        return (hx * vy) < (vx * hy), (hx, hy, vx, vy)

    @property
    def swaps_axes(self) -> bool:
        kh, kv = self._k_in((1.0, 1.0), (0.0, 0.0), "density")
        return self._swap(kh, kv)[0]

    # ---------------------------------------------------------------- label maps
    def _weave(self, label):
        w = label[2] if len(label) > 2 else None
        return w if self.weave_defined else None

    def density(self, label, angles=(0.0, 0.0), kind="density"):
        """Intrinsic per-inch densities of the output, assigned to the output axes."""
        kh, kv = self._k_in(label, angles, kind)
        sw, _ = self._swap(kh, kv)
        J = self.jacobian()
        s = np.array(self.ppi_in, float)
        dh = float(np.linalg.norm(J.T @ (kh * s)))
        dv = float(np.linalg.norm(J.T @ (kv * s)))
        h, v = (dv, dh) if sw else (dh, dv)
        return (h, v, self._weave(label))

    __call__ = density

    def crossing(self, label, angles=(0.0, 0.0), kind="crossing", return_angles=False):
        """Per-inch crossing counts along the output axes (exact for straight uniform yarns).

        label: (h, v[, weave]) of the input, interpreted as crossing counts (kind='crossing') or
        intrinsic densities (kind='density'); angles: yarn tilt of the input (ours' convention)."""
        kh, kv = self._k_in(label, angles, kind)
        sw, (hx, hy, vx, vy) = self._swap(kh, kv)
        h, v = (vx, hy) if sw else (hx, vy)
        if not return_angles:
            return (h, v, self._weave(label))
        J = self.jacobian()
        s = np.array(self.ppi_in, float)
        ko_h, ko_v = J.T @ (kh * s), J.T @ (kv * s)
        if sw:
            ko_h, ko_v = ko_v, ko_h
        ko_h = ko_h if ko_h[0] >= 0 else -ko_h
        ko_v = ko_v if ko_v[1] >= 0 else -ko_v
        a_out = (math.degrees(math.atan2(ko_h[1], ko_h[0])), math.degrees(math.atan2(ko_v[0], ko_v[1])))
        return (h, v, self._weave(label)), a_out

    def counts(self, label, angles=(0.0, 0.0), kind="crossing", out="crossing"):
        """Per-crop counts on the output image (= per-inch x physical extent of the output)."""
        f = self.crossing if out == "crossing" else self.density
        h, v, w = f(label, angles, kind)
        ex, ey = self.extent_out
        return (h * ex, v * ey, w)

    def factors(self, angles=(0.0, 0.0)):
        """Convenience: (swap, crossing factor x, y, density factor x, y, extent x, y), all relative to
        the input label of the family that lands on that output axis."""
        sw = self.swaps_axes
        hc, vc, _ = self.crossing((1.0, 1.0), angles)
        hd, vd, _ = self.density((1.0, 1.0), angles, kind="crossing")
        return dict(swap=sw, cross_x=hc, cross_y=vc, dens_x=hd, dens_y=vd,
                    extent_x=self.extent_out[0], extent_y=self.extent_out[1])

    def describe(self):
        f = self.factors()
        return (f"{self.name}: in {self.in_size} @ {tuple(round(p, 1) for p in self.ppi_in)} ppi -> out "
                f"{self.out_size} @ {tuple(round(p, 1) for p in self.ppi_out)} ppi, extent "
                f"{tuple(round(e, 4) for e in self.extent_out)} in, swap={f['swap']}, crossing x"
                f"({f['cross_x']:.4f}, {f['cross_y']:.4f}), density x({f['dens_x']:.4f}, {f['dens_y']:.4f}), "
                f"diag_flip={self.diag_flip}, status={self.status}")


def identity_lt(img, extent=(1.0, 1.0), name="identity", status="exact", **params) -> LabelTransform:
    H, W = img.shape[:2]
    ppi = (W / extent[0], H / extent[1])
    return LabelTransform(name, (W, H), (W, H), ppi, ppi, _ident, status=status, params=params)


def compose(lt1: LabelTransform, lt2: LabelTransform) -> LabelTransform:
    """lt2 after lt1."""
    if tuple(lt1.out_size) != tuple(lt2.in_size):
        raise ValueError(f"size mismatch {lt1.out_size} -> {lt2.in_size}")
    i1, i2 = lt1.inv, lt2.inv
    order = {"exact": 0, "approx": 1, "excluded": 2}
    st = max((lt1.status, lt2.status), key=order.get)
    return LabelTransform(f"{lt1.name}|{lt2.name}", lt1.in_size, lt2.out_size, lt1.ppi_in, lt2.ppi_out,
                          lambda p: i1(i2(p)), lt1.diag_flip ^ lt2.diag_flip, lt1.exact and lt2.exact, st,
                          lt1.weave_defined and lt2.weave_defined, {"steps": [lt1.params, lt2.params]})


def _affine_lt(name, img, out_size, A, extent, ppi_out=None, **kw):
    """A: 2x3 forward map input px -> output px."""
    H, W = img.shape[:2]
    ppi_in = (W / extent[0], H / extent[1])
    Ai = cv2.invertAffineTransform(np.asarray(A, np.float64))

    def inv(p, Ai=Ai):
        p = np.asarray(p, float)
        return p @ Ai[:, :2].T + Ai[:, 2]

    return LabelTransform(name, (W, H), tuple(out_size), ppi_in, ppi_out or ppi_in, inv, **kw)


# =============================================================================================
# (a) flips, (b) 90-degree rotations
# =============================================================================================
def hflip(img, extent=(1.0, 1.0), rng=None):
    H, W = img.shape[:2]
    A = np.array([[-1, 0, W - 1], [0, 1, 0]], float)
    return np.ascontiguousarray(img[:, ::-1]), _affine_lt("hflip", img, (W, H), A, extent, diag_flip=True)


def vflip(img, extent=(1.0, 1.0), rng=None):
    H, W = img.shape[:2]
    A = np.array([[1, 0, 0], [0, -1, H - 1]], float)
    return np.ascontiguousarray(img[::-1]), _affine_lt("vflip", img, (W, H), A, extent, diag_flip=True)


def rot90(img, extent=(1.0, 1.0), rng=None, k=1):
    """Counter-clockwise (as displayed) rotation by k*90 deg (np.rot90 convention)."""
    k = k % 4
    H, W = img.shape[:2]
    out = np.ascontiguousarray(np.rot90(img, k))
    if k == 0:
        return out, identity_lt(img, extent, "rot0")
    if k == 1:      # dst[i, j] = src[j, W-1-i]:  x_out = y_in, y_out = W-1-x_in
        A = np.array([[0, 1, 0], [-1, 0, W - 1]], float)
    elif k == 2:
        A = np.array([[-1, 0, W - 1], [0, -1, H - 1]], float)
    else:           # dst[i, j] = src[H-1-j, i]
        A = np.array([[0, -1, H - 1], [1, 0, 0]], float)
    Ho, Wo = out.shape[:2]
    ppi_in = (W / extent[0], H / extent[1])
    ppi_out = (ppi_in[1], ppi_in[0]) if k % 2 else ppi_in
    return out, _affine_lt(f"rot{90 * k}", img, (Wo, Ho), A, extent, ppi_out, diag_flip=bool(k % 2))


def rot270(img, extent=(1.0, 1.0), rng=None):
    return rot90(img, extent, rng, k=3)


def rot180(img, extent=(1.0, 1.0), rng=None):
    return rot90(img, extent, rng, k=2)


# =============================================================================================
# (c) small rotations that keep the physical scale
# =============================================================================================
def inner_rect_factor(W, H, deg):
    """Largest f such that an axis-aligned (fW x fH) rectangle fits inside the W x H image rotated
    by deg about its centre."""
    c, s = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    return min(W / (W * c + H * s), H / (W * s + H * c))


def rotate(img, extent=(1.0, 1.0), rng=None, deg=0.0, mode="inner", interp=cv2.INTER_CUBIC):
    """Rotate by ``deg`` (counter-clockwise as displayed, cv2 convention) about the centre.

    mode='inner'   : output = largest centred axis-aligned rectangle inside the rotated image, at the
                     input's px/inch (no scale change); extent_out = f * extent_in.
    mode='reflect' : output keeps the input frame; corners are filled by reflection (approximate)."""
    H, W = img.shape[:2]
    cin = ((W - 1) / 2.0, (H - 1) / 2.0)
    R = cv2.getRotationMatrix2D(cin, float(deg), 1.0)
    if mode == "inner":
        f = inner_rect_factor(W, H, deg)
        Wo, Ho = max(8, int(math.floor(W * f))), max(8, int(math.floor(H * f)))
        R[:, 2] += np.array([(Wo - 1) / 2.0 - cin[0], (Ho - 1) / 2.0 - cin[1]])
        out = cv2.warpAffine(img, R, (Wo, Ho), flags=interp, borderMode=cv2.BORDER_REFLECT_101)
        lt = _affine_lt(f"rotate{deg:+g}_inner", img, (Wo, Ho), R, extent, params={"deg": deg, "mode": mode})
    elif mode == "reflect":
        out = cv2.warpAffine(img, R, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT_101)
        lt = _affine_lt(f"rotate{deg:+g}_reflect", img, (W, H), R, extent, exact=False, status="approx",
                        params={"deg": deg, "mode": mode})
    else:
        raise ValueError(mode)
    return out, lt


# =============================================================================================
# (d) photometric
# =============================================================================================
def _f(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def brightness(img, extent=(1.0, 1.0), rng=None, factor=1.0):
    return _u8(_f(img) * factor), identity_lt(img, extent, f"brightness{factor:g}", factor=factor)


def contrast(img, extent=(1.0, 1.0), rng=None, factor=1.0):
    x = _f(img)
    m = float(x.mean())
    return _u8((x - m) * factor + m), identity_lt(img, extent, f"contrast{factor:g}", factor=factor)


def gamma(img, extent=(1.0, 1.0), rng=None, g=1.0):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut), identity_lt(img, extent, f"gamma{g:g}", g=g)


def hue_shift(img, extent=(1.0, 1.0), rng=None, deg=0.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = (hsv[..., 0].astype(np.int32) + int(round(deg / 2.0))) % 180
    hsv[..., 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), identity_lt(img, extent, f"hue{deg:+g}", deg=deg)


def saturation(img, extent=(1.0, 1.0), rng=None, factor=1.0):
    x = _f(img)
    g = x.mean(-1, keepdims=True)
    return _u8(g + (x - g) * factor), identity_lt(img, extent, f"saturation{factor:g}", factor=factor)


def greyscale(img, extent=(1.0, 1.0), rng=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), identity_lt(img, extent, "greyscale")


def clahe(img, extent=(1.0, 1.0), rng=None, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), identity_lt(img, extent, f"clahe{clip:g}", clip=clip)


def white_balance(img, extent=(1.0, 1.0), rng=None, gains=(1.0, 1.0, 1.0)):
    """gains in B, G, R order."""
    x = _f(img) * np.asarray(gains, np.float32)[None, None, :]
    return _u8(x), identity_lt(img, extent, "wb" + "/".join(f"{g:g}" for g in gains), gains=tuple(gains))


# =============================================================================================
# (e) sensor / compression
# =============================================================================================
def gaussian_noise(img, extent=(1.0, 1.0), rng=None, sigma=0.02):
    """Additive Gaussian noise, sigma in [0, 1] intensity units (independent per channel)."""
    r = _rng(rng)
    x = _f(img) + r.normal(0.0, sigma, img.shape).astype(np.float32)
    return _u8(x), identity_lt(img, extent, f"noise{sigma:g}", sigma=sigma)


def jpeg(img, extent=(1.0, 1.0), rng=None, q=75):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), identity_lt(img, extent, f"jpeg{int(q)}", q=int(q))


def gaussian_blur(img, extent=(1.0, 1.0), rng=None, sigma=1.0):
    """sigma in INPUT pixels.  Physically meaningful scale: sigma / yarn pitch (keep <= ~0.15)."""
    if sigma <= 0:
        return img.copy(), identity_lt(img, extent, "blur0")
    return cv2.GaussianBlur(img, (0, 0), float(sigma)), identity_lt(img, extent, f"blur{sigma:g}", sigma=sigma)


def motion_blur(img, extent=(1.0, 1.0), rng=None, length=5, angle=None):
    """Linear motion blur over ``length`` px at ``angle`` deg (random if None)."""
    r = _rng(rng)
    ang = float(r.uniform(0, 180)) if angle is None else float(angle)
    L = max(1, int(round(length)))
    k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
    c = L
    t = np.linspace(-(L - 1) / 2.0, (L - 1) / 2.0, max(2, 4 * L))
    xs = c + t * math.cos(math.radians(ang))
    ys = c - t * math.sin(math.radians(ang))
    for x, y in zip(xs, ys):
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= y0 + dy < k.shape[0] and 0 <= x0 + dx < k.shape[1]:
                k[y0 + dy, x0 + dx] += w
    k /= k.sum()
    out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    return out, identity_lt(img, extent, f"motion{L}", length=L, angle=ang)


def sharpen(img, extent=(1.0, 1.0), rng=None, amount=1.0, sigma=1.0):
    """Unsharp mask: x + amount * (x - G_sigma * x)."""
    x = _f(img)
    b = cv2.GaussianBlur(x, (0, 0), float(sigma))
    return _u8(x + amount * (x - b)), identity_lt(img, extent, f"sharpen{amount:g}", amount=amount, sigma=sigma)


# =============================================================================================
# (f) illumination
# =============================================================================================
def glare(img, extent=(1.0, 1.0), rng=None, n_blobs=2, amp=0.6, radius=(0.03, 0.10)):
    """Additive specular highlights: ``n_blobs`` elliptical Gaussian blobs (radius as a fraction of
    the side) of peak amplitude ``amp`` (in [0, 1] units; >= ~0.5 saturates the core)."""
    r = _rng(rng)
    H, W = img.shape[:2]
    x = _f(img)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    add = np.zeros((H, W), np.float32)
    for _ in range(int(n_blobs)):
        cx, cy = r.uniform(0.1, 0.9) * W, r.uniform(0.1, 0.9) * H
        sx = r.uniform(*radius) * W
        sy = sx * r.uniform(0.5, 1.5)
        th = r.uniform(0, math.pi)
        dx, dy = xx - cx, yy - cy
        u = dx * math.cos(th) + dy * math.sin(th)
        v = -dx * math.sin(th) + dy * math.cos(th)
        add += float(amp) * r.uniform(0.7, 1.0) * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return _u8(x + add[..., None]), identity_lt(img, extent, f"glare{n_blobs}x{amp:g}", n_blobs=n_blobs, amp=amp)


def shadow(img, extent=(1.0, 1.0), rng=None, depth=0.5, soft=0.15):
    """Multiplicative shadow over a random half-plane: light falls to ``depth`` (0.15 = deep hand /
    phone shadow) with a soft edge of width ``soft`` x side."""
    r = _rng(rng)
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = r.uniform(0, 2 * math.pi)
    off = r.uniform(-0.25, 0.25) * W
    d = (xx - W / 2) * math.cos(a) + (yy - H / 2) * math.sin(a) - off
    m = 0.5 * (1 + np.tanh(d / max(1.0, soft * W)))
    x = _f(img) * (1 - (1 - float(depth)) * m)[..., None]
    return _u8(x), identity_lt(img, extent, f"shadow{depth:g}", depth=depth, soft=soft)


def vignette(img, extent=(1.0, 1.0), rng=None, strength=0.4):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = ((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2) / (((W - 1) / 2) ** 2 + ((H - 1) / 2) ** 2)
    x = _f(img) * (1 - float(strength) * rr)[..., None]
    return _u8(x), identity_lt(img, extent, f"vignette{strength:g}", strength=strength)


# =============================================================================================
# (g) zoom with exact physical bookkeeping, (h) physical-size tiles, (i) perspective jitter
# =============================================================================================
def zoom_xy(img, extent=(1.0, 1.0), rng=None, sx=1.0, sy=1.0, center=None):
    """Per-axis zoom by (sx, sy) with the physical extent carried along.

    s >= 1 on an axis: a window of 1/s of the input along it (physical extent extent/s) is cut at
    ``center`` (fractions; random if None and rng given, else the middle) and resized back to the
    input pixel size.  s < 1: the axis is down-sampled (INTER_AREA), extent unchanged.  In both cases
    ppi_out = s * ppi_in, density per inch is unchanged, the per-crop count scales with the physical
    width (x min(1, 1/s)) and the per-pixel frequency by 1/s."""
    H, W = img.shape[:2]
    r = _rng(rng) if (center is None and rng is not None) else None
    sx, sy = float(sx), float(sy)
    # output size: input size for s >= 1, round(s * size) for s < 1
    Wo = W if sx >= 1 else max(8, int(round(W * sx)))
    Ho = H if sy >= 1 else max(8, int(round(H * sy)))
    ex = Wo / W if sx < 1 else sx      # actual output-px per input-px scale
    ey = Ho / H if sy < 1 else sy
    # window (in input edge coordinates) for the axes that crop
    wx, wy = (W / sx if sx >= 1 else W), (H / sy if sy >= 1 else H)
    if center is None:
        center = (r.uniform(0, 1), r.uniform(0, 1)) if r is not None else (0.5, 0.5)
    ax = (W - wx) * float(center[0])
    ay = (H - wy) * float(center[1])
    # forward map (pixel centres): x_out = (x_in + .5 - a) * e - .5
    A = np.array([[ex, 0, (0.5 - ax) * ex - 0.5], [0, ey, (0.5 - ay) * ey - 0.5]], float)
    if sx < 1 and sy < 1:
        out = cv2.resize(img, (Wo, Ho), interpolation=cv2.INTER_AREA)
    elif sx >= 1 and sy >= 1:
        out = cv2.warpAffine(img, A, (Wo, Ho), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    else:   # mixed: crop/upsample one axis, area-downsample the other
        tmp = cv2.resize(img, (Wo if sx < 1 else W, Ho if sy < 1 else H), interpolation=cv2.INTER_AREA)
        A2 = A.copy()
        if sx < 1:
            A2[0] = [1, 0, 0]
        if sy < 1:
            A2[1] = [0, 1, 0]
        out = cv2.warpAffine(tmp, A2, (Wo, Ho), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    ppi_in = (W / extent[0], H / extent[1])
    ppi_out = (ppi_in[0] * ex, ppi_in[1] * ey)
    lt = _affine_lt(f"zoom{sx:g}x{sy:g}", img, (Wo, Ho), A, extent, ppi_out,
                    params={"sx": sx, "sy": sy, "center": tuple(float(c) for c in center)})
    return out, lt


def zoom(img, extent=(1.0, 1.0), rng=None, s=1.0, center=None):
    return zoom_xy(img, extent, rng, s, s, center)


def tile(img, extent=(1.0, 1.0), rng=None, size_in=0.5, ix=None, iy=None):
    """Physical-size tile (size_in inches square).  Grid index (ix, iy) of the non-overlapping
    floor(extent/size) grid, or a random position if None.  Density unchanged; per-tile count =
    density x size_in."""
    H, W = img.shape[:2]
    ppi = (W / extent[0], H / extent[1])
    tw, th = int(round(size_in * ppi[0])), int(round(size_in * ppi[1]))
    tw, th = min(tw, W), min(th, H)
    if ix is None or iy is None:
        r = _rng(rng)
        x0, y0 = int(r.integers(0, W - tw + 1)), int(r.integers(0, H - th + 1))
    else:
        nx, ny = max(1, W // tw), max(1, H // th)
        # centre the grid
        gx, gy = (W - nx * tw) // 2, (H - ny * th) // 2
        x0, y0 = gx + int(ix) * tw, gy + int(iy) * th
    out = np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw])
    A = np.array([[1, 0, -x0], [0, 1, -y0]], float)
    lt = _affine_lt(f"tile{size_in:g}", img, (tw, th), A, extent, ppi, params={"size_in": size_in, "x0": x0, "y0": y0})
    return out, lt


def tiles(img, extent=(1.0, 1.0), size_in=0.5):
    """All tiles of the centred non-overlapping grid: [(tile, lt), ...]."""
    H, W = img.shape[:2]
    ppi = (W / extent[0], H / extent[1])
    tw, th = int(round(size_in * ppi[0])), int(round(size_in * ppi[1]))
    nx, ny = max(1, W // max(1, tw)), max(1, H // max(1, th))
    return [tile(img, extent, None, size_in, ix, iy) for iy in range(ny) for ix in range(nx)]


def perspective(img, extent=(1.0, 1.0), rng=None, eps=0.02, jitter=None):
    """Simulated mis-rectification: the window's four corners are displaced by up to ``eps`` of the
    side (uniform per coordinate, or ``jitter`` = 4x2 array of fractions) and that quadrilateral is
    mapped onto the output frame.  The crossing-count change is computed exactly from the homography."""
    H, W = img.shape[:2]
    dst = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]], np.float64)
    if jitter is None:
        jitter = _rng(rng).uniform(-eps, eps, (4, 2))
    src = dst + np.asarray(jitter, float) * np.array([W, H], float)
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    out = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    Hi = np.linalg.inv(Hm)

    def inv(p, Hi=Hi):
        p = np.asarray(p, float)
        q = np.concatenate([p, np.ones((len(p), 1))], 1) @ Hi.T
        return q[:, :2] / q[:, 2:3]

    ppi = (W / extent[0], H / extent[1])
    lt = LabelTransform(f"persp{eps:g}", (W, H), (W, H), ppi, ppi, inv, status="exact",
                        params={"eps": eps, "jitter": np.asarray(jitter).tolist()})
    return out, lt


# =============================================================================================
# EXCLUDED transforms (negative controls for verify_augment.py; never use for training)
# =============================================================================================
def elastic(img, extent=(1.0, 1.0), rng=None, alpha=0.02, sigma=0.05):
    """Smooth random displacement field, amplitude alpha x side, correlation length sigma x side."""
    r = _rng(rng)
    H, W = img.shape[:2]
    s = max(1.0, sigma * W)
    dx = cv2.GaussianBlur(r.standard_normal((H, W)).astype(np.float32), (0, 0), s)
    dy = cv2.GaussianBlur(r.standard_normal((H, W)).astype(np.float32), (0, 0), s)
    dx *= alpha * W / (np.abs(dx).max() + 1e-9)
    dy *= alpha * H / (np.abs(dy).max() + 1e-9)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xx + dx, yy + dy
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    def inv(p, mx=mx, my=my):
        p = np.asarray(p, float)
        xi = np.clip(np.rint(p[:, 0]), 0, W - 1).astype(int)
        yi = np.clip(np.rint(p[:, 1]), 0, H - 1).astype(int)
        return np.stack([mx[yi, xi] + (p[:, 0] - xi), my[yi, xi] + (p[:, 1] - yi)], 1)

    ppi = (W / extent[0], H / extent[1])
    return out, LabelTransform(f"elastic{alpha:g}", (W, H), (W, H), ppi, ppi, inv, exact=False, status="excluded",
                               params={"alpha": alpha, "sigma": sigma})


def mixup(img, extent=(1.0, 1.0), rng=None, partner=None, lam=0.5):
    p = cv2.resize(partner, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    out = _u8(lam * _f(img) + (1 - lam) * _f(p))
    lt = identity_lt(img, extent, f"mixup{lam:g}", status="excluded", lam=lam)
    lt.exact, lt.weave_defined = False, False
    return out, lt


def cutmix(img, extent=(1.0, 1.0), rng=None, partner=None, frac=0.5):
    r = _rng(rng)
    H, W = img.shape[:2]
    p = cv2.resize(partner, (W, H), interpolation=cv2.INTER_AREA)
    bw, bh = int(round(W * math.sqrt(frac))), int(round(H * math.sqrt(frac)))
    x0, y0 = int(r.integers(0, W - bw + 1)), int(r.integers(0, H - bh + 1))
    out = img.copy()
    out[y0:y0 + bh, x0:x0 + bw] = p[y0:y0 + bh, x0:x0 + bw]
    lt = identity_lt(img, extent, f"cutmix{frac:g}", status="excluded", frac=frac)
    lt.exact, lt.weave_defined = False, False
    return out, lt


def strong_blur(img, extent=(1.0, 1.0), rng=None, sigma=4.0):
    out, lt = gaussian_blur(img, extent, rng, sigma)
    lt.name, lt.status, lt.exact = f"strongblur{sigma:g}", "excluded", False
    return out, lt


def label_agnostic_resized_crop(img, extent=(1.0, 1.0), rng=None, s=1.4):
    """The common mistake: zoom by s but keep claiming the crop is the original physical size.
    Returns the WRONG (identity) label transform on purpose; the correct one is in lt.params['true_lt']."""
    out, true_lt = zoom(img, extent, rng, s, center=(0.5, 0.5))
    lt = identity_lt(img, extent, f"naiveRRC{s:g}", status="excluded", s=s)
    lt.params["true_lt"] = true_lt
    lt.exact = False
    return out, lt


# =============================================================================================
# registry, composition and a training policy
# =============================================================================================
FAMILIES = {
    "a_flip": ["hflip", "vflip"],
    "b_rot90": ["rot90", "rot270", "rot180"],
    "c_rotate": ["rotate"],
    "d_photometric": ["brightness", "contrast", "gamma", "hue_shift", "saturation", "greyscale", "clahe",
                      "white_balance"],
    "e_sensor": ["gaussian_noise", "jpeg", "gaussian_blur", "motion_blur", "sharpen"],
    "f_illumination": ["glare", "shadow", "vignette"],
    "g_zoom": ["zoom", "zoom_xy"],
    "h_tile": ["tile"],
    "i_perspective": ["perspective"],
    "x_excluded": ["elastic", "mixup", "cutmix", "strong_blur", "label_agnostic_resized_crop"],
}
TRANSFORMS = {n: globals()[n] for fam in FAMILIES.values() for n in fam}


class Pipeline:
    """Apply a list of (fn, kwargs) in order, composing the label transforms."""

    def __init__(self, steps: Sequence):
        self.steps = list(steps)

    def __call__(self, img, extent=(1.0, 1.0), rng=None):
        r = _rng(rng)
        lt = identity_lt(img, extent)
        for fn, kw in self.steps:
            fn = TRANSFORMS[fn] if isinstance(fn, str) else fn
            ex = lt.extent_out
            img, lt_i = fn(img, extent=ex, rng=r, **kw)
            lt = compose(lt, lt_i)
        return img, lt


# Verified safe strength ranges (tier A of verify_augment.py, 2026-10-07: within ours' clean-run noise on
# 150 real crops AND 100 synthetic crops).  augment_whitelist.json holds the same 'policy' plus the
# 'policy_extended' (tier A + B: label map exact and unbiased on both instruments, e.g. rotate +-10 deg,
# zoom 0.5-1.25 with >= 6 px per yarn, perspective <= 3 %, hue/saturation/greyscale, JPEG q >= 20).
# 'p' = probability per sample (a choice, not a verification result).
DEFAULT_POLICY = {
    "hflip": {"p": 0.5},
    "vflip": {"p": 0.5},
    "rot90": {"p": 0.5},                                  # rot90 or rot270, h <-> v swapped
    "brightness": {"p": 0.5, "factor": [0.6, 1.0]},
    "contrast": {"p": 0.5, "factor": [0.6, 1.0]},
    "gaussian_noise": {"p": 0.3, "sigma": [0.0, 0.01]},
    "gaussian_blur_px": {"p": 0.2, "sigma": [0.0, 0.5]},
    "sharpen": {"p": 0.2, "amount": [0.0, 1.0]},
    "glare": {"p": 0.2, "n_blobs": [1, 2], "amp": [0.2, 0.4]},
    "shadow": {"p": 0.3, "depth": [0.5, 0.9]},
    "vignette": {"p": 0.3, "strength": [0.0, 0.6]},
}


def load_policy(path: str = WHITELIST_PATH, extended: bool = False) -> dict:
    """Training policy from the verification whitelist.

    extended=False: tier A only (within ours' clean-run noise on real-150 and synth-100) -- default.
    extended=True : tier A + tier B (label map exact by construction and unbiased on both instruments,
                    but ours' lattice search is not stable under that resampling; see verify_augment.py).
    Without the JSON, DEFAULT_POLICY is returned."""
    if not os.path.exists(path):
        return json.loads(json.dumps(DEFAULT_POLICY))
    wl = json.load(open(path))
    return json.loads(json.dumps(wl.get("policy_extended" if extended else "policy") or {}))


def random_augment(img, label=None, rng=None, extent=(1.0, 1.0), policy: Optional[dict] = None,
                   angles=(0.0, 0.0), pitch_px: Optional[float] = None):
    """Sample a random label-consistent augmentation from the whitelist policy.

    Returns (img_out, label_out, lt).  label_out = lt.crossing(label, angles) when a label is given
    (crossing counts per inch along the output axes -- the annotation-tool convention); use
    lt.density(label) for intrinsic-density labels.  ``pitch_px`` (yarn pitch in input px, e.g.
    side / max(h, v)) scales the blur so that sigma stays a fraction of the pitch."""
    r = _rng(rng)
    pol = policy or load_policy()
    steps = []

    def on(k):
        return k in pol and r.random() < pol[k].get("p", 0)

    def U(k, a):
        lo, hi = pol[k][a]
        return float(r.uniform(lo, hi))

    # geometry first (so photometric noise is not resampled)
    if on("zoom"):
        lo, hi = pol["zoom"]["s"]
        mpt = pol["zoom"].get("min_px_per_thread")
        if mpt and pitch_px:                 # keep >= mpt output px per yarn when zooming out
            lo = min(max(lo, mpt / pitch_px), max(hi, 1.0))
        steps.append(("zoom", {"s": float(math.exp(r.uniform(math.log(lo), math.log(max(hi, lo)))))}))
    if on("rotate"):
        steps.append(("rotate", {"deg": U("rotate", "deg"), "mode": pol["rotate"].get("mode", "inner")}))
    if on("perspective"):
        steps.append(("perspective", {"eps": U("perspective", "eps")}))
    if on("rot90"):
        steps.append(("rot90", {"k": int(r.choice([1, 3]))}))
    if on("hflip"):
        steps.append(("hflip", {}))
    if on("vflip"):
        steps.append(("vflip", {}))
    # illumination, colour
    if on("shadow"):
        steps.append(("shadow", {"depth": U("shadow", "depth")}))
    if on("glare"):
        steps.append(("glare", {"n_blobs": int(r.integers(pol["glare"]["n_blobs"][0], pol["glare"]["n_blobs"][1] + 1)),
                                "amp": U("glare", "amp")}))
    if on("vignette"):
        steps.append(("vignette", {"strength": U("vignette", "strength")}))
    for k, a in (("brightness", "factor"), ("contrast", "factor"), ("gamma", "g"), ("saturation", "factor"),
                 ("hue_shift", "deg"), ("clahe", "clip")):
        if on(k):
            steps.append((k, {a: U(k, a)}))
    if on("white_balance"):
        lo, hi = pol["white_balance"]["gain"]
        steps.append(("white_balance", {"gains": tuple(float(r.uniform(lo, hi)) for _ in range(3))}))
    if on("greyscale"):
        steps.append(("greyscale", {}))
    # sensor
    if on("gaussian_blur") and pitch_px:
        steps.append(("gaussian_blur", {"sigma": U("gaussian_blur", "sigma_pitch") * pitch_px}))
    if on("gaussian_blur_px"):
        steps.append(("gaussian_blur", {"sigma": U("gaussian_blur_px", "sigma")}))
    if on("motion_blur"):
        steps.append(("motion_blur", {"length": int(r.integers(pol["motion_blur"]["length"][0],
                                                                pol["motion_blur"]["length"][1] + 1))}))
    if on("sharpen"):
        steps.append(("sharpen", {"amount": U("sharpen", "amount")}))
    if on("gaussian_noise"):
        steps.append(("gaussian_noise", {"sigma": U("gaussian_noise", "sigma")}))
    if on("jpeg"):
        steps.append(("jpeg", {"q": int(r.integers(pol["jpeg"]["q"][0], pol["jpeg"]["q"][1] + 1))}))
    out, lt = Pipeline(steps)(img, extent, r)
    lab = lt.crossing(label, angles) if label is not None else None
    return out, lab, lt

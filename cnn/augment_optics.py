"""augment_optics.py -- pick-glass optics, phone-resolution and label-TRANSFORMING geometric
augmentations with exact labels for 1-inch pick-glass crops (WVAQ 2027, Job B).

Companion to augment.py (NOT a replacement; augment.py is untouched).  Same call signature:

    img_out, lt = T(img, *, extent=(1.0, 1.0), rng=None, **strength)

``img`` is an HxWx3 uint8 BGR crop covering ``extent`` inches; ``lt`` is an :class:`OLT` whose
``crossing(label, angles)`` / ``density(label, angles)`` map the input label (h_tpi, v_tpi[, weave])
to the label of the output image.  Conventions are those of augment.py (read its docstring):
h = vertical yarns per inch counted walking horizontally, v = horizontal yarns per inch counted
walking vertically; 'crossing' = per-inch count along an image-axis guide line (what the annotation
tool and ours' h_count / v_count measure); 'density' = intrinsic yarns per inch perpendicular to the
yarns.  Yarn-tilt angles use ours' convention (angle_x: k_h ~ (cos ax, sin ax); angle_y:
k_v ~ (sin ay, cos ay); image y down).

Label engine (same principle as augment.LabelTransform, re-implemented here so that this file does
not depend on a module another job is editing).  Every transform stores the exact inverse pixel map
``inv`` (output px -> input px, pixel centres at integers).  Straight uniformly spaced yarns are the
level sets of a phase phi(p) = k.p, so the number of yarns crossing ANY monotone curve equals
|phi(end) - phi(start)|: the crossing count of an output row / column follows exactly from the
preimages of its two end points -- also for the non-affine maps (radial distortion, bow).  The
per-inch label of the output is that count divided by the DECLARED physical extent of the output.

THREE FAMILIES
==============
(1) optics of a phone looking through a pick-glass magnifier          label effect
    vignette_cos4(strength)        cos^4 natural fall-off in linear light,   none (identity geometry)
                                   off-axis optical centre
    radial_distortion(k1, k2)      Brown radial model on the INVERSE map     exact, computed: see (D)
    chromatic_aberration(eps)      lateral CA: R magnified by (1+eps), B by  G channel exact; R / B
                                   (1-eps) about the optical centre         counts x 1/(1+-eps)
                                   (rainbow fringes, as on photo #77)        (|change| <= eps)
    dof_gradient(sigma_max)        defocus growing across the window         none (blur); observability
                                   (tilted phone; 'ramp' or 'vee' profile)   lost when sigma ~ 0.35 pitch
    led_hotspot(amp, r0)           specular LED lobe (1 + r^2/r0^2)^-p in    none; texture destroyed in
                                   linear light, then sensor clipping        the clipped core
(2) phone resolution
    phone_capture(ppt)             scene -> lens PSF -> pixel-aperture       none (scale map, same 1-inch
                                   integration at ppt px per yarn -> RGGB    window, fewer pixels)
                                   Bayer mosaic -> shot/read noise ->
                                   bilinear demosaic -> sRGB -> ISP sharpen
                                   -> JPEG
    area_downsample(ppt)           INTER_AREA only (ablation: pure          none
                                   resolution, no camera pipeline)
(3) label-TRANSFORMING geometry (output = a NEW 1-inch crop with a new label)
    stretch(sx, sy)                fabric stretched so 1 inch -> s inches,   h' = h / sx, v' = v / sy
                                   re-cropped to the 1-inch frame            (crossing; exact, any tilt)
    zoom(s) = stretch(s, s)                                                  h' = h / s, v' = v / s
                                                                             (crossing AND density)
    shear(gamma, axis='y')         skewed weft  y_out = y_in + g (x_in-cx)   v' = v,  h' = h (1 - g tan ax)
                                   (axis='x': skewed warp, symmetric)        per inch, mode 'inner'
    bow(depth, axis='y')           bowed weft: parabolic vertical shift,     h' = h, v' = v exactly
                                   depth x side at the centre column         per inch, mode 'inner'

DERIVATIONS (unit-tested in test_augment_optics.py on analytic gratings)
------------------------------------------------------------------------
(S) Stretch.  Output px p_o maps to input inch q = A p_o + b with A = diag(1/sx, 1/sy) (output inch ->
    input inch).  A yarn family with reciprocal vector k (cycles / inch) has phase k.q, so in the
    output frame k_o = A^T k = (k_x / sx, k_y / sy).  The crossing count along an output row is
    |k_o,x| x (row length) = (k_x / sx) x 1 inch -> h' = h / sx; likewise v' = v / sy.  This is exact
    for crossing counts at any yarn tilt; the intrinsic density |k_o| changes as
    sqrt((k_x/sx)^2 + (k_y/sy)^2) and the tilt as tan ax' = tan ax * sx / sy.  sx >= 1 only needs a
    window of 1/sx inch of the input (exact); sx < 1 needs fabric outside the crop and is filled by
    reflection (status 'approx': each mirror seam adds at most +-1 yarn per guide line).
(Z) Zoom s = stretch(s, s): k_o = k / s, so crossing AND density both scale by exactly 1/s and the
    yarn tilt is unchanged.
(H) Shear.  Weft skew (axis 'y'): y_out = y_in + g (x_in - cx), i.e. inverse J = d(in)/d(out) =
    [[1, 0], [-g, 1]] and k_o = J^T k = (k_x - g k_y, k_y).  Along a vertical guide line the count is
    k_o,y = k_y -> v' = v EXACTLY (any g, any tilt).  Along a horizontal guide line the h family has
    k = h (1, tan ax) -> h' = h (1 - g tan ax): unchanged for untilted warp, a second-order change
    otherwise (g = 0.1, ax = 5 deg -> 0.9 %).  The intrinsic weft density rises to
    v sqrt(1 + (tan ay - g)^2)/sqrt(1 + tan^2 ay) because the weft now runs at atan(tan ay - g):
    a skewed fabric is still counted correctly along the warp-direction guide line, which is the
    textile convention.  Mode 'inner' keeps the native px/inch and crops the valid band of height
    H - |g| W (declared extent (1, 1 - |g| W/H) inch); mode 'frame' resizes that band to the full
    frame and declares it 1 inch, so additionally v' = v (H_o / H) (an exact vertical stretch).
(B) Bow (axis 'y'): y_in = y_out + d - b(x), b(x) = depth H (1 - (2 (x - cx) / W)^2).  Columns: the
    displacement is constant along a column, so v' = v exactly.  Rows: both end points of every
    output row have b = 0, so their preimages differ by (W, 0) and h' = h exactly, at ANY tilt --
    a bowed weft is count-preserving along both guide lines.  The local weft angle varies by
    +-atan(4 depth) across the window (depth 0.02 -> +-4.6 deg), which is what stresses a global
    lattice estimator.
(D) Radial distortion.  Inverse map p_in = c + (p_out - c) g(r) / G with g(r) = 1 + k1 r^2 + k2 r^4,
    r = |p_out - c| / (half side).  k1 > 0: scene compressed towards the edges (barrel, as in an
    uncorrected wide phone lens); k1 < 0: pincushion (simple magnifier).  norm='inside' chooses G so
    that the whole output samples inside the input (no padding, exact).  The label is local-scale
    dependent: the crossing count of row y is |phi(p_in(W-.5, y)) - phi(p_in(-.5, y))|, and the label
    = mean over rows = the area-weighted mean of the local scale d x_in / d x_out (identical for
    untilted yarns, because integral_x dx_in/dx_out = x_in(end) - x_in(start)).  Both are reported
    (lt.params['area_mean_scale'], lt.row_spread()).  norm='mean' instead fixes the area-mean scale
    to 1 (label unchanged to first order; the corners need a thin reflected band -> 'approx').

Safe strength ranges are measured by augment_optics_verify.py (frozen classical counter vs the
predicted label change, against a no-op re-encode noise floor) and stored in
augment_optics_whitelist.json; :func:`sample` draws only from those ranges.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHITELIST_PATH = os.path.join(HERE, "augment_optics_whitelist.json")


# =============================================================================================
# helpers
# =============================================================================================
def _rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def srgb_to_lin(x):
    """x in [0, 1] (float32) -> linear light."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055).astype(np.float32)


def _f(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _grid(W, H):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx, yy


# =============================================================================================
# label engine
# =============================================================================================
@dataclass
class OLT:
    """Label bookkeeping of one (possibly composed) transform; API mirrors augment.LabelTransform."""

    name: str
    in_size: tuple                      # (W, H) px of the input
    out_size: tuple                     # (W, H) px of the output
    extent_in: tuple                    # physical (wx, wy) inches covered by the input
    extent_out: tuple                   # DECLARED physical extent of the output (inches)
    inv: Optional[Callable] = None      # (N, 2) output px -> input px; None = pure scale map
    status: str = "exact"               # 'exact' | 'approx'
    family: str = ""
    label_changing: bool = False        # True for family (3) (and distortion)
    diag_flip: bool = False
    weave_defined: bool = True
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.inv is None:            # same window, different pixel grid
            sx = self.in_size[0] / self.out_size[0]
            sy = self.in_size[1] / self.out_size[1]

            def inv(p, sx=sx, sy=sy):
                p = np.asarray(p, float)
                return np.stack([(p[:, 0] + 0.5) * sx - 0.5, (p[:, 1] + 0.5) * sy - 0.5], 1)

            self.inv = inv

    # ------------------------------------------------------------ geometry
    @property
    def ppi_in(self):
        return (self.in_size[0] / self.extent_in[0], self.in_size[1] / self.extent_in[1])

    @property
    def ppi_out(self):
        return (self.out_size[0] / self.extent_out[0], self.out_size[1] / self.extent_out[1])

    def _k_in(self, label, angles, kind):
        """Reciprocal vectors (cycles / INPUT px) of the h and v families."""
        h, v = float(label[0]), float(label[1])
        ax, ay = math.radians(angles[0]), math.radians(angles[1])
        if kind == "crossing":
            kh = np.array([h, h * math.tan(ax)])
            kv = np.array([v * math.tan(ay), v])
        elif kind == "density":
            kh = h * np.array([math.cos(ax), math.sin(ax)])
            kv = v * np.array([math.sin(ay), math.cos(ay)])
        else:
            raise ValueError(kind)
        s = np.array(self.ppi_in, float)
        return kh / s, kv / s

    def _row_col(self, k_px, n=33):
        """Per-row crossings per output inch (x) and per-column crossings per output inch (y)."""
        W, H = self.out_size
        ys = np.linspace(0, H - 1, n)
        xs = np.linspace(0, W - 1, n)
        r0 = self.inv(np.stack([np.full(n, -0.5), ys], 1))
        r1 = self.inv(np.stack([np.full(n, W - 0.5), ys], 1))
        c0 = self.inv(np.stack([xs, np.full(n, -0.5)], 1))
        c1 = self.inv(np.stack([xs, np.full(n, H - 0.5)], 1))
        cx = np.abs((r1 - r0) @ k_px) / self.extent_out[0]
        cy = np.abs((c1 - c0) @ k_px) / self.extent_out[1]
        return cx, cy

    def crossing(self, label, angles=(0.0, 0.0), kind="crossing"):
        """Per-inch guide-line crossing counts of the output (mean over rows / columns)."""
        kh, kv = self._k_in(label, angles, kind)
        hx, _ = self._row_col(kh)
        _, vy = self._row_col(kv)
        w = label[2] if len(label) > 2 else None
        return float(hx.mean()), float(vy.mean()), (w if self.weave_defined else None)

    __call__ = crossing

    def row_spread(self, angles=(0.0, 0.0)):
        """(min, max) over rows / columns of the per-line count relative to the mean (non-uniformity)."""
        kh, kv = self._k_in((1.0, 1.0), angles, "crossing")
        hx, _ = self._row_col(kh, n=65)
        _, vy = self._row_col(kv, n=65)
        return (float(hx.min() / hx.mean()), float(hx.max() / hx.mean()),
                float(vy.min() / vy.mean()), float(vy.max() / vy.mean()))

    def _jac(self, P, e=0.5):
        """d(input inch)/d(output inch) at output px P (N, 2) -> (N, 2, 2), columns = d/dx_o, d/dy_o."""
        s_in = np.array(self.ppi_in, float)
        q0 = self.inv(P) / s_in
        qx = self.inv(P + [e, 0.0]) / s_in
        qy = self.inv(P + [0.0, e]) / s_in
        dx = (qx - q0) / (e / self.ppi_out[0])
        dy = (qy - q0) / (e / self.ppi_out[1])
        return np.stack([dx, dy], 2)

    def density(self, label, angles=(0.0, 0.0), kind="density", n=15):
        """Area-weighted mean intrinsic per-inch density of the output (|J^T k| averaged on a grid)."""
        kh, kv = self._k_in(label, angles, kind)
        s = np.array(self.ppi_in, float)
        W, H = self.out_size
        gx, gy = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, n))
        J = self._jac(np.stack([gx.ravel(), gy.ravel()], 1))
        kh_o = np.einsum("nij,i->nj", J, kh * s)
        kv_o = np.einsum("nij,i->nj", J, kv * s)
        w = label[2] if len(label) > 2 else None
        return (float(np.linalg.norm(kh_o, axis=1).mean()), float(np.linalg.norm(kv_o, axis=1).mean()),
                (w if self.weave_defined else None))

    def area_mean_scale(self, n=64):
        """Area-weighted mean of d x_in / d x_out and d y_in / d y_out (input inch per output inch),
        midpoint rule over the full output window [-0.5, W-0.5] x [-0.5, H-0.5]."""
        W, H = self.out_size
        gx, gy = np.meshgrid((np.arange(n) + 0.5) / n * W - 0.5, (np.arange(n) + 0.5) / n * H - 0.5)
        J = self._jac(np.stack([gx.ravel(), gy.ravel()], 1))
        return float(J[:, 0, 0].mean()), float(J[:, 1, 1].mean())

    def weighted_scale(self, weight="hann2", n=64):
        """Window-weighted mean of d x_in / d x_out and d y_in / d y_out.  A spectral estimator with a
        2-D Hann window locates the peak of |FFT(w . I)|^2, i.e. (to first order) the w^2-weighted
        mean local frequency ('hann2'); 'hann' = w, 'uniform' = area mean (== mean guide-line count for
        untilted yarns).  For non-uniform maps (radial distortion) these differ: the label of such an
        image depends on the estimator's spatial weighting."""
        W, H = self.out_size
        u = (np.arange(n) + 0.5) / n
        gx, gy = np.meshgrid(u * W - 0.5, u * H - 0.5)
        J = self._jac(np.stack([gx.ravel(), gy.ravel()], 1))
        w1 = np.outer(np.sin(np.pi * u) ** 2, np.sin(np.pi * u) ** 2).ravel()   # 2-D Hann
        w = {"uniform": np.ones_like(w1), "hann": w1, "hann2": w1 ** 2}[weight]
        return float((J[:, 0, 0] * w).sum() / w.sum()), float((J[:, 1, 1] * w).sum() / w.sum())

    def counts(self, label, angles=(0.0, 0.0), kind="crossing"):
        h, v, w = self.crossing(label, angles, kind)
        return h * self.extent_out[0], v * self.extent_out[1], w

    def factors(self, angles=(0.0, 0.0)):
        hc, vc, _ = self.crossing((1.0, 1.0), angles)
        hd, vd, _ = self.density((1.0, 1.0), angles, kind="crossing")
        return dict(cross_x=hc, cross_y=vc, dens_x=hd, dens_y=vd, extent_x=self.extent_out[0],
                    extent_y=self.extent_out[1])

    def describe(self):
        f = self.factors()
        return (f"{self.name}: {self.in_size}px/{self.extent_in}in -> {self.out_size}px/"
                f"{tuple(round(e, 4) for e in self.extent_out)}in, crossing x({f['cross_x']:.4f}, "
                f"{f['cross_y']:.4f}), density x({f['dens_x']:.4f}, {f['dens_y']:.4f}), status={self.status}")


def identity_lt(img, extent=(1.0, 1.0), name="identity", family="", status="exact", **params):
    H, W = img.shape[:2]
    return OLT(name, (W, H), (W, H), tuple(extent), tuple(extent), _ident_inv, status=status,
               family=family, params=params)


def _ident_inv(p):
    return np.asarray(p, float)


def compose(lt1: OLT, lt2: OLT) -> OLT:
    """lt2 after lt1 (the declared extent of lt1's output is lt2's input extent)."""
    if tuple(lt1.out_size) != tuple(lt2.in_size):
        raise ValueError(f"size mismatch {lt1.out_size} -> {lt2.in_size}")
    # lt2 measures in units of lt1's declared output inch; the composed inverse is in input px
    i1, i2 = lt1.inv, lt2.inv
    st = "approx" if "approx" in (lt1.status, lt2.status) else "exact"
    return OLT(f"{lt1.name}|{lt2.name}", lt1.in_size, lt2.out_size, lt1.extent_in, lt2.extent_out,
               lambda p: i1(i2(p)), st, lt2.family or lt1.family, lt1.label_changing or lt2.label_changing,
               lt1.diag_flip ^ lt2.diag_flip, lt1.weave_defined and lt2.weave_defined,
               {"steps": [lt1.params, lt2.params]})


def _remap(img, inv, out_size, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT_101):
    W, H = out_size
    xx, yy = _grid(W, H)
    P = inv(np.stack([xx.ravel(), yy.ravel()], 1))
    mx = P[:, 0].reshape(H, W).astype(np.float32)
    my = P[:, 1].reshape(H, W).astype(np.float32)
    return cv2.remap(img, mx, my, interp, borderMode=border)


def _optical_centre(W, H, rng, center, jitter):
    if center is not None:
        return float(center[0]) * (W - 1), float(center[1]) * (H - 1)
    if rng is not None and jitter > 0:
        r = _rng(rng)
        return ((W - 1) * (0.5 + r.uniform(-jitter, jitter)), (H - 1) * (0.5 + r.uniform(-jitter, jitter)))
    return (W - 1) / 2.0, (H - 1) / 2.0


# =============================================================================================
# (1) optics
# =============================================================================================
def vignette_cos4(img, extent=(1.0, 1.0), rng=None, strength=0.3, center=None, jitter=0.15):
    """Natural (cos^4) vignetting in linear light: E(r) = (1 + (r/f)^2)^-2, f chosen so that the
    farthest window corner receives (1 - strength) of the on-axis irradiance.  The optical axis is
    offset from the window centre by up to ``jitter`` x side (random if rng is given)."""
    H, W = img.shape[:2]
    cx, cy = _optical_centre(W, H, rng, center, jitter)
    xx, yy = _grid(W, H)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    rc2 = max((cx - x) ** 2 + (cy - y) ** 2 for x in (0, W - 1) for y in (0, H - 1))
    s = float(np.clip(strength, 0.0, 0.95))
    a = ((1 - s) ** -0.5 - 1.0) / rc2 if s > 0 else 0.0     # (r/f)^2 = a r^2
    E = (1.0 + a * r2) ** -2
    out = _u8(lin_to_srgb(srgb_to_lin(_f(img)) * E[..., None]))
    return out, identity_lt(img, extent, f"vignette_cos4{strength:g}", "optics", strength=strength,
                            center=(cx / max(1, W - 1), cy / max(1, H - 1)))


def _radial_G(W, H, cx, cy, R, k1, k2, n=400):
    """Normaliser so that the preimage of the whole output boundary lies inside the input."""
    t = np.linspace(-0.5, W - 0.5, n)
    u = np.linspace(-0.5, H - 0.5, n)
    B = np.concatenate([np.stack([t, np.full(n, -0.5)], 1), np.stack([t, np.full(n, H - 0.5)], 1),
                        np.stack([np.full(n, -0.5), u], 1), np.stack([np.full(n, W - 0.5), u], 1)])
    d = B - [cx, cy]
    r = np.linalg.norm(d, axis=1) / R
    g = 1 + k1 * r ** 2 + k2 * r ** 4
    lim_x = np.where(d[:, 0] >= 0, (W - 0.5) - cx, cx + 0.5)
    lim_y = np.where(d[:, 1] >= 0, (H - 0.5) - cy, cy + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.maximum(np.abs(d[:, 0] * g) / lim_x, np.abs(d[:, 1] * g) / lim_y)
    return float(np.nanmax(need))


def radial_distortion(img, extent=(1.0, 1.0), rng=None, k1=0.0, k2=0.0, center=None, jitter=0.0,
                      norm="inside"):
    """Brown radial distortion on the inverse map (see module docstring (D)).  Label: computed
    exactly by the engine (mean over guide lines), with the area-mean scale stored in params."""
    H, W = img.shape[:2]
    cx, cy = _optical_centre(W, H, rng, center, jitter)
    R = (W + H) / 4.0
    if norm == "inside":
        G = _radial_G(W, H, cx, cy, R, k1, k2)
        status = "exact"
    elif norm == "mean":
        xx, yy = _grid(W, H)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        G = float((1 + k1 * r ** 2 + k2 * r ** 4).mean())
        status = "approx"
    else:
        raise ValueError(norm)

    def inv(p, cx=cx, cy=cy, R=R, k1=k1, k2=k2, G=G):
        p = np.asarray(p, float)
        d = p - [cx, cy]
        r2 = (d ** 2).sum(1, keepdims=True) / R ** 2
        return np.array([cx, cy]) + d * (1 + k1 * r2 + k2 * r2 ** 2) / G

    out = _remap(img, inv, (W, H))
    lt = OLT(f"distort_k1{k1:+g}_k2{k2:+g}", (W, H), (W, H), tuple(extent), tuple(extent), inv, status,
             "optics", True, params=dict(k1=k1, k2=k2, G=G, norm=norm, center=(cx, cy)))
    lt.params["area_mean_scale"] = lt.area_mean_scale()
    rr = np.array([0.0, 1.0, math.sqrt(2)])
    gg = 1 + k1 * rr ** 2 + k2 * rr ** 4
    lt.params["corner_scale_vs_centre"] = float(gg[2] / gg[0])      # local radial scale, corner / centre
    return out, lt


def chromatic_aberration(img, extent=(1.0, 1.0), rng=None, eps=0.005, center=None, jitter=0.15, sign=None):
    """Lateral CA: the red image is magnified by (1 + eps) and the blue by (1 - eps) relative to
    green about the optical centre (sign randomised if rng is given and sign is None).  Fringe width
    at the window corner ~ eps x (corner distance) px.  Label: the G channel is unchanged; R / B
    channels carry counts x 1/(1 +- eps); a luminance or PCA grey image is between them, so the
    count change is bounded by eps (stored as params['label_bound'])."""
    H, W = img.shape[:2]
    cx, cy = _optical_centre(W, H, rng, center, jitter)
    if sign is None:
        sign = (1 if _rng(rng).uniform() < 0.5 else -1) if rng is not None else 1
    e = float(eps) * sign
    out = img.copy()
    for ch, m in ((2, 1 + e), (0, 1 - e)):          # BGR: 2 = red, 0 = blue

        def inv(p, m=m):
            p = np.asarray(p, float)
            return np.array([cx, cy]) + (p - [cx, cy]) / m

        out[..., ch] = _remap(np.ascontiguousarray(img[..., ch]), inv, (W, H))
    corner = max(math.hypot(cx - x, cy - y) for x in (0, W - 1) for y in (0, H - 1))
    return out, identity_lt(img, extent, f"ca{eps:g}", "optics", status="approx", eps=e,
                            channel_factors={"R": 1 / (1 + e), "G": 1.0, "B": 1 / (1 - e)},
                            label_bound=abs(e), fringe_px_corner=abs(e) * corner * 2)


def _blur_stack(x, sigmas):
    return [x if s <= 0.05 else cv2.GaussianBlur(x, (0, 0), float(s)) for s in sigmas]


def dof_gradient(img, extent=(1.0, 1.0), rng=None, sigma_max=2.0, pitch_px=None, sigma_pitch=None,
                 profile="ramp", angle=None, n_levels=9):
    """Depth-of-field gradient of a tilted phone: the defocus sigma grows linearly across the window
    in direction ``angle`` (deg; random if None and rng given), from 0 to ``sigma_max`` px ('ramp',
    focus at one edge) or from 0 at the centre line to sigma_max at both edges ('vee').  If
    ``sigma_pitch`` and ``pitch_px`` are given, sigma_max = sigma_pitch x pitch_px.  Blur is applied
    in linear light by per-pixel interpolation in a blur stack.  Label unchanged."""
    H, W = img.shape[:2]
    if sigma_pitch is not None and pitch_px is not None:
        sigma_max = float(sigma_pitch) * float(pitch_px)
    if angle is None:
        angle = float(_rng(rng).uniform(0, 360)) if rng is not None else 0.0
    xx, yy = _grid(W, H)
    a = math.radians(angle)
    d = (xx - (W - 1) / 2) * math.cos(a) + (yy - (H - 1) / 2) * math.sin(a)
    t = (d - d.min()) / max(1e-6, d.max() - d.min())
    t = t if profile == "ramp" else np.abs(2 * t - 1)
    sig = t * float(sigma_max)
    lev = np.linspace(0.0, float(sigma_max), n_levels)
    x = srgb_to_lin(_f(img))
    st = _blur_stack(x, lev)
    pos = np.clip(sig / max(1e-9, lev[1] - lev[0]) if sigma_max > 0 else 0 * sig, 0, n_levels - 1 - 1e-6)
    i0 = np.floor(pos).astype(int)
    fr = (pos - i0)[..., None].astype(np.float32)
    out = np.zeros_like(x)
    for k in range(n_levels - 1):
        m = (i0 == k)[..., None]
        if m.any():
            out = np.where(m, st[k] * (1 - fr) + st[k + 1] * fr, out)
    return _u8(lin_to_srgb(out)), identity_lt(img, extent, f"dof_{profile}{sigma_max:.3g}", "optics",
                                              sigma_max_px=float(sigma_max), profile=profile, angle=angle,
                                              sigma_pitch=sigma_pitch)


def led_hotspot(img, extent=(1.0, 1.0), rng=None, amp=1.0, r0=0.08, falloff=1.5, center=None,
                tint=(1.04, 1.0, 0.95), veil=0.15):
    """Specular reflection of the phone's LED: an additive lobe amp (1 + (r/r0)^2)^-falloff in LINEAR
    light (r0 as a fraction of the side; falloff 1.5 = a point source reflected by a slightly rough
    surface), slightly blue-white (BGR ``tint``), plus a contrast-reducing veil (``veil`` x lobe
    multiplies the texture towards its mean), then sensor clipping at full scale.  ``amp`` is in
    units of full scale: >= ~1 produces a clipped core whose area grows with amp.  Label unchanged;
    the clipped fraction is stored in params['clipped_frac']."""
    H, W = img.shape[:2]
    r = _rng(rng) if rng is not None else None
    if center is None:
        center = (r.uniform(0.2, 0.8), r.uniform(0.2, 0.8)) if r is not None else (0.5, 0.5)
    cx, cy = center[0] * (W - 1), center[1] * (H - 1)
    xx, yy = _grid(W, H)
    rr2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r0 * (W + H) / 2) ** 2
    lobe = float(amp) * (1.0 + rr2) ** (-float(falloff))
    x = srgb_to_lin(_f(img))
    m = x.mean((0, 1), keepdims=True)
    v = np.clip(veil * lobe, 0, 0.9)[..., None]
    x = x * (1 - v) + m * v + lobe[..., None] * np.asarray(tint, np.float32)[None, None, :]
    clipped = float((x.max(-1) >= 1.0).mean())
    out = _u8(lin_to_srgb(x))
    return out, identity_lt(img, extent, f"led{amp:g}", "optics", amp=amp, r0=r0, falloff=falloff,
                            center=tuple(center), clipped_frac=clipped)


# =============================================================================================
# (2) phone resolution
# =============================================================================================
def bayer_mosaic(lin_bgr):
    """RGGB colour-filter array: R at (even row, even col), G at (even, odd) and (odd, even), B at
    (odd, odd).  Input: linear BGR float image at sensor resolution."""
    H, W = lin_bgr.shape[:2]
    m = np.empty((H, W), np.float32)
    m[0::2, 0::2] = lin_bgr[0::2, 0::2, 2]
    m[0::2, 1::2] = lin_bgr[0::2, 1::2, 1]
    m[1::2, 0::2] = lin_bgr[1::2, 0::2, 1]
    m[1::2, 1::2] = lin_bgr[1::2, 1::2, 0]
    return m


def _cfa_masks(H, W):
    R = np.zeros((H, W), np.float32)
    G = np.zeros((H, W), np.float32)
    B = np.zeros((H, W), np.float32)
    R[0::2, 0::2] = 1
    G[0::2, 1::2] = 1
    G[1::2, 0::2] = 1
    B[1::2, 1::2] = 1
    return R, G, B


def demosaic_bilinear(m):
    """Classic bilinear demosaic of an RGGB mosaic -> BGR float.  BORDER_REFLECT_101 keeps the CFA
    parity at the borders (pixel -k <-> +k)."""
    H, W = m.shape
    R, G, B = _cfa_masks(H, W)
    kG = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0
    kRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
    f = lambda x, k: cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101)  # noqa: E731
    return np.stack([f(m * B, kRB), f(m * G, kG), f(m * R, kRB)], -1)


_EA_CODE = None


def demosaic_ea(m):
    """OpenCV edge-aware demosaic (pattern code auto-detected once for the RGGB layout above)."""
    global _EA_CODE
    if _EA_CODE is None:
        test = np.zeros((8, 8, 3), np.float32)
        test[..., 2] = 1.0                      # pure red scene
        mm = _u8(bayer_mosaic(test))
        for name in ("COLOR_BayerRG2BGR_EA", "COLOR_BayerBG2BGR_EA", "COLOR_BayerGR2BGR_EA", "COLOR_BayerGB2BGR_EA"):
            o = cv2.cvtColor(mm, getattr(cv2, name))
            if o[2:6, 2:6, 2].mean() > 200 and o[2:6, 2:6, 0].mean() < 50:
                _EA_CODE = getattr(cv2, name)
                break
    return _f(cv2.cvtColor(_u8(m), _EA_CODE))


def phone_capture(img, extent=(1.0, 1.0), rng=None, ppt=4.0, count=None, side=None, lens_sigma=0.4,
                  noise=1.0, demosaic="bilinear", sharpen=0.5, q=90):
    """Simulate re-photographing the 1-inch window with a phone that resolves ``ppt`` pixels per yarn.

    side = round(ppt x count) px (count = per-crop count of the FINER axis, e.g. max(h, v)) unless
    ``side`` is given.  Pipeline (``img`` is taken as the scene radiance, sRGB-encoded):
    sRGB -> linear; lens PSF = Gaussian with sigma ``lens_sigma`` SENSOR pixels; pixel-aperture
    integration and sampling (INTER_AREA, 100 % fill factor) at side x side; RGGB Bayer mosaic;
    Poisson-Gaussian noise var = noise * (2e-4 x + 1e-6) (base-ISO phone); demosaic ('bilinear' or
    'ea'); linear -> sRGB; ISP unsharp mask (amount ``sharpen``, sigma 1 px); JPEG quality ``q``.
    The output covers the same window with fewer pixels: label unchanged (scale map)."""
    H, W = img.shape[:2]
    if side is None:
        if count is None or not np.isfinite(count) or count <= 0:
            raise ValueError("phone_capture needs count (per-crop yarns of the finer axis) or side")
        side = int(round(float(ppt) * float(count)))
    side = int(max(8, min(side, min(W, H))))
    side -= side % 2                                 # whole Bayer quads
    r = _rng(rng)
    x = srgb_to_lin(_f(img))
    scale = W / side
    if lens_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(lens_sigma) * scale)
    x = cv2.resize(x, (side, side), interpolation=cv2.INTER_AREA)
    m = bayer_mosaic(x)
    if noise > 0:
        var = noise * (2e-4 * np.clip(m, 0, 1) + 1e-6)
        m = m + r.normal(0.0, 1.0, m.shape).astype(np.float32) * np.sqrt(var).astype(np.float32)
    m = np.clip(m, 0, 1)
    y = demosaic_bilinear(m) if demosaic == "bilinear" else srgb_to_lin(demosaic_ea(lin_to_srgb(m)))
    s = lin_to_srgb(y)
    if sharpen > 0:
        s = s + sharpen * (s - cv2.GaussianBlur(s, (0, 0), 1.0))
    out = _jpeg(_u8(s), q)
    lt = OLT(f"phone_ppt{ppt:g}", (W, H), (side, side), tuple(extent), tuple(extent), None, "exact", "phone",
             params=dict(ppt=ppt, side=side, count=count, lens_sigma=lens_sigma, noise=noise,
                         demosaic=demosaic, sharpen=sharpen, q=q))
    return out, lt


def area_downsample(img, extent=(1.0, 1.0), rng=None, ppt=4.0, count=None, side=None):
    """Pure-resolution ablation of phone_capture: INTER_AREA to side = ppt x count, nothing else."""
    H, W = img.shape[:2]
    if side is None:
        side = int(round(float(ppt) * float(count)))
    side = int(max(8, min(side, min(W, H))))
    out = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return out, OLT(f"area_ppt{ppt:g}", (W, H), (side, side), tuple(extent), tuple(extent), None, "exact",
                    "phone", params=dict(ppt=ppt, side=side, count=count))


def upsample(img, extent=(1.0, 1.0), rng=None, factor=2.0):
    """Bicubic up-sampling of the whole window (label unchanged).  Used to test whether a counter's
    failure at low px/yarn is its own search band (e.g. ours' 0.42 cycles/px limit) or information."""
    H, W = img.shape[:2]
    Wo, Ho = int(round(W * factor)), int(round(H * factor))
    return (cv2.resize(img, (Wo, Ho), interpolation=cv2.INTER_CUBIC),
            OLT(f"up{factor:g}", (W, H), (Wo, Ho), tuple(extent), tuple(extent), None, "exact", "phone",
                params=dict(factor=factor)))


# =============================================================================================
# (3) label-transforming geometry
# =============================================================================================
def stretch(img, extent=(1.0, 1.0), rng=None, sx=1.0, sy=1.0, center=None):
    """The fabric is stretched so that 1 inch becomes (sx, sy) inches and the 1-inch frame is
    re-cropped: output = same pixel size and DECLARED extent as the input, label h' = h/sx, v' = v/sy
    (crossing, exact).  s >= 1 crops an input window of 1/s (exact); s < 1 needs a window larger
    than the crop, filled by reflection (status 'approx').  The integer window actually used is
    stored and the label follows it exactly (params sx_eff, sy_eff)."""
    H, W = img.shape[:2]
    r = _rng(rng) if (center is None and rng is not None) else None
    if center is None:
        center = (r.uniform(0, 1), r.uniform(0, 1)) if r is not None else (0.5, 0.5)
    ww = max(4, int(round(W / float(sx))))
    wh = max(4, int(round(H / float(sy))))
    x0 = int(round((W - ww) * float(center[0]))) if ww <= W else -int(round((ww - W) / 2))
    y0 = int(round((H - wh) * float(center[1]))) if wh <= H else -int(round((wh - H) / 2))
    padl, padt = max(0, -x0), max(0, -y0)
    padr, padb = max(0, x0 + ww - W), max(0, y0 + wh - H)
    src = img
    if padl or padt or padr or padb:
        if max(padl, padr) >= W or max(padt, padb) >= H:
            raise ValueError("stretch: s too small for single reflection")
        src = cv2.copyMakeBorder(img, padt, padb, padl, padr, cv2.BORDER_REFLECT_101)
    win = src[y0 + padt:y0 + padt + wh, x0 + padl:x0 + padl + ww]
    interp_x = cv2.INTER_CUBIC if ww <= W and wh <= H else cv2.INTER_AREA
    out = cv2.resize(win, (W, H), interpolation=interp_x)
    ax_, ay_ = ww / W, wh / H                         # input px per output px

    def inv(p, x0=x0, y0=y0, ax_=ax_, ay_=ay_):
        p = np.asarray(p, float)
        return np.stack([x0 + (p[:, 0] + 0.5) * ax_ - 0.5, y0 + (p[:, 1] + 0.5) * ay_ - 0.5], 1)

    status = "exact" if (ww <= W and wh <= H) else "approx"
    lt = OLT(f"stretch{sx:g}x{sy:g}", (W, H), (W, H), tuple(extent), tuple(extent), inv, status, "geometric",
             True, params=dict(sx=sx, sy=sy, sx_eff=W / ww, sy_eff=H / wh, window=(x0, y0, ww, wh)))
    return out, lt


def zoom(img, extent=(1.0, 1.0), rng=None, s=1.25, center=None):
    """Isotropic stretch: crossing AND density x 1/s exactly (tilt unchanged)."""
    out, lt = stretch(img, extent, rng, sx=s, sy=s, center=center)
    lt.name = f"zoom{s:g}"
    return out, lt


def shear(img, extent=(1.0, 1.0), rng=None, gamma=0.1, axis="y", mode="inner"):
    """Skewed weft (axis='y': y_out = y_in + gamma (x_in - cx)) or skewed warp (axis='x').
    mode 'inner': the valid band (height H - |gamma| W) at native px/inch, declared extent shrinks
    accordingly -> per-inch v' = v exactly and h' = h (1 - gamma tan ax) (module docstring (H)).
    mode 'frame': the band is resized to the full frame and declared 1 inch."""
    H, W = img.shape[:2]
    g = float(gamma)
    if axis == "x":                                   # transpose trick: shear the transposed image
        t, lt_t = shear(np.ascontiguousarray(img.transpose(1, 0, 2)), (extent[1], extent[0]), rng, gamma, "y", mode)
        out = np.ascontiguousarray(t.transpose(1, 0, 2))
        it = lt_t.inv

        def inv(p, it=it):
            p = np.asarray(p, float)
            return it(p[:, ::-1])[:, ::-1]

        lt = OLT(f"shearx{g:+g}_{mode}", (W, H), (out.shape[1], out.shape[0]), tuple(extent),
                 (lt_t.extent_out[1], lt_t.extent_out[0]), inv, lt_t.status, "geometric", True,
                 params=dict(lt_t.params, axis="x"))
        return out, lt
    cx = (W - 1) / 2.0
    Ho = int(math.floor(H - abs(g) * W))
    if Ho < 8:
        raise ValueError("shear too strong")
    d = abs(g) * W / 2.0 + (H - abs(g) * W - Ho) / 2.0

    def inv_in(p, g=g, cx=cx, d=d):
        p = np.asarray(p, float)
        return np.stack([p[:, 0], p[:, 1] - g * (p[:, 0] - cx) + d], 1)

    M = np.array([[1, 0, 0], [-g, 1, g * cx + d]], float)     # output -> input (WARP_INVERSE_MAP)
    band = cv2.warpAffine(img, M, (W, Ho), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT_101)
    if mode == "inner":
        ext = (extent[0], extent[1] * Ho / H)
        return band, OLT(f"sheary{g:+g}_inner", (W, H), (W, Ho), tuple(extent), ext, inv_in, "exact", "geometric",
                         True, params=dict(gamma=g, axis="y", mode=mode, Ho=Ho, tilt_deg=math.degrees(math.atan(g))))
    if mode == "frame":
        out = cv2.resize(band, (W, H), interpolation=cv2.INTER_CUBIC)
        fy = Ho / H

        def inv(p, fy=fy):
            p = np.asarray(p, float)
            return inv_in(np.stack([p[:, 0], (p[:, 1] + 0.5) * fy - 0.5], 1))

        return out, OLT(f"sheary{g:+g}_frame", (W, H), (W, H), tuple(extent), tuple(extent), inv, "exact",
                        "geometric", True, params=dict(gamma=g, axis="y", mode=mode, Ho=Ho,
                                                       tilt_deg=math.degrees(math.atan(g))))
    raise ValueError(mode)


def bow(img, extent=(1.0, 1.0), rng=None, depth=0.02, axis="y", sign=1):
    """Bowed weft (axis 'y'): vertical displacement b(x) = sign depth H (1 - (2 (x - cx)/W)^2), i.e.
    the weft sags by depth x side at the centre column and not at the selvedge-side edges.  Mode
    'inner' (valid band of height H(1 - depth), native px/inch): h' = h and v' = v per inch EXACTLY
    (module docstring (B)).  axis 'x': bowed warp."""
    H, W = img.shape[:2]
    if axis == "x":
        t, lt_t = bow(np.ascontiguousarray(img.transpose(1, 0, 2)), (extent[1], extent[0]), rng, depth, "y", sign)
        out = np.ascontiguousarray(t.transpose(1, 0, 2))
        it = lt_t.inv

        def inv(p, it=it):
            p = np.asarray(p, float)
            return it(p[:, ::-1])[:, ::-1]

        return out, OLT(f"bowx{depth:g}", (W, H), (out.shape[1], out.shape[0]), tuple(extent),
                        (lt_t.extent_out[1], lt_t.extent_out[0]), inv, "exact", "geometric", True,
                        params=dict(lt_t.params, axis="x"))
    cx = (W - 1) / 2.0
    D = float(depth) * H
    Ho = int(math.floor(H - D))
    d = D + (H - D - Ho) / 2.0 if sign > 0 else (H - D - Ho) / 2.0

    def inv(p, cx=cx, d=d, D=D, W=W, sign=sign):
        p = np.asarray(p, float)
        b = D * (1 - (2 * (p[:, 0] - cx) / W) ** 2)
        return np.stack([p[:, 0], p[:, 1] + d - (b if sign > 0 else -b)], 1)

    out = _remap(img, inv, (W, Ho))
    return out, OLT(f"bowy{depth:g}", (W, H), (W, Ho), tuple(extent), (extent[0], extent[1] * Ho / H), inv,
                    "exact", "geometric", True,
                    params=dict(depth=depth, axis="y", sign=sign, Ho=Ho,
                                edge_tilt_deg=math.degrees(math.atan(4 * depth * H / W))))


# =============================================================================================
# registry, whitelist-driven sampler
# =============================================================================================
TRANSFORMS = {
    "vignette_cos4": vignette_cos4, "radial_distortion": radial_distortion,
    "chromatic_aberration": chromatic_aberration, "dof_gradient": dof_gradient, "led_hotspot": led_hotspot,
    "phone_capture": phone_capture, "area_downsample": area_downsample, "upsample": upsample,
    "stretch": stretch, "zoom": zoom, "shear": shear, "bow": bow,
}

# Fallback policy (used only if augment_optics_whitelist.json is absent): conservative ranges.
DEFAULT_POLICY = {
    "vignette_cos4": {"p": 0.3, "strength": [0.0, 0.4]},
    "radial_distortion": {"p": 0.2, "k1": [-0.02, 0.02]},
    "chromatic_aberration": {"p": 0.2, "eps": [0.0, 0.005]},
    "dof_gradient": {"p": 0.2, "sigma_pitch": [0.0, 0.2]},
    "led_hotspot": {"p": 0.2, "amp": [0.2, 1.0]},
    "phone_capture": {"p": 0.3, "ppt": [4.0, 8.0]},
    "zoom": {"p": 0.3, "s": [1.0, 1.5]},
    "stretch": {"p": 0.3, "s": [1.0, 1.3]},
    "shear": {"p": 0.2, "gamma": [-0.1, 0.1]},
    "bow": {"p": 0.1, "depth": [0.0, 0.02]},
}


def load_policy(path=WHITELIST_PATH):
    if os.path.exists(path):
        try:
            return json.load(open(path))["policy"]
        except Exception:
            pass
    return dict(DEFAULT_POLICY)


def sample(img, rng=None, extent=(1.0, 1.0), count=None, policy=None, order=None):
    """Random whitelisted chain.  ``count`` = per-crop count of the finer axis (needed for the
    pitch-scaled DOF blur and the phone simulation; those steps are skipped without it).
    Returns (img, OLT) with the exact label transform of the chain."""
    r = _rng(rng)
    pol = policy or load_policy()
    order = order or ["zoom", "stretch", "shear", "bow", "radial_distortion", "chromatic_aberration",
                      "vignette_cos4", "led_hotspot", "dof_gradient", "phone_capture"]
    x, lt = img, identity_lt(img, extent)
    cnt = count
    for k in order:
        if k not in pol or r.uniform() >= pol[k].get("p", 0):
            continue
        P = pol[k]
        u = lambda key: float(r.uniform(*P[key]))  # noqa: E731
        if k == "zoom":
            y, l2 = zoom(x, lt.extent_out, r, s=u("s"))
        elif k == "stretch":
            s = u("s")
            y, l2 = stretch(x, lt.extent_out, r, sx=s, sy=1.0) if r.uniform() < 0.5 else \
                stretch(x, lt.extent_out, r, sx=1.0, sy=s)
        elif k == "shear":
            y, l2 = shear(x, lt.extent_out, r, gamma=u("gamma"), axis="y" if r.uniform() < 0.7 else "x", mode="frame")
        elif k == "bow":
            y, l2 = bow(x, lt.extent_out, r, depth=u("depth"), axis="y", sign=1 if r.uniform() < 0.5 else -1)
            y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
            l2 = compose(l2, OLT("toframe", tuple(l2.out_size), (x.shape[1], x.shape[0]), l2.extent_out, lt.extent_out))
        elif k == "radial_distortion":
            y, l2 = radial_distortion(x, lt.extent_out, r, k1=u("k1"))
        elif k == "chromatic_aberration":
            y, l2 = chromatic_aberration(x, lt.extent_out, r, eps=u("eps"))
        elif k == "vignette_cos4":
            y, l2 = vignette_cos4(x, lt.extent_out, r, strength=u("strength"))
        elif k == "led_hotspot":
            y, l2 = led_hotspot(x, lt.extent_out, r, amp=u("amp"))
        elif k == "dof_gradient":
            if cnt is None:
                continue
            y, l2 = dof_gradient(x, lt.extent_out, r, pitch_px=x.shape[1] / cnt, sigma_pitch=u("sigma_pitch"))
        elif k == "phone_capture":
            if cnt is None:
                continue
            y, l2 = phone_capture(x, lt.extent_out, r, ppt=u("ppt"), count=cnt)
        else:
            continue
        x, lt = y, compose(lt, l2)
        if count is not None:   # per-crop count of the finer axis after the chain (untilted estimate)
            f = lt.factors()
            cnt = count * max(f["cross_x"] * f["extent_x"], f["cross_y"] * f["extent_y"])
    return x, lt


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "annotation_tool", "data",
                                                          "crops", "1.jpg")
    im = cv2.imread(p)
    for name, (y, l) in {"zoom1.5": zoom(im, s=1.5), "stretchx1.3": stretch(im, sx=1.3),
                         "shear0.1": shear(im, gamma=0.1), "bow0.02": bow(im, depth=0.02),
                         "barrel0.05": radial_distortion(im, k1=0.05),
                         "phone3": phone_capture(im, ppt=3, count=55)}.items():
        print(name, y.shape, l.describe())

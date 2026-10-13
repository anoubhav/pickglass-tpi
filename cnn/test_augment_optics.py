"""Unit tests for augment_optics.py (no pytest needed:  python test_augment_optics.py).

Geometric tests render an analytic yarn grating with KNOWN reciprocal vector, apply the transform,
MEASURE the crossings along output rows / columns by Hilbert phase unwrapping (independent of the
label code), and compare with (a) the closed-form derivation in the module docstring and (b) the
label engine.  Photometric / optics / phone tests check that the measured frequency is unchanged.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment_optics as AO  # noqa: E402


def render(W, H, extent=(1.0, 1.0), h=None, v=None, ax=0.0, ay=0.0, kind="crossing", color=False):
    ppi = (W / extent[0], H / extent[1])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    X, Y = (xx + 0.5) / ppi[0], (yy + 0.5) / ppi[1]
    img = np.full((H, W), 0.5)
    a, b = math.radians(ax), math.radians(ay)
    if h is not None:
        k = (h, h * math.tan(a)) if kind == "crossing" else (h * math.cos(a), h * math.sin(a))
        img += 0.35 * np.cos(2 * np.pi * (k[0] * X + k[1] * Y) + 0.3)
    if v is not None:
        k = (v * math.tan(b), v) if kind == "crossing" else (v * math.sin(b), v * math.cos(b))
        img += 0.35 * np.cos(2 * np.pi * (k[0] * X + k[1] * Y) + 1.1)
    g = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out = np.repeat(g[..., None], 3, 2)
    if color:   # coloured yarns (different contrast per channel) for the CA / Bayer tests
        out = np.clip(out.astype(np.float32) * np.array([0.6, 0.9, 1.1])[None, None], 0, 255).astype(np.uint8)
    return out


def measure(img, axis="x", trim=0.1, n=21, ch=1):
    """Mean crossings per output PIXEL along rows (axis x) or columns (y), full-row phase difference
    extrapolated to the full length."""
    g = img[..., ch].astype(np.float64)
    if axis == "y":
        g = g.T
    H, W = g.shape
    a, b = int(trim * W), int((1 - trim) * W)
    rates = []
    for r in np.linspace(int(trim * H), int((1 - trim) * H) - 1, n).astype(int):
        p = g[r] - g[r].mean()
        ph = np.unwrap(np.angle(hilbert(p)))
        rates.append(abs(ph[b - 1] - ph[a]) / (2 * np.pi * (b - 1 - a)))
    return float(np.mean(rates))


def per_inch(img, lt, axis):
    W, H = img.shape[1], img.shape[0]
    return measure(img, axis) * (W / lt.extent_out[0] if axis == "x" else H / lt.extent_out[1])


def close(a, b, rel, msg):
    assert abs(a - b) <= rel * abs(b) + 1e-9, f"{msg}: {a:.4f} vs {b:.4f} (rel tol {rel})"


N = 720
H0, V0 = 41.0, 53.0
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ------------------------------------------------------------------------- (3) label-transforming
@test
def test_stretch_closed_form_tilted():
    for sx, sy, ax, ay in ((1.3, 1.0, 0.0, 0.0), (1.6, 1.15, 6.0, -4.0), (1.0, 1.45, -8.0, 3.0)):
        img_h = render(N, N, h=H0, ax=ax)
        img_v = render(N, N, v=V0, ay=ay)
        oh, lt = AO.stretch(img_h, sx=sx, sy=sy)
        ov, _ = AO.stretch(img_v, sx=sx, sy=sy)
        h_eng, v_eng, _ = lt.crossing((H0, V0), (ax, ay))
        sxe, sye = lt.params["sx_eff"], lt.params["sy_eff"]
        close(h_eng, H0 / sxe, 1e-6, "stretch engine h vs h/sx")
        close(v_eng, V0 / sye, 1e-6, "stretch engine v vs v/sy")
        close(per_inch(oh, lt, "x"), H0 / sxe, 0.01, f"stretch measured h sx={sx}")
        close(per_inch(ov, lt, "y"), V0 / sye, 0.01, f"stretch measured v sy={sy}")
        # tilt rule tan ax' = tan ax * sx / sy (via density engine consistency)
        hd, _, _ = lt.density((H0, V0), (ax, ay), kind="crossing")
        kx, ky = H0 / sxe, H0 * math.tan(math.radians(ax)) / sye
        close(hd, math.hypot(kx, ky), 1e-3, "stretch density")


@test
def test_stretch_below_one_reflect():
    img = render(N, N, h=H0)
    o, lt = AO.stretch(img, sx=0.85, sy=1.0)
    assert lt.status == "approx"
    close(lt.crossing((H0, V0))[0], H0 / lt.params["sx_eff"], 1e-6, "stretch<1 engine")
    # mirror seam: whole-row count within +-1 yarn per seam of the ideal (2 seams)
    meas = per_inch(o, lt, "x")
    assert abs(meas - H0 / lt.params["sx_eff"]) <= 2.0 + 0.02 * H0, (meas, H0 / lt.params["sx_eff"])


@test
def test_zoom_density_and_crossing():
    for s, ax in ((1.25, 0.0), (2.0, 7.0)):
        img = render(N, N, h=H0, ax=ax, kind="density")
        o, lt = AO.zoom(img, s=s)
        se = lt.params["sx_eff"]
        hd, _, _ = lt.density((H0, V0), (ax, 0.0), kind="density")
        close(hd, H0 / se, 1e-3, "zoom density")
        hc, _, _ = lt.crossing((H0, V0), (ax, 0.0), kind="density")
        close(hc, H0 * math.cos(math.radians(ax)) / se, 1e-6, "zoom crossing")
        close(per_inch(o, lt, "x"), hc, 0.01, f"zoom measured s={s}")


@test
def test_shear_weft_skew():
    for g, ax, ay in ((0.1, 0.0, 0.0), (0.18, 5.0, -3.0), (-0.12, -6.0, 4.0)):
        oh, lt = AO.shear(render(N, N, h=H0, ax=ax), gamma=g, axis="y", mode="inner")
        ov, _ = AO.shear(render(N, N, v=V0, ay=ay), gamma=g, axis="y", mode="inner")
        h_eng, v_eng, _ = lt.crossing((H0, V0), (ax, ay))
        close(v_eng, V0, 1e-6, "shear v' = v")
        close(h_eng, H0 * (1 - g * math.tan(math.radians(ax))), 1e-6, "shear h' = h(1 - g tan ax)")
        close(per_inch(ov, lt, "y"), V0, 0.01, f"shear measured v g={g}")
        close(per_inch(oh, lt, "x"), h_eng, 0.01, f"shear measured h g={g}")
    # frame mode: additionally v' = v * Ho/H
    o, lt = AO.shear(render(N, N, v=V0), gamma=0.1, mode="frame")
    close(lt.crossing((H0, V0))[1], V0 * lt.params["Ho"] / N, 1e-6, "shear frame engine")
    close(per_inch(o, lt, "y"), V0 * lt.params["Ho"] / N, 0.01, "shear frame measured")


@test
def test_shear_warp_skew_axis_x():
    g, ax = 0.12, 4.0
    o, lt = AO.shear(render(N, N, h=H0, ax=ax), gamma=g, axis="x", mode="inner")
    h_eng, v_eng, _ = lt.crossing((H0, V0), (ax, -3.0))
    close(h_eng, H0, 1e-6, "x-shear h' = h")
    close(per_inch(o, lt, "x"), H0, 0.01, "x-shear measured h")


@test
def test_bow_count_preserving():
    for depth, ax, ay in ((0.02, 0.0, 0.0), (0.04, 6.0, -5.0)):
        oh, lt = AO.bow(render(N, N, h=H0, ax=ax), depth=depth)
        ov, _ = AO.bow(render(N, N, v=V0, ay=ay), depth=depth)
        h_eng, v_eng, _ = lt.crossing((H0, V0), (ax, ay))
        close(h_eng, H0, 1e-6, "bow h' = h")
        close(v_eng, V0, 1e-6, "bow v' = v")
        # measured along FULL rows (end points at the undisplaced edges) -> exact
        close(measure(oh, "x", trim=0.0) * N, H0, 0.012, "bow measured h")
        close(per_inch(ov, lt, "y"), V0, 0.01, "bow measured v")


@test
def test_radial_distortion_label():
    for k1, k2 in ((0.05, 0.0), (-0.05, 0.0), (0.1, -0.02)):
        oh, lt = AO.radial_distortion(render(N, N, h=H0), k1=k1, k2=k2)
        h_eng = lt.crossing((H0, V0))[0]
        amx, amy = lt.params["area_mean_scale"]
        close(h_eng, H0 * amx, 2e-3, "distortion: mean row count == area-mean scale (untilted)")
        # whole-row measurement (end points) averaged over rows == engine
        close(measure(oh, "x", trim=0.0, n=41) * N, h_eng, 0.012, f"distortion measured k1={k1}")
        assert lt.status == "exact"


@test
def test_compose():
    img = render(N, N, h=H0, v=V0)
    a, l1 = AO.zoom(img, s=1.25)
    b, l2 = AO.stretch(a, sx=1.2, sy=1.0)
    lt = AO.compose(l1, l2)
    close(lt.crossing((H0, V0))[0], H0 / (l1.params["sx_eff"] * l2.params["sx_eff"]), 1e-6, "compose h")
    close(lt.crossing((H0, V0))[1], V0 / l1.params["sy_eff"], 1e-6, "compose v")


# ------------------------------------------------------------------------- (1) optics, (2) phone
@test
def test_photometric_optics_label_free():
    img = render(N, N, h=H0)
    for fn, kw in ((AO.vignette_cos4, dict(strength=0.5)), (AO.led_hotspot, dict(amp=0.25, r0=0.08, center=(0.2, 0.2))),
                   (AO.dof_gradient, dict(sigma_max=2.5))):
        o, lt = fn(img, rng=1, **kw)
        assert lt.crossing((H0, V0))[:2] == (H0, V0) or np.allclose(lt.crossing((H0, V0))[:2], (H0, V0))
        close(per_inch(o, lt, "x"), H0, 0.01, f"{fn.__name__} measured")


@test
def test_led_clipping_grows_with_amp():
    img = render(N, N, h=H0)
    c = [AO.led_hotspot(img, amp=a, r0=0.08)[1].params["clipped_frac"] for a in (0.5, 1.0, 2.0, 4.0)]
    assert c == sorted(c) and c[-1] > c[0], c


@test
def test_chromatic_aberration_channels():
    img = render(N, N, h=H0, color=True)
    o, lt = AO.chromatic_aberration(img, eps=0.01, center=(0.5, 0.5), sign=1)
    close(measure(o, "x", ch=1) * N, H0, 0.01, "CA green unchanged")
    close(measure(o, "x", ch=2) * N, H0 / 1.01, 0.01, "CA red x 1/(1+eps)")
    close(measure(o, "x", ch=0) * N, H0 / 0.99, 0.012, "CA blue x 1/(1-eps)")


@test
def test_bayer_demosaic_roundtrip():
    rng = np.random.default_rng(0)
    flat = np.ones((16, 16, 3), np.float32) * np.array([0.2, 0.5, 0.8], np.float32)
    d = AO.demosaic_bilinear(AO.bayer_mosaic(flat))
    assert np.allclose(d, flat, atol=1e-5), "bilinear demosaic must reproduce a flat colour exactly"
    x = rng.uniform(0, 1, (32, 32, 3)).astype(np.float32)
    m = AO.bayer_mosaic(x)
    assert np.isclose(m[0, 0], x[0, 0, 2]) and np.isclose(m[1, 1], x[1, 1, 0]) and np.isclose(m[0, 1], x[0, 1, 1])
    e = AO.demosaic_ea(AO.bayer_mosaic(flat))
    assert e[4:12, 4:12, 2].mean() > e[4:12, 4:12, 0].mean(), "EA pattern code detection"


@test
def test_phone_capture_label_unchanged():
    img = render(1440, 1440, h=H0, v=V0, ax=3.0, color=True)
    for ppt in (3.0, 4.0, 6.0):
        o, lt = AO.phone_capture(img, rng=0, ppt=ppt, count=V0)
        assert o.shape[0] == lt.params["side"]
        assert np.allclose(lt.crossing((H0, V0), (3.0, 0.0))[:2], (H0, V0))
        side = o.shape[0]
        close(side / V0, ppt, 0.05, "px per yarn")
        ih, _ = AO.phone_capture(render(1440, 1440, h=H0, color=True), rng=0, ppt=ppt, count=V0)
        close(measure(ih, "x") * side, H0, 0.015, f"phone measured h ppt={ppt}")


@test
def test_area_downsample_and_upsample():
    img = render(1440, 1440, h=H0)
    o, lt = AO.area_downsample(img, ppt=4, count=H0)
    close(measure(o, "x") * o.shape[1], H0, 0.01, "area downsample")
    u, lu = AO.upsample(o, factor=2)
    close(measure(u, "x") * u.shape[1], H0, 0.01, "upsample")


@test
def test_sample_chain_runs():
    img = render(N, N, h=H0, v=V0)
    for s in range(6):
        o, lt = AO.sample(img, rng=s, count=V0)
        assert o.dtype == np.uint8 and o.ndim == 3
        h, v, _ = lt.crossing((H0, V0))
        assert 0.3 * H0 < h < 2 * H0 and 0.3 * V0 < v < 2 * V0


if __name__ == "__main__":
    cv2.setNumThreads(2)
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)

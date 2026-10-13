"""Unit tests for augment.py label transforms.

Run:  python test_augment.py          (no pytest needed; pytest-compatible names)

Every geometric test renders an analytic yarn grating with KNOWN reciprocal vector, applies the
transform to the image, MEASURES the yarn crossings along the output rows / columns (Hilbert phase
unwrapping, independent of the label code) and compares with the label transform's prediction.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A  # noqa: E402


# ------------------------------------------------------------------------------------- helpers
def render(W, H, extent=(1.0, 1.0), h=None, v=None, ax=0.0, ay=0.0, kind="crossing"):
    """Grating image (uint8 BGR).  h/v: crossing counts per inch (kind='crossing') or densities,
    tilt angles in ours' convention.  Only the families that are not None are drawn."""
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
    return np.repeat(g[..., None], 3, 2)


def measure(img, axis="x", trim=0.12):
    """Mean yarn crossings per output pixel along rows (axis='x') or columns ('y')."""
    g = img[..., 1].astype(np.float64)
    if axis == "y":
        g = g.T
    H, W = g.shape
    a, b = int(trim * W), int((1 - trim) * W)
    rates = []
    for r in np.linspace(int(trim * H), int((1 - trim) * H) - 1, 15).astype(int):
        p = g[r] - g[r].mean()
        ph = np.unwrap(np.angle(hilbert(p)))
        rates.append(abs(ph[b - 1] - ph[a]) / (2 * np.pi * (b - 1 - a)))
    return float(np.mean(rates))


def close(a, b, rel=0.01, msg=""):
    assert abs(a - b) <= rel * abs(b) + 1e-9, f"{msg}: {a} vs {b} (rel tol {rel})"


# ------------------------------------------------------------------------------------- tests
def test_rot90_swaps_labels():
    for k, swap in ((1, True), (2, False), (3, True)):
        img = np.zeros((300, 400, 3), np.uint8)              # non-square: 1 x 0.75 inch at 400 ppi
        out, lt = A.rot90(img, extent=(1.0, 0.75), k=k)
        assert lt.swaps_axes == swap, (k, lt.swaps_axes)
        h, v, w = lt((60.0, 40.0, "twill_2_1"))
        if swap:
            close(h, 40.0, 1e-6, f"rot{90 * k} h")
            close(v, 60.0, 1e-6, f"rot{90 * k} v")
            assert out.shape[:2] == (400, 300)
            close(lt.extent_out[0], 0.75, 1e-9), close(lt.extent_out[1], 1.0, 1e-9)
        else:
            close(h, 60.0, 1e-6), close(v, 40.0, 1e-6)
        assert w == "twill_2_1"
        assert lt.diag_flip == bool(k % 2)
        hc, vc, _ = lt.counts((60.0, 40.0))
        close(hc, (40.0 if swap else 60.0) * lt.extent_out[0], 1e-6)
        close(vc, (60.0 if swap else 40.0) * lt.extent_out[1], 1e-6)


def test_rot90_on_rendered_grating():
    img = render(512, 512, h=48.0)                          # vertical yarns only
    out, lt = A.rot90(img, k=1)
    h, v, _ = lt.crossing((48.0, 1e-6))
    close(measure(out, "y") * 512, 48.0, 0.01, "rot90 moves the h family to v")
    close(v, 48.0, 1e-6)
    assert measure(out, "x") * 512 < 1.0


def test_flips_identity_and_diag():
    img = render(400, 400, h=50.0, v=37.0, ax=4.0, ay=-3.0)
    for fn in (A.hflip, A.vflip):
        out, lt = fn(img)
        assert not lt.swaps_axes and lt.diag_flip
        (h, v, _), (ax, ay) = lt.crossing((50.0, 37.0, "plain"), (4.0, -3.0), return_angles=True)
        close(h, 50.0, 1e-6), close(v, 37.0, 1e-6)
        assert abs(abs(ax) - 4.0) < 1e-6 and abs(abs(ay) - 3.0) < 1e-6
        single = render(400, 400, h=50.0, ax=4.0)          # one family: clean phase measurement
        out1, _ = fn(single)
        close(measure(out1, "x") * 400, 50.0, 0.01, fn.__name__)


def test_zoom_crop_scaling():
    """s = 2: density unchanged, per-crop count halves, ppi doubles, per-pixel frequency halves."""
    img = render(600, 600, h=60.0)
    f0 = measure(img, "x")
    for s in (1.25, 1.6, 2.0):
        out, lt = A.zoom(img, s=s, center=(0.3, 0.7))
        h, v, _ = lt((60.0, 45.0))
        close(h, 60.0, 1e-6, "density unchanged")
        close(v, 45.0, 1e-6)
        hc, vc, _ = lt.counts((60.0, 45.0))
        close(hc, 60.0 / s, 1e-6, "per-crop count")
        close(lt.ppi_out[0], 600 * s, 1e-9, "ppi")
        close(lt.extent_out[0], 1 / s, 1e-9, "extent")
        assert out.shape == img.shape
        close(measure(out, "x"), f0 / s, 0.01, f"per-pixel frequency at s={s}")
        close(measure(out, "x") * out.shape[1], hc, 0.01, "measured per-crop count")


def test_zoom_out_downsample():
    img = render(800, 800, v=70.0)
    f0 = measure(img, "y")
    for s in (0.5, 0.7):
        out, lt = A.zoom(img, s=s)
        assert out.shape[0] == round(800 * s)
        hc, vc, _ = lt.counts((50.0, 70.0))
        close(vc, 70.0, 1e-6, "per-crop count unchanged (whole inch)")
        h, v, _ = lt((50.0, 70.0))
        close(v, 70.0, 1e-6, "density unchanged")
        close(measure(out, "y"), f0 / s, 0.01, f"per-pixel frequency x 1/s at s={s}")


def test_zoom_anisotropic():
    img = render(500, 500, h=40.0, v=55.0)
    out, lt = A.zoom_xy(img, sx=1.25, sy=0.8)
    hc, vc, _ = lt.counts((40.0, 55.0))
    close(hc, 40.0 / 1.25, 1e-6)
    close(vc, 55.0, 1e-6)
    h, v, _ = lt((40.0, 55.0))
    close(h, 40.0, 1e-6), close(v, 55.0, 1e-6)
    close(measure(out, "x") * out.shape[1], hc, 0.015, "aniso x")
    close(measure(out, "y") * out.shape[0], vc, 0.015, "aniso y")


def test_tiles_physical_size():
    img = render(800, 800, h=64.0, v=48.0)
    for size, n in ((0.5, 4), (0.25, 16)):
        ts = A.tiles(img, size_in=size)
        assert len(ts) == n
        for t, lt in ts:
            hc, vc, _ = lt.counts((64.0, 48.0))
            close(hc, 64.0 * size, 1e-6), close(vc, 48.0 * size, 1e-6)
            h, v, _ = lt((64.0, 48.0))
            close(h, 64.0, 1e-6)
        t, lt = ts[0]
        close(measure(t, "x", trim=0.15) * t.shape[1], 64.0 * size, 0.02, f"tile {size}")


def test_rotation_cos_factor_and_measured():
    """Crossing factor cos(ax - deg)/cos(ax) (h) and cos(ay + deg)/cos(ay) (v); density unchanged."""
    for ax, ay, deg in ((0.0, 0.0, 10.0), (6.0, -4.0, 12.0), (-8.0, 5.0, -15.0), (3.0, 3.0, 5.0)):
        lt = A.rotate(np.zeros((500, 500, 3), np.uint8), deg=deg)[1]
        (h, v, _), (ax2, ay2) = lt.crossing((50.0, 40.0), (ax, ay), return_angles=True)
        r = math.radians
        close(h, 50.0 * math.cos(r(ax - deg)) / math.cos(r(ax)), 1e-6, "h cos factor")
        close(v, 40.0 * math.cos(r(ay + deg)) / math.cos(r(ay)), 1e-6, "v cos factor")
        assert abs(ax2 - (ax - deg)) < 1e-6 and abs(ay2 - (ay + deg)) < 1e-6
        hd, vd, _ = lt.density((50.0, 40.0), (ax, ay), kind="crossing")
        close(hd, 50.0 / math.cos(r(ax)), 1e-6, "density unchanged")
        close(vd, 40.0 / math.cos(r(ay)), 1e-6)
    # measured on rendered single-family gratings
    for ax, deg in ((5.0, 12.0), (-3.0, -15.0), (0.0, 7.0)):
        img = render(700, 700, h=45.0, ax=ax)
        out, lt = A.rotate(img, deg=deg, mode="inner")
        hc, _, _ = lt.counts((45.0, 30.0), (ax, 0.0))
        close(measure(out, "x") * out.shape[1], hc, 0.01, f"rotate {deg} h (ax={ax})")
        img = render(700, 700, v=52.0, ay=ax)
        out, lt = A.rotate(img, deg=deg, mode="inner")
        _, vc, _ = lt.counts((30.0, 52.0), (0.0, ax))
        close(measure(out, "y") * out.shape[0], vc, 0.01, f"rotate {deg} v (ay={ax})")
    # inner crop keeps px/inch, extent shrinks by 1/(|cos|+|sin|)
    lt = A.rotate(np.zeros((600, 600, 3), np.uint8), deg=15)[1]
    close(lt.ppi_out[0], 600.0, 1e-9)
    close(lt.extent_out[0], math.floor(600 / (math.cos(r(15)) + math.sin(r(15)))) / 600, 1e-9)


def test_perspective_exact_factor():
    for seed in range(4):
        img = render(600, 600, h=55.0)
        out, lt = A.perspective(img, rng=seed, eps=0.03)
        hc, _, _ = lt.counts((55.0, 40.0))
        close(measure(out, "x", trim=0.02) * out.shape[1], hc, 0.01, f"persp h seed {seed}")
        img = render(600, 600, v=47.0)
        out, lt = A.perspective(img, rng=seed, eps=0.03)
        _, vc, _ = lt.counts((40.0, 47.0))
        close(measure(out, "y", trim=0.02) * out.shape[0], vc, 0.01, f"persp v seed {seed}")
        assert abs(hc / 55.0 - 1) < 0.07


def test_composition_and_pipeline():
    img = render(600, 600, h=60.0, v=30.0)
    pipe = A.Pipeline([("zoom", {"s": 2.0, "center": (0.5, 0.5)}), ("rot90", {"k": 1})])
    out, lt = pipe(img)
    hc, vc, _ = lt.counts((60.0, 30.0))
    close(hc, 15.0, 1e-6, "swap then half")
    close(vc, 30.0, 1e-6)
    h, v, _ = lt((60.0, 30.0))
    close(h, 30.0, 1e-6), close(v, 60.0, 1e-6)
    close(lt.extent_out[0], 0.5, 1e-9)
    close(measure(out, "y", trim=0.05) * out.shape[0], 30.0, 0.015, "measured after pipeline")
    # a tile of a rotated crop
    out, lt = A.Pipeline([("rotate", {"deg": 8.0}), ("tile", {"size_in": 0.5, "ix": 0, "iy": 0})])(img)
    hc, _, _ = lt.counts((60.0, 30.0))
    close(hc, 0.5 * 60.0 * math.cos(math.radians(8.0)), 1e-6)


def test_photometric_identity():
    img = render(256, 256, h=40.0, v=33.0)
    for fn, kw in ((A.brightness, {"factor": 1.4}), (A.contrast, {"factor": 0.6}), (A.gamma, {"g": 1.7}),
                   (A.hue_shift, {"deg": 40}), (A.saturation, {"factor": 0.3}), (A.greyscale, {}),
                   (A.clahe, {"clip": 3.0}), (A.white_balance, {"gains": (1.2, 1.0, 0.8)}),
                   (A.gaussian_noise, {"sigma": 0.02}), (A.jpeg, {"q": 30}), (A.gaussian_blur, {"sigma": 1.0}),
                   (A.motion_blur, {"length": 3}), (A.sharpen, {"amount": 1.0}), (A.glare, {"n_blobs": 2}),
                   (A.shadow, {"depth": 0.3}), (A.vignette, {"strength": 0.4})):
        out, lt = fn(img, rng=0, **kw)
        assert out.shape == img.shape and out.dtype == np.uint8, fn.__name__
        h, v, w = lt((40.0, 33.0, "plain"))
        assert (h, v, w) == (40.0, 33.0, "plain"), fn.__name__
        assert lt.factors()["cross_x"] == 1.0


def test_excluded_flags():
    img = render(256, 256, h=40.0)
    other = render(256, 256, h=70.0)
    for out, lt in (A.elastic(img, rng=0), A.mixup(img, partner=other), A.cutmix(img, rng=0, partner=other),
                    A.strong_blur(img, sigma=5), A.label_agnostic_resized_crop(img, s=1.4)):
        assert lt.status == "excluded"
    _, lt = A.mixup(img, partner=other)
    assert lt((40.0, 40.0, "plain"))[2] is None
    _, lt = A.label_agnostic_resized_crop(img, s=1.4)
    close(lt.counts((40.0, 40.0))[0], 40.0, 1e-9)                       # what it (wrongly) claims
    close(lt.params["true_lt"].counts((40.0, 40.0))[0], 40.0 / 1.4, 1e-6)  # what is true


def test_random_augment_labels_match_image():
    img = render(640, 640, h=50.0)
    pol = {"zoom": {"p": 1.0, "s": [0.7, 1.6]}, "rotate": {"p": 1.0, "deg": [-10.0, 10.0], "mode": "inner"},
           "perspective": {"p": 1.0, "eps": [0.0, 0.03]}, "rot90": {"p": 1.0}, "hflip": {"p": 1.0},
           "vflip": {"p": 1.0}}
    for seed in range(6):
        out, lab, lt = A.random_augment(img, (50.0, 1e-3, "plain"), rng=seed, policy=pol)
        hc, vc, _ = lt.counts((50.0, 1e-3))
        axis = "y" if lt.swaps_axes else "x"
        n = out.shape[0] if axis == "y" else out.shape[1]
        meas = measure(out, axis, trim=0.05) * n
        close(meas, vc if lt.swaps_axes else hc, 0.02, f"random_augment seed {seed}: {lt.name}")


def test_policies_load_and_run():
    """Both whitelist policies load and produce label-consistent samples (photometric steps included)."""
    img = render(512, 512, h=40.0, v=30.0)
    for ext in (False, True):
        pol = A.load_policy(extended=ext)
        assert "hflip" in pol and "rot90" in pol
        for seed in range(4):
            out, lab, lt = A.random_augment(img, (40.0, 30.0, "plain"), rng=seed, policy=pol, pitch_px=512 / 40)
            assert out.dtype == np.uint8 and out.ndim == 3
            assert lt.status in ("exact", "approx")
            h, v, w = lab
            assert w == "plain" and h > 0 and v > 0
    if os.path.exists(A.WHITELIST_PATH):
        assert "rotate" not in A.load_policy() and "rotate" in A.load_policy(extended=True)


if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    fails = 0
    for k, f in tests:
        try:
            f()
            print(f"PASS {k}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {k}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)

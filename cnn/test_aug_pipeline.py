"""Tests for aug_pipeline.py: the tile targets of AugTileDataset must equal the yarn crossings
MEASURED on the produced tile (Hilbert phase, independent of the label code), for every geometric
family (px/inch + anisotropic jitter, perspective, rotate, flips, rot90 with swap), and the
test-time views of predict_crop_tta must map back exactly.

    python test_aug_pipeline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aug_pipeline as AP  # noqa: E402
from test_augment import measure, render  # noqa: E402

GEOM_ONLY = ("photometric", "compression_blur", "illumination")


def _check(kind, h, v, ax, ay, n=24, tol=0.02, policy="whitelist_ext", seed=0):
    errs = []
    for fam in ("h", "v"):
        img = render(1024, 1024, (1.0, 1.0), h=h if fam == "h" else None, v=v if fam == "v" else None,
                     ax=ax, ay=ay, kind=kind)
        it = AP.AugItem(img, None, h, v, 0, 1.0, kind, (ax, ay))
        ds = AP.AugTileDataset([it], tiles_per_image=1, policy=policy, drop=GEOM_ONLY, seed=seed)
        # force every geometric transform on (p = 1) so each sample exercises the full chain
        for k in ("perspective", "rotate", "hflip", "vflip", "rot90"):
            if k in ds.pol:
                ds.pol[k]["p"] = 1.0 if k not in ("rot90",) else 0.5
        r = np.random.default_rng(seed)
        for _ in range(n):
            t, th, tv, info = ds.sample(it, r)
            # the drawn family lands on axis x unless rot90 swapped it
            on_x = (fam == "h") != info["swap"]
            meas = measure(t, "x" if on_x else "y") * AP.TILE
            want = th if on_x else tv
            errs.append(abs(meas / want - 1))
            assert abs(meas / want - 1) < tol, (kind, fam, info, meas, want)
    return float(np.max(errs))


def test_crossing_labels_tilted():
    e = _check("crossing", 60.0, 45.0, 4.0, -3.0)
    print(f"  crossing labels, tilted grating: max rel err {e:.4f}")


def test_density_labels():
    # density labels: crossings measured on the tile = density x cos(tilt of the output yarns);
    # at zero input tilt and |rotation| <= 5 deg the difference is <= 0.4 %, so 2 % tolerance holds
    e = _check("density", 70.0, 50.0, 0.0, 0.0)
    print(f"  density labels, untilted grating: max rel err {e:.4f}")


def test_zoom_drop_fixes_scale_and_centre_tile():
    img = render(1024, 1024, h=60.0, v=None)
    it = AP.AugItem(img, None, 60.0, 45.0, 0, 1.0, "crossing", (0.0, 0.0))
    ds = AP.AugTileDataset([it], 1, "whitelist_ext", drop=("zoom", "tiling", "geometry") + GEOM_ONLY)
    r = np.random.default_rng(1)
    for _ in range(5):
        t, th, tv, info = ds.sample(it, r)
        assert info["Px"] == info["Py"] == 768 and info["x0"] == (768 - 224) // 2, info
        assert abs(th - 60 * 224 / 768) < 1e-9 and abs(tv - 45 * 224 / 768) < 1e-9


def test_photometric_keeps_labels():
    img = render(800, 800, h=50.0, v=None)
    it = AP.AugItem(img, None, 50.0, 40.0, 0, 1.0, "crossing", (0.0, 0.0))
    ds = AP.AugTileDataset([it], 1, "whitelist_ext", drop=("geometry", "zoom", "tiling"))
    r = np.random.default_rng(2)
    for _ in range(10):
        t, th, tv, info = ds.sample(it, r)
        assert abs(th - 50 * 224 / 768) < 1e-9 and abs(tv - 40 * 224 / 768) < 1e-9


def _check_optics(kind, h, v, ax, ay, keys, n=16, tol=0.02, seed=0, drop_extra=()):
    """'whitelist_optics': only the augment_optics steps in ``keys`` are on (p = 1), all other
    families off; the tile target must equal the crossings measured on the tile."""
    errs = []
    for fam in ("h", "v"):
        img = render(1024, 1024, (1.0, 1.0), h=h if fam == "h" else None, v=v if fam == "v" else None,
                     ax=ax, ay=ay, kind=kind)
        it = AP.AugItem(img, None, h, v, 0, 1.0, kind, (ax, ay))
        ds = AP.AugTileDataset([it], tiles_per_image=1, policy="whitelist_optics",
                               drop=GEOM_ONLY + tuple(drop_extra), seed=seed)
        for k in list(ds.pol):                       # augment.py geometry off (tested above)
            if k in ("perspective", "rotate", "hflip", "vflip", "rot90"):
                ds.pol[k]["p"] = 0.0
        ds.opol = {k: dict(v_, p=1.0) for k, v_ in ds.opol.items() if k in keys}
        assert ds.opol, keys
        r = np.random.default_rng(seed)
        for _ in range(n):
            t, th, tv, info = ds.sample(it, r)
            on_x = fam == "h"
            meas = measure(t, "x" if on_x else "y") * AP.TILE
            want = th if on_x else tv
            errs.append(abs(meas / want - 1))
            assert abs(meas / want - 1) < tol, (kind, fam, keys, info, meas, want)
    return float(np.max(errs))


def test_optics_fabric_geometry_exact():
    # stretch / shear / bow change (or keep) the label exactly: tilted crossing grating + density grating
    for keys in (("stretch",), ("shear",), ("bow",), ("stretch", "shear", "bow")):
        e1 = _check_optics("crossing", 60.0, 45.0, 3.0, -2.0, keys)
        e2 = _check_optics("density", 70.0, 50.0, 0.0, 0.0, keys)
        print(f"  optics {'+'.join(keys)}: max rel err crossing {e1:.4f}, density {e2:.4f}")


def test_optics_radial_and_phone():
    # radial distortion: label = crop area mean; a tile sees the local scale (|k1| <= 0.01 -> <= ~1.5 %)
    e = _check_optics("crossing", 60.0, 45.0, 2.0, -1.0, ("radial_distortion",), tol=0.03)
    print(f"  optics radial_distortion: max rel err {e:.4f} (tile-local vs crop-mean label)")
    # phone capture + lens CA + vignette + LED hotspot: identity geometry, label unchanged
    e = _check_optics("crossing", 50.0, 40.0, 0.0, 0.0, ("phone_capture", "chromatic_aberration",
                                                          "vignette_cos4"), tol=0.03)
    print(f"  optics phone_capture + CA + vignette: max rel err {e:.4f}")


def test_optics_policy_only_in_whitelist_optics():
    assert AP.get_optics_policy("whitelist_ext") == {} and AP.get_optics_policy("whitelist") == {}
    op = AP.get_optics_policy("whitelist_optics")
    assert "zoom" not in op and "phone_capture" in op and "radial_distortion" in op, op
    ds = AP.AugTileDataset([], 1, "whitelist_optics", drop=("optics", "phone"))
    assert set(ds.opol) == {"stretch", "shear", "bow"}, ds.opol
    ds = AP.AugTileDataset([], 1, "whitelist_optics", drop=("geometry",))
    assert not {"stretch", "shear", "bow"} & set(ds.opol), ds.opol


def test_tta_rot90_swap_is_exact():
    """A model that reads the label from the pixels (oracle via measure) must give the same per-inch
    answer through every TTA view; here we check the bookkeeping with a stub model that returns the
    measured log tile counts."""
    import math
    import torch

    class Stub(torch.nn.Module):
        def forward(self, x):
            out = []
            for t in x:
                g = (t[1].numpy() * 0.224 + 0.456)          # undo ImageNet norm (green channel)
                im = np.repeat((np.clip(g, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
                hx = max(measure(im, "x"), 1e-3) * 224
                vy = max(measure(im, "y"), 1e-3) * 224
                out.append([math.log(hx), math.log(vy)])
            return torch.tensor(out), torch.zeros(len(x), 4)

    # h family only (v family absent -> its measurement is noise; only h is checked)
    crop = render(1024, 1024, h=55.0, v=None)
    o = AP.predict_crop_tta(Stub(), crop, "cpu", Ps=(768,), degs=(0.0,), stride=272)
    assert abs(o["h"] / 55.0 - 1) < 0.02, o
    crop = render(1024, 1024, h=None, v=48.0)
    o = AP.predict_crop_tta(Stub(), crop, "cpu", Ps=(768,), degs=(0.0,), stride=272)
    assert abs(o["v"] / 48.0 - 1) < 0.02, o
    print(f"  TTA bookkeeping ok")


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"{n}/{n} passed")

"""aug_pipeline.py -- count-preserving augmentation for the tile CNN (training data + test-time views).

Connects the verified label-aware augmentation library (augment.py, augment_whitelist.json) to the
tile-regression formulation of cnn_common.py.  Every training sample is

    1-inch crop  --(zoom: resample to Px x Py px/inch, label-consistent scale jitter)-->
    --(geometry, whole crop: perspective <= 3 %, rotate +-5 deg 'inner'; label via LabelTransform)-->
    --(illumination, whole crop: shadow / glare / vignette)-->
    --(tiling: one 224-px physical tile at a random interior position)-->
    --(geometry, tile: hflip / vflip / rot90 with h <-> v swap)-->
    --(photometric, then compression/blur on the tile)--> tensor,  target = log(yarns crossing the tile)

Tile target = per-inch label of the transformed crop x 224 / px-per-inch of the transformed crop.
The per-inch label is mapped exactly by augment.LabelTransform:
  * synthetic crops   (kind='density')  : lt.density(label)   (generator labels are intrinsic density)
  * pseudo / human    (kind='crossing') : lt.crossing(label, angles) with the crop's yarn tilt
                                          (ours' angle_x, angle_y) -- guide-line crossing counts.

Families (the units of the drop-one ablation, ``drop=``):
  geometry          hflip, vflip, rot90 (tile level); rotate, perspective (crop level, tier B);
                    with 'whitelist_optics' also stretch, shear, bow (augment_optics, crop level)
  photometric       brightness, contrast, gamma, saturation, hue, greyscale, white balance (tile)
  compression_blur  Gaussian noise, JPEG, Gaussian blur (px and fraction of the yarn pitch), motion
                    blur, unsharp mask (tile)
  illumination      shadow, glare, vignette (crop level, so they look like whole-photo effects)
  zoom              the label-consistent px/inch jitter P ~ U(576, 960) plus the anisotropic x/y
                    jitter (zoom_xy); dropped -> fixed P = 768, isotropic
  tiling            random physical-tile position; dropped -> the centre tile only
  optics            (policy 'whitelist_optics' only) pick-glass optics of augment_optics.py:
                    radial_distortion, chromatic_aberration, dof_gradient, vignette_cos4, led_hotspot
  phone             (policy 'whitelist_optics' only) augment_optics.phone_capture: re-capture of the
                    1-inch window at 3.5-10 px per yarn (lens PSF, pixel integration, RGGB mosaic,
                    shot/read noise, demosaic, ISP sharpen, JPEG), then resampled back to Px x Py

Policies: 'whitelist'     = augment_whitelist.json 'policy' (tier A: within ours' clean-run noise)
          'whitelist_ext' = 'policy_extended' (tier A + B) with rotate limited to +-5 deg (the
                            verification job's recommended start) and its 'zoom' entry mapped onto
                            the px/inch jitter above (the tile CNN already zooms with exact labels).
          'whitelist_optics' = 'whitelist_ext' + the LABEL-EXACT tier of augment_optics_whitelist.json
                            ('policy': optics, phone capture, stretch / shear / bow; its 'zoom'
                            entry is dropped because the px/inch jitter already zooms with exact
                            labels).  The optics chain (augment_optics.sample) runs on the Px x Py
                            crop right after the zoom resample; its exact label map (OLT: crossing
                            for pseudo / human counts with the crop's yarn tilt, density for synthetic
                            crops) gives the new per-inch label of the whole crop.  Uniform maps
                            (stretch, shear, bow, phone, CA) are exact on every tile; radial
                            distortion (|k1| <= 0.01) is exact as the crop area mean and within ~1.5 %
                            locally (test_aug_pipeline.py).  The yarn tilt passed on to the tier-B
                            rotate / perspective maps is not updated for stretch / shear (second order).
          'existing'      = cnn_common.TileDataset + photometric_aug (the original recipe; see
                            make_dataset).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

import augment as A
import augment_optics as AO
import cnn_common as C

TILE = C.TILE
BORDER = C.BORDER
FAMILIES = ("geometry", "photometric", "compression_blur", "illumination", "zoom", "tiling", "optics", "phone")
FAMILY_OF = {
    "hflip": "geometry", "vflip": "geometry", "rot90": "geometry", "rotate": "geometry", "perspective": "geometry",
    "brightness": "photometric", "contrast": "photometric", "gamma": "photometric", "saturation": "photometric",
    "hue_shift": "photometric", "greyscale": "photometric", "white_balance": "photometric", "clahe": "photometric",
    "gaussian_noise": "compression_blur", "jpeg": "compression_blur", "gaussian_blur": "compression_blur",
    "gaussian_blur_px": "compression_blur", "motion_blur": "compression_blur", "sharpen": "compression_blur",
    "glare": "illumination", "shadow": "illumination", "vignette": "illumination",
    "zoom": "zoom",
    # augment_optics (policy 'whitelist_optics')
    "stretch": "geometry", "shear": "geometry", "bow": "geometry",
    "radial_distortion": "optics", "chromatic_aberration": "optics", "dof_gradient": "optics",
    "vignette_cos4": "optics", "led_hotspot": "optics",
    "phone_capture": "phone",
}
POLICIES = ("existing", "whitelist", "whitelist_ext", "whitelist_optics")
OPTICS_KEYS = ("stretch", "shear", "bow", "radial_distortion", "chromatic_aberration", "vignette_cos4",
               "led_hotspot", "dof_gradient", "phone_capture")


def get_policy(name: str) -> dict:
    if name == "whitelist":
        pol = A.load_policy(extended=False)
    elif name in ("whitelist_ext", "whitelist_optics"):
        pol = A.load_policy(extended=True)
        if "rotate" in pol:
            pol["rotate"]["deg"] = [-5.0, 5.0]
        pol.pop("zoom", None)          # realised as the px/inch jitter of the tile formulation
    else:
        raise ValueError(f"unknown policy {name!r} (expected one of {POLICIES})")
    return pol


def get_optics_policy(name: str) -> dict:
    """augment_optics label-exact tier ('policy' of augment_optics_whitelist.json) for
    'whitelist_optics'; {} for every other policy."""
    if name != "whitelist_optics":
        return {}
    pol = AO.load_policy()
    return {k: v for k, v in pol.items() if k in OPTICS_KEYS}


@dataclass
class AugItem:
    img: Optional[np.ndarray]          # BGR square 1-inch crop (or None -> load from path)
    path: Optional[str]
    h: float                           # per-inch label along x (vertical yarns per inch)
    v: float                           # per-inch label along y
    weave: int = -100
    w: float = 1.0
    kind: str = "density"              # 'density' (synthetic) | 'crossing' (pseudo / human counts)
    angles: tuple = (0.0, 0.0)         # yarn tilt (ours' angle_x, angle_y, degrees) for 'crossing'
    src: str = "synth"


def to_aug_items(items: Sequence[C.Item], kind="density", src="synth") -> list:
    return [AugItem(it.img, it.path, it.h, it.v, it.weave, it.w, kind, (0.0, 0.0), src) for it in items]


class AugTileDataset(torch.utils.data.Dataset):
    """Random 224-px tiles with whitelist augmentation and exact label bookkeeping (see module doc).
    An entry of ``items`` may be a list of AugItems (the files of one scene): each sample then draws
    one of them uniformly, so every scene has the same weight whatever its number of copies."""

    def __init__(self, items: Sequence[AugItem], tiles_per_image: int = 4, policy: str = "whitelist_ext",
                 drop: Sequence[str] = (), p_range=C.P_TRAIN, aniso: float = 0.5, p_fixed: int = 768,
                 seed: int = 0, max_side: int = 1024):
        self.items = list(items)
        self.tpi = tiles_per_image
        self.policy_name = policy
        self.pol = get_policy(policy)
        bad = set(drop) - set(FAMILIES)
        if bad:
            raise ValueError(f"unknown families {bad}")
        self.fam = set(FAMILIES) - set(drop)
        self.opol = {k: v for k, v in get_optics_policy(policy).items() if FAMILY_OF[k] in self.fam}
        self.p_range, self.aniso, self.p_fixed = p_range, aniso, p_fixed
        self.seed, self.max_side = seed, max_side

    def __len__(self):
        return len(self.items) * self.tpi

    def _load(self, it: AugItem):
        if it.img is not None:
            return it.img
        im = cv2.imread(it.path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(it.path)
        return C.cap_side(im, self.max_side)

    # ------------------------------------------------------------------ policy helpers
    def _on(self, r, k):
        return k in self.pol and FAMILY_OF.get(k) in self.fam and r.random() < self.pol[k].get("p", 0.0)

    def _U(self, r, k, a):
        lo, hi = self.pol[k][a]
        return float(r.uniform(lo, hi))

    def sample(self, it: AugItem, r: np.random.Generator, center_tile: Optional[bool] = None):
        """-> (tile uint8 BGR, tile_h, tile_v, info dict).  Exposed for tests."""
        img = self._load(it)
        # ---- zoom family: label-consistent px/inch jitter (isotropic P, anisotropic ratio)
        if "zoom" in self.fam:
            P = float(r.uniform(*self.p_range))
            ratio = math.exp(r.uniform(-self.aniso / 2, self.aniso / 2)) if self.aniso > 0 else 1.0
        else:
            P, ratio = float(self.p_fixed), 1.0
        Px, Py = int(round(P * math.sqrt(ratio))), int(round(P / math.sqrt(ratio)))
        interp = cv2.INTER_AREA if img.shape[0] > max(Px, Py) else cv2.INTER_CUBIC
        im = cv2.resize(img, (Px, Py), interpolation=interp)
        # ---- optics / phone / fabric geometry (augment_optics, label-exact tier) on the whole crop
        ochain = []
        if self.opol:
            im, olt = AO.sample(im, rng=r, extent=(1.0, 1.0), count=max(it.h, it.v), policy=self.opol)
            if olt.name != "identity":
                if it.kind == "density":
                    h1, v1, _ = olt.density((it.h, it.v), (0.0, 0.0), kind="density")
                else:
                    h1, v1, _ = olt.crossing((it.h, it.v), tuple(it.angles), kind="crossing")
                # per-crop counts over the declared output extent -> the image is again '1 inch'
                it = AugItem(None, it.path, h1 * olt.extent_out[0], v1 * olt.extent_out[1], it.weave, it.w,
                             it.kind, it.angles, it.src)
                ochain = olt.name.split("|")[1:]
            if im.shape[:2] != (Py, Px):          # phone capture: fewer px for the same window
                im = cv2.resize(im, (Px, Py), interpolation=cv2.INTER_CUBIC)
        # ---- geometry on the whole crop (exact label maps)
        steps = []
        if self._on(r, "perspective"):
            steps.append(("perspective", {"eps": self._U(r, "perspective", "eps")}))
        if self._on(r, "rotate"):
            steps.append(("rotate", {"deg": self._U(r, "rotate", "deg"), "mode": self.pol["rotate"].get("mode", "inner")}))
        lt = None
        if steps:
            im, lt = A.Pipeline(steps)(im, (1.0, 1.0), r)
            ppi = lt.ppi_out
        else:
            ppi = (float(Px), float(Py))
        # ---- illumination on the whole crop
        if self._on(r, "shadow"):
            im, _ = A.shadow(im, rng=r, depth=self._U(r, "shadow", "depth"))
        if self._on(r, "glare"):
            nb = self.pol["glare"]["n_blobs"]
            im, _ = A.glare(im, rng=r, n_blobs=int(r.integers(nb[0], nb[1] + 1)), amp=self._U(r, "glare", "amp"))
        if self._on(r, "vignette"):
            im, _ = A.vignette(im, rng=r, strength=self._U(r, "vignette", "strength"))
        # ---- tiling
        H, W = im.shape[:2]
        bx, by = int(BORDER * W), int(BORDER * H)
        hx, hy = W - bx - TILE, H - by - TILE
        if hx < bx or hy < by:
            raise ValueError(f"crop {W}x{H} too small for a {TILE}-px tile")
        centre = ("tiling" not in self.fam) if center_tile is None else center_tile
        if centre:
            x0, y0 = (W - TILE) // 2, (H - TILE) // 2
        else:
            x0, y0 = int(r.integers(bx, hx + 1)), int(r.integers(by, hy + 1))
        t = np.ascontiguousarray(im[y0:y0 + TILE, x0:x0 + TILE])
        if lt is None:
            hh, vv = it.h, it.v
        else:
            # LOCAL label of this tile: perspective makes the density vary across the crop, so the
            # label map is evaluated on the tile itself (crop composed onto the geometric chain)
            crop_lt = A._affine_lt("tilecrop", im, (TILE, TILE), np.array([[1, 0, -x0], [0, 1, -y0]], float),
                                   lt.extent_out)
            lt_t = A.compose(lt, crop_lt)
            if it.kind == "density":
                hh, vv, _ = lt_t.density((it.h, it.v), (0.0, 0.0), kind="density")
            else:
                hh, vv, _ = lt_t.crossing((it.h, it.v), tuple(it.angles), kind="crossing")
        th, tv = hh * TILE / ppi[0], vv * TILE / ppi[1]
        pitch_px = min(ppi) / max(hh, vv)
        # ---- tile-level geometry
        if self._on(r, "hflip"):
            t = t[:, ::-1]
        if self._on(r, "vflip"):
            t = t[::-1]
        swap = False
        if self._on(r, "rot90"):
            t = np.rot90(t, int(r.choice([1, 3])))
            th, tv = tv, th
            swap = True
        t = np.ascontiguousarray(t)
        # ---- photometric
        for k, a in (("brightness", "factor"), ("contrast", "factor"), ("gamma", "g"), ("saturation", "factor"),
                     ("hue_shift", "deg"), ("clahe", "clip")):
            if self._on(r, k):
                t, _ = A.TRANSFORMS[k](t, rng=r, **{a: self._U(r, k, a)})
        if self._on(r, "white_balance"):
            lo, hi = self.pol["white_balance"]["gain"]
            t, _ = A.white_balance(t, rng=r, gains=tuple(float(r.uniform(lo, hi)) for _ in range(3)))
        if self._on(r, "greyscale"):
            t, _ = A.greyscale(t)
        # ---- compression / blur / noise
        if self._on(r, "gaussian_blur"):
            t, _ = A.gaussian_blur(t, sigma=self._U(r, "gaussian_blur", "sigma_pitch") * pitch_px)
        if self._on(r, "gaussian_blur_px"):
            t, _ = A.gaussian_blur(t, sigma=self._U(r, "gaussian_blur_px", "sigma"))
        if self._on(r, "motion_blur"):
            lo, hi = self.pol["motion_blur"]["length"]
            t, _ = A.motion_blur(t, rng=r, length=int(r.integers(lo, hi + 1)))
        if self._on(r, "sharpen"):
            t, _ = A.sharpen(t, amount=self._U(r, "sharpen", "amount"))
        if self._on(r, "gaussian_noise"):
            t, _ = A.gaussian_noise(t, rng=r, sigma=self._U(r, "gaussian_noise", "sigma"))
        if self._on(r, "jpeg"):
            lo, hi = self.pol["jpeg"]["q"]
            t, _ = A.jpeg(t, q=int(r.integers(lo, hi + 1)))
        info = {"P": P, "Px": Px, "Py": Py, "steps": [s[0] for s in steps], "optics": ochain, "h_inch": hh, "v_inch": vv,
                "ppi": tuple(float(p) for p in ppi), "x0": x0, "y0": y0, "swap": swap}
        return np.ascontiguousarray(t), th, tv, info

    def __getitem__(self, k):
        it = self.items[k // self.tpi]
        r = np.random.default_rng([self.seed, k, random.getrandbits(32)])
        if isinstance(it, (list, tuple)):     # one SCENE: its files share the label (already mapped to
            it = it[int(r.integers(len(it)))]  # each file's orientation); one file drawn per sample
        t, th, tv, _ = self.sample(it, r)
        y = torch.tensor([math.log(th), math.log(tv)], dtype=torch.float32)
        return C.to_tensor(t), y, torch.tensor(it.weave), torch.tensor(it.w, dtype=torch.float32)


def make_dataset(items: Sequence[AugItem], policy: str, tiles_per_image: int, seed: int = 0, aniso: float = 0.5,
                 drop: Sequence[str] = (), rot90: bool = True):
    """'existing' -> the original cnn_common.TileDataset (flips, optional rot90, photometric_aug, P and
    anisotropic jitter); 'whitelist' / 'whitelist_ext' / 'whitelist_optics' -> AugTileDataset."""
    if policy == "existing":   # no geometric transform that needs the yarn tilt
        if any(isinstance(it, (list, tuple)) for it in items):
            raise ValueError("scene-grouped items need a whitelist policy (AugTileDataset)")
        base = [C.Item(it.img, it.path, it.h, it.v, it.weave, it.w) for it in items]
        return C.TileDataset(base, tiles_per_image=tiles_per_image, train=True, rot90=rot90, seed=seed, aniso=aniso)
    return AugTileDataset(items, tiles_per_image=tiles_per_image, policy=policy, drop=drop, aniso=aniso, seed=seed)


# =============================================================================================
# test-time augmentation (V5): flips, rot90 / rot270 with h <-> v swap, small rotations
# =============================================================================================
def _grid(im, stride):
    H, W = im.shape[:2]
    out = []
    for n, b in ((H, int(BORDER * H)), (W, int(BORDER * W))):
        hi = n - b - TILE
        pos = list(range(b, hi + 1, stride)) if hi >= b else [max(0, (n - TILE) // 2)]
        if pos[-1] != hi and hi >= b:
            pos.append(hi)
        out.append(pos)
    return [(y, x) for y in out[0] for x in out[1]]


@torch.no_grad()
def predict_crop_tta(model, crop, device, Ps=C.P_TEST, stride=112, flips=True, rot90=True,
                     degs=(-5.0, 0.0, 5.0), batch=128):
    """Per-inch (h, v): median over tiles x scales x {identity, hflip, vflip, rot90, rot270} x small
    rotations ('inner' crop, px/inch kept) of the log per-inch estimate.  rot90 / rot270 views are
    mapped back by swapping the two outputs (exact).  Small rotations keep the intrinsic density and
    change a crossing count by cos(a+-d)/cos(a); the +-d pair averages to cos d (-0.38 % at 5 deg),
    which is left uncorrected (the student is trained on a mix of density and crossing targets)."""
    model.eval()
    logs, wps = [], []
    for P in Ps:
        base = C.resize_to(crop, P)
        for d in degs:
            im = base if d == 0 else A.rotate(base, deg=float(d), mode="inner")[0]
            tiles = [im[y:y + TILE, x:x + TILE] for y, x in _grid(im, stride)]
            views = [(tiles, False)]
            if flips:
                views += [([t[:, ::-1] for t in tiles], False), ([t[::-1] for t in tiles], False)]
            if rot90:
                views += [([np.rot90(t, 1) for t in tiles], True), ([np.rot90(t, 3) for t in tiles], True)]
            for vt, swap in views:
                for k in range(0, len(vt), batch):
                    xb = torch.stack([C.to_tensor(np.ascontiguousarray(t)) for t in vt[k:k + batch]]).to(device)
                    pc, pw = model(xb)
                    pc = pc.float().cpu().numpy() + math.log(P / TILE)
                    if swap:
                        pc = pc[:, ::-1]
                    logs.append(pc)
                    wps.append(torch.softmax(pw.float(), -1).cpu().numpy())
    L = np.concatenate(logs)
    W = np.concatenate(wps)
    hv = np.exp(np.median(L, 0))
    q = np.exp(np.percentile(L, [25, 75], axis=0))
    return {"h": float(hv[0]), "v": float(hv[1]), "weave_probs": W.mean(0),
            "h_iqr": float(q[1, 0] - q[0, 0]), "v_iqr": float(q[1, 1] - q[0, 1]), "n_views": int(len(L))}

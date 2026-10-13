"""Shared pieces for the learned thread-density baselines.

Tile formulation (resolution-aware)
-----------------------------------
Every rectified crop is exactly 1 inch x 1 inch.  We resample it to P x P
pixels (P = "pixels per inch", randomised during training), cut 224 x 224
tiles from its interior, and regress the number of yarns crossing the tile:

    tile_h = h_count * 224 / P      tile_v = v_count * 224 / P

At test time the per-inch estimate is   exp(pred_log_tile) * P / 224,
aggregated (median) over a grid of tiles, a few P values and flips.  This
(a) multiplies the number of training samples ~10-30x per image, (b) makes
the model see a known physical scale, and (c) lets scale jitter act as a
label-consistent augmentation.

Axis convention: h_count = threads crossed walking horizontally (vertical
yarns per inch), v_count = threads crossed walking vertically.
Flips (h or v) never change (h_count, v_count); a 90-degree rotation swaps
them (optional --rot90 augmentation swaps the labels accordingly).

Anisotropy (added after the first synthetic run): a model trained only with
isotropic scale jitter learned the geometric-mean density (log-mean r=0.97 on
synthetic validation) but almost no per-axis information (predicted log(h/v)
std 0.026 vs 0.26 true).  Two label-consistent remedies:
  * anisotropic scale jitter: x and y are resampled to independent px/inch
    Px, Py = P * exp(+-aniso/2) -> tile labels h*224/Px, v*224/Py, so the
    warp/weft ratio of every training tile is randomised;
  * an explicit loss on the log-ratio  (pred_h - pred_v) vs (y_h - y_v).
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TILE = 224
P_TRAIN = (576, 960)          # px / inch range used during training
P_TEST = (672, 768, 864)      # px / inch used at test time (TTA)
BORDER = 0.035                # fraction of the crop ignored on each side
WEAVE_CLASSES = ["plain", "twill", "satin", "other"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def weave_to_idx(w) -> int:
    """Map free-text weave labels to the coarse class index (-100 = unknown)."""
    if w is None or (isinstance(w, float) and math.isnan(w)):
        return -100
    s = str(w).strip().lower().replace(" ", "")
    if s in ("", "nan", "none", "?", "unknown", "unsure", "unclear"):
        return -100
    if s in ("plain", "1/1", "11", "tabby", "poplin"):
        return 0
    if "twill" in s or s in ("2/1", "1/2", "3/1", "1/3", "2/2", "21", "31", "22") or "herring" in s \
            or "denim" in s or "drill" in s or "gabardine" in s:
        return 1
    if "satin" in s or "sateen" in s:
        return 2
    return 3


def resize_to(img: np.ndarray, P: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > P else cv2.INTER_CUBIC
    return cv2.resize(img, (P, P), interpolation=interp)


def cap_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    if img.shape[0] > max_side or img.shape[1] > max_side:
        return cv2.resize(img, (max_side, max_side), interpolation=cv2.INTER_AREA)
    if img.shape[0] != img.shape[1]:  # anisotropic -> square (crop is 1x1 inch)
        s = max(img.shape[:2])
        return cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
    return img


# ---------------------------------------------------------------------------
# augmentation (numpy / cv2, fast, label-preserving)
# ---------------------------------------------------------------------------
def photometric_aug(t: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """t: HxWx3 uint8 BGR tile -> augmented uint8 tile (no geometric change)."""
    x = t.astype(np.float32) / 255.0
    if rng.random() < 0.8:  # brightness / contrast / gamma
        x = x * rng.uniform(1 - 0.35 * strength, 1 + 0.35 * strength)
        m = x.mean()
        x = (x - m) * rng.uniform(1 - 0.4 * strength, 1 + 0.4 * strength) + m
        x = np.clip(x, 0, 1) ** rng.uniform(1 - 0.3 * strength, 1 + 0.3 * strength)
    if rng.random() < 0.6:  # saturation + hue (channel mixing) jitter
        g = x.mean(-1, keepdims=True)
        x = g + (x - g) * rng.uniform(1 - 0.6 * strength, 1 + 0.4 * strength)
        if rng.random() < 0.5:
            x = x[..., np.random.permutation(3)]
        x = x * np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.3 * strength + 0.01))
    if rng.random() < 0.3:
        x = x + np.random.normal(0, rng.uniform(0.003, 0.03 * strength + 0.001), x.shape).astype(np.float32)
    if rng.random() < 0.25:  # partial shadow (hard edge) on the tile
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        a = rng.uniform(0, 2 * np.pi)
        d = (xx - w / 2) * math.cos(a) + (yy - h / 2) * math.sin(a) - rng.uniform(-w / 2, w / 2)
        m = np.clip(d / max(1.0, rng.uniform(1, 12)), 0, 1)[..., None]
        x = x * (1 - rng.uniform(0.2, 0.6) * m)
    x = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:  # JPEG
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


def to_tensor(tile_bgr: np.ndarray) -> torch.Tensor:
    x = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@dataclass
class Item:
    img: Optional[np.ndarray]   # BGR square crop (or None -> load from path)
    path: Optional[str]
    h: float                    # per-inch count along x (vertical yarns / inch)
    v: float                    # per-inch count along y
    weave: int = -100           # coarse weave index or -100
    w: float = 1.0              # sample weight (e.g. 0.5 for flagged GT)


class TileDataset(torch.utils.data.Dataset):
    """Random 224-px tiles from 1-inch crops, resampled at random px/inch."""

    def __init__(self, items: Sequence[Item], tiles_per_image: int = 8, train: bool = True,
                 p_range=P_TRAIN, rot90: bool = False, aug_strength: float = 1.0,
                 max_side: int = 1024, seed: int = 0, aniso: float = 0.0):
        self.items = list(items)
        self.tpi = tiles_per_image
        self.train = train
        self.p_range = p_range
        self.rot90 = rot90
        self.aug = aug_strength
        self.max_side = max_side
        self.seed = seed
        self.aniso = aniso  # total log-range of the x/y px-per-inch ratio jitter

    def __len__(self):
        return len(self.items) * self.tpi

    def _load(self, it: Item):
        if it.img is not None:
            return it.img
        im = cv2.imread(it.path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(it.path)
        return cap_side(im, self.max_side)

    def __getitem__(self, k):
        it = self.items[k // self.tpi]
        rng = random.Random((self.seed * 1_000_003 + k * 7919 + random.getrandbits(20)) if self.train else k)
        img = self._load(it)
        P = int(rng.uniform(*self.p_range)) if self.train else int(np.mean(self.p_range))
        Px = Py = P
        if self.train and self.aniso > 0:
            r = math.exp(rng.uniform(-self.aniso / 2, self.aniso / 2))
            Px, Py = int(round(P * math.sqrt(r))), int(round(P / math.sqrt(r)))
        interp = cv2.INTER_AREA if img.shape[0] > max(Px, Py) else cv2.INTER_CUBIC
        im = cv2.resize(img, (Px, Py), interpolation=interp)
        bx, by = int(BORDER * Px), int(BORDER * Py)
        hx, hy = Px - bx - TILE, Py - by - TILE
        if hx <= bx or hy <= by:
            raise ValueError(f"P=({Px},{Py}) too small for TILE={TILE}")
        y0 = rng.randint(by, hy)
        x0 = rng.randint(bx, hx)
        t = im[y0:y0 + TILE, x0:x0 + TILE]
        h, v = it.h * TILE / Px, it.v * TILE / Py   # yarns crossing the tile
        if self.train:
            if rng.random() < 0.5:
                t = t[:, ::-1]
            if rng.random() < 0.5:
                t = t[::-1]
            if self.rot90 and rng.random() < 0.5:
                t = np.ascontiguousarray(np.rot90(t))
                h, v = v, h
            t = photometric_aug(np.ascontiguousarray(t), rng, self.aug)
        y = torch.tensor([math.log(h), math.log(v)], dtype=torch.float32)
        return to_tensor(np.ascontiguousarray(t)), y, torch.tensor(it.weave), torch.tensor(it.w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class CountNet(nn.Module):
    """ImageNet backbone -> (log tile_h, log tile_v) + coarse weave logits."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 n_weave: int = len(WEAVE_CLASSES), init_log_count: float = math.log(20.0)):
        super().__init__()
        import timm
        self.backbone_name = backbone
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.body.num_features
        self.count = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2))
        self.weave = nn.Linear(d, n_weave)
        nn.init.zeros_(self.count[-1].weight)
        nn.init.constant_(self.count[-1].bias, init_log_count)

    def forward(self, x):
        f = self.body(x)
        return self.count(f), self.weave(f)


def build_model(backbone: str, pretrained: bool = True) -> CountNet:
    return CountNet(backbone, pretrained=pretrained)


def save_ckpt(model: CountNet, path: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"backbone": model.backbone_name, "state": model.state_dict(), "extra": extra or {}}, path)


def load_ckpt(path: str, device="cpu") -> CountNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CountNet(ck["backbone"], pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


# ---------------------------------------------------------------------------
# losses / training loop
# ---------------------------------------------------------------------------
def loss_fn(pred_c, pred_w, y, wv, sw, lam_weave: float = 0.2, lam_ratio: float = 0.0):
    l_c = (F.smooth_l1_loss(pred_c, y, reduction="none", beta=0.05).mean(1) * sw).sum() / sw.sum()
    if lam_ratio > 0:  # explicit per-axis (anisotropy) term on log(h/v)
        lr = F.smooth_l1_loss(pred_c[:, 0] - pred_c[:, 1], y[:, 0] - y[:, 1], reduction="none", beta=0.05)
        l_c = l_c + lam_ratio * (lr * sw).sum() / sw.sum()
    if (wv >= 0).any():
        l_w = F.cross_entropy(pred_w, wv, ignore_index=-100)
    else:
        l_w = pred_w.sum() * 0.0
    return l_c + lam_weave * l_w, l_c.detach(), l_w.detach()


def train_loop(model, loader, device, epochs=None, max_steps=None, max_minutes=None, lr=3e-4,
               wd=1e-4, lam_weave=0.2, log_every=50, warmup=100, body_lr_mult=1.0, log=print,
               lam_ratio=0.0):
    """Simple AdamW + cosine schedule loop.  Stops at whichever of epochs /
    max_steps / max_minutes comes first.  Returns list of (step, loss)."""
    import time
    model.to(device).train()
    params = [{"params": model.body.parameters(), "lr": lr * body_lr_mult},
              {"params": list(model.count.parameters()) + list(model.weave.parameters()), "lr": lr}]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n_per_epoch = len(loader)
    total = max_steps or (epochs or 1) * n_per_epoch
    base = [g["lr"] for g in opt.param_groups]
    t0 = time.time()
    hist, step, done = [], 0, False
    run_c = run_w = 0.0
    while not done:
        for x, y, wv, sw in loader:
            x, y, wv, sw = x.to(device), y.to(device), wv.to(device), sw.to(device)
            frac = step / max(1, total)
            if max_minutes:
                frac = max(frac, (time.time() - t0) / 60 / max_minutes)
            sc = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * sc
            pc, pw = model(x)
            loss, lc, lw = loss_fn(pc, pw, y, wv, sw, lam_weave, lam_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_c = 0.95 * run_c + 0.05 * float(lc) if step else float(lc)
            run_w = 0.95 * run_w + 0.05 * float(lw) if step else float(lw)
            step += 1
            if step % log_every == 0:
                log(f"  step {step}/{total}  count-loss {run_c:.4f}  weave-loss {run_w:.3f}  "
                    f"lr {opt.param_groups[-1]['lr']:.2e}  {time.time() - t0:.0f}s")
                hist.append((step, run_c, run_w))
            if step >= total or (max_minutes and time.time() - t0 > max_minutes * 60):
                done = True
                break
    if not hist or hist[-1][0] != step:  # always record the final step count
        hist.append((step, run_c, run_w))
    model.eval()
    return hist


# ---------------------------------------------------------------------------
# inference on whole crops
# ---------------------------------------------------------------------------
def grid_tiles(im: np.ndarray, stride: int):
    P = im.shape[0]
    b = int(BORDER * P)
    hi = P - b - TILE
    pos = list(range(b, hi + 1, stride))
    if pos[-1] != hi:
        pos.append(hi)
    return [(y, x) for y in pos for x in pos]


@torch.no_grad()
def predict_crop(model, crop: np.ndarray, device, Ps=P_TEST, stride: int = 112, flips: bool = True,
                 batch: int = 64):
    """Per-inch (h, v) estimate for one rectified 1-inch crop.

    Returns dict(h, v, h_tiles, v_tiles, weave_probs, h_iqr, v_iqr)."""
    model.eval()
    logs, wps = [], []
    for P in Ps:
        im = resize_to(crop, P)
        tiles = [im[y:y + TILE, x:x + TILE] for y, x in grid_tiles(im, stride)]
        views = [tiles]
        if flips:
            views += [[t[:, ::-1] for t in tiles], [t[::-1] for t in tiles]]
        for vtiles in views:
            for k in range(0, len(vtiles), batch):
                xb = torch.stack([to_tensor(np.ascontiguousarray(t)) for t in vtiles[k:k + batch]]).to(device)
                pc, pw = model(xb)
                pc = pc.float().cpu().numpy() + math.log(P / TILE)  # -> log per-inch
                logs.append(pc)
                wps.append(torch.softmax(pw.float(), -1).cpu().numpy())
    L = np.concatenate(logs)
    W = np.concatenate(wps)
    hv = np.exp(np.median(L, 0))
    q = np.exp(np.percentile(L, [25, 75], axis=0))
    return {"h": float(hv[0]), "v": float(hv[1]), "h_tiles": np.exp(L[:, 0]), "v_tiles": np.exp(L[:, 1]),
            "weave_probs": W.mean(0), "h_iqr": float(q[1, 0] - q[0, 0]), "v_iqr": float(q[1, 1] - q[0, 1])}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def count_metrics(pred_h, pred_v, gt_h, gt_v, prefix=""):
    ph, pv, gh, gv = map(lambda a: np.asarray(a, float), (pred_h, pred_v, gt_h, gt_v))
    eh, ev = np.abs(ph - gh), np.abs(pv - gv)
    rh, rv = eh / gh, ev / gv
    both = np.concatenate([rh, rv])
    return {f"{prefix}n": int(len(gh)),
            f"{prefix}MAE_h": float(eh.mean()), f"{prefix}MAE_v": float(ev.mean()),
            f"{prefix}MAE": float(np.concatenate([eh, ev]).mean()),
            f"{prefix}MAPE_h": float(100 * rh.mean()), f"{prefix}MAPE_v": float(100 * rv.mean()),
            f"{prefix}n10_h": int((rh <= 0.10).sum()), f"{prefix}n10_v": int((rv <= 0.10).sum()),
            f"{prefix}MAPE": float(100 * both.mean()),
            f"{prefix}MdAPE": float(100 * np.median(both)),
            f"{prefix}acc@5%": float(100 * (both <= 0.05).mean()),
            f"{prefix}acc@10%": float(100 * (both <= 0.10).mean()),
            f"{prefix}bias_pct": float(100 * np.concatenate([(ph - gh) / gh, (pv - gv) / gv]).mean())}


def best_assignment(pred_h, pred_v, gt_a, gt_b):
    """For GT given as (warp, weft) without known orientation: per image pick
    the axis permutation with the smaller total absolute error.
    Returns (pa, pb) aligned with (gt_a, gt_b) and the boolean 'swapped'."""
    ph, pv, ga, gb = map(lambda a: np.asarray(a, float), (pred_h, pred_v, gt_a, gt_b))
    e_keep = np.abs(ph - ga) + np.abs(pv - gb)
    e_swap = np.abs(pv - ga) + np.abs(ph - gb)
    sw = e_swap < e_keep
    return np.where(sw, pv, ph), np.where(sw, ph, pv), sw

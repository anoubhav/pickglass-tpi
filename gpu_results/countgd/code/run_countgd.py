"""CountGD (NeurIPS 2024) baseline for counting threads per inch in rectified 1-inch crops.

PROTOCOL (fixed before looking at any error against the hand counts)
====================================================================
Input: a square BGR crop of side S px that spans exactly the 1-inch pick-glass window
(rect150: annotation-tool crops; legacy: the VLM harness' trimmed window, same code).

0. Label-free periodicity (also the exemplar generator).  p_x = spacing of the vertical
   yarns along x, p_y = spacing of the horizontal yarns along y, each from the windowed
   autocorrelation of the mean intensity profile of the central 1/4-inch band (detrended,
   lag range TPI 8..250, harmonic guard).  Reported on its own as the reference row
   "PER" (vertical = S/p_x, horizontal = S/p_y) so that CountGD's contribution can be
   separated from what the exemplar generator alone already knows.

1. Strips.  H strip = rows [S/2 - p_y, S/2 + p_y] x all columns (2 spacings of the
   perpendicular yarns tall, exactly 1 inch long); it is crossed once by every vertical
   yarn, and in a 2-pick window every end of a plain or 2/1 warp-faced weave is on top at
   least once.  V strip = the transpose (2 p_x wide, 1 inch tall).  Count in the H strip
   -> vertical_yarns, in the V strip -> horizontal_yarns.  Because the strip spans exactly
   1 inch, TPI = number of counted objects (no scale factor, no fitting).

2. Prompts.  Text "yarn" or "thread" (caption "<text> ."), either
     tx    : text only (no exemplars), or
     tx+ex : text + 3 visual exemplars.  Exemplar = one yarn segment: a box p_along wide
             (0.9 x spacing) and the full strip thickness, centred on a local maximum of the
             strip's across-averaged intensity profile (yarn centres are the bright ridges),
             the three maxima closest to 1/4, 1/2, 3/4 of the strip/window length.  No human
             labels anywhere.

3. Strip input modes.
     A (whole) : the whole strip is one CountGD input (CountGD's own resize: short side 800,
                 long side <= 1333).  Count = number of detections with score > 0.23.
     B (tiled) : CountGD's official adaptive-crop rule applied along the strip: windows of
                 4 x exemplar length (4 spacings) by the full strip thickness, overlap
                 1.25 x exemplar length, last window shifted to end at the strip end (code of
                 engine_inference.crop).  Windows are cut from the native-resolution strip and
                 resized by CountGD's transform; exemplars for a window are found inside that
                 window.  Every detection is counted once by the window whose centre is
                 nearest to the detection centre (Voronoi ownership; the official code uses
                 fractional 1/2, 1/4 weights in overlaps, which is the same in expectation).

4. Whole-crop interlacing count (W).  Text "yarn crossing", tx or tx+ex (exemplar = one
   interlacing cell p_x x p_y centred at an intersection of a column-profile maximum and a
   row-profile maximum).  Tiles of 12 p_x x 12 p_y (about 144 cells, well under the 900
   queries; official rule would be 4 x 4 cells but that needs ~1800 tiles for dense cloth),
   overlap 1.25 spacings, Voronoi ownership in 2-D.  Output: N = number of cells.  For plain
   weave N should equal ends x picks.  A per-axis TPI is derived WITHOUT a square root only
   from the detection lattice: ends = S / median horizontal nearest-neighbour spacing,
   picks = S / median vertical nearest-neighbour spacing (neighbours within +-30 deg of the
   axis).  sqrt(N) is logged as a diagnostic only.

Thresholds are CountGD's official FSC-147 ones (box_threshold 0.23, text_threshold 0) and
were not tuned.  Detections with score > 0.10 are stored so threshold sensitivity can be
studied post hoc.  Timing = wall clock per image and configuration (GPU synchronised),
excluding model load, after a warm-up pass.

Output: raw/countgd__{set}__shard{k}.jsonl, one record per (image, method, strip/whole).

    CUDA_VISIBLE_DEVICES=0 python run_countgd.py --set legacy
    CUDA_VISIBLE_DEVICES=1 python run_countgd.py --set rect150 --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cg_lib as L                       # noqa: E402

ROOT = L.ROOT
REPO = os.path.join(ROOT, "repo")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
HF_REPO, HF_REV = "nikigoli/CountGD", "6b989ad11db9413e44863ea5c57735cae6cacff8"
RAW_DIR = os.path.join(ROOT, "outputs", "raw")

BOX_THRESHOLD = 0.23          # official (config/cfg_fsc147_vit_b_test.py)
STORE_THRESHOLD = 0.10
MODEL_KEY = "countgd"

STRIP_PROMPTS = ("yarn", "thread")
W_PROMPT = "yarn crossing"


def method_name(kind, prompt=None, ex=0):
    if kind == "PER":
        return "PER"
    if kind == "W":
        return "CG-W-" + ("ex3" if ex else "tx")
    return f"CG-{kind}-{prompt}-" + ("ex3" if ex else "tx")


def all_methods():
    ms = []
    for mode in ("A", "B"):
        for p in STRIP_PROMPTS:
            for ex in (0, 3):
                ms.append((method_name(mode, p, ex), mode, p, ex))
    for ex in (0, 3):
        ms.append((method_name("W", None, ex), "W", W_PROMPT, ex))
    return ms


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class CountGD:
    def __init__(self, device="cuda"):
        import torch
        sys.path.insert(0, REPO)
        cwd = os.getcwd()
        os.chdir(REPO)                      # repo modules use relative imports from its root
        from util.slconfig import SLConfig
        import datasets_inference.transforms as T
        from models_inference.registry import MODULE_BUILD_FUNCS
        from util.misc import nested_tensor_from_tensor_list
        self.torch, self.T, self.nested = torch, T, nested_tensor_from_tensor_list
        cfg = SLConfig.fromfile(os.path.join(REPO, "config", "cfg_fsc147_vit_b_test.py"))
        cfg.merge_from_dict({"text_encoder_type": os.path.join(CKPT_DIR, "bert-base-uncased"),
                             "use_checkpoint": False, "use_transformer_ckpt": False})
        args = argparse.Namespace(device=device)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        assert args.box_threshold == BOX_THRESHOLD and args.text_threshold == 0
        build = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build(args)
        from safetensors.torch import load_file
        sd = load_file(os.path.join(CKPT_DIR, "hf_model", "model.safetensors"))
        res = model.load_state_dict(sd, strict=False)
        # the safetensors file stores tied tensors once (dec_pred_bbox_embed_share=True etc.):
        # every missing key must share storage with a parameter that IS in the file
        own = model.state_dict(keep_vars=True)
        ptr_in_file = {own[k].data_ptr() for k in sd if k in own}
        untied = [k for k in res.missing_keys if own[k].data_ptr() not in ptr_in_file]
        assert not untied, f"missing untied keys: {untied[:10]}"
        assert not res.unexpected_keys, res.unexpected_keys[:10]
        self.n_missing_tied = len(res.missing_keys)
        model.to(device).eval()
        os.chdir(cwd)
        self.model, self.device = model, device
        self.tf = T.Compose([T.RandomResize([800], max_size=1333),
                             T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, tiles_bgr, exemplars, caption, batch=8):
        """tiles_bgr: list of HxWx3 uint8 BGR arrays; exemplars: list of (k,4) xyxy px arrays
        (k = 0 or 3, same k for all tiles).  Returns list of (boxes_xyxy_px [n,4], scores [n])
        for detections with score > STORE_THRESHOLD, in each tile's pixel coordinates."""
        torch = self.torch
        from PIL import Image
        out = []
        for b0 in range(0, len(tiles_bgr), batch):
            imgs, exs, sizes = [], [], []
            for t, e in zip(tiles_bgr[b0:b0 + batch], exemplars[b0:b0 + batch]):
                pil = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
                ex = torch.as_tensor(np.asarray(e, np.float32).reshape(-1, 4))
                im, tg = self.tf(pil, {"exemplars": ex})
                imgs.append(im.to(self.device))
                exs.append(tg["exemplars"].to(self.device))
                sizes.append(t.shape[:2])
            with torch.no_grad():
                o = self.model(self.nested(imgs), exs,
                               [torch.tensor([0]).to(self.device) for _ in imgs],
                               captions=[caption + " ."] * len(imgs))
            logits = o["pred_logits"].sigmoid().max(dim=-1).values.float().cpu().numpy()
            boxes = o["pred_boxes"].float().cpu().numpy()             # cx, cy, w, h normalised
            for s, bx, (h, w) in zip(logits, boxes, sizes):
                keep = s > STORE_THRESHOLD
                bb = bx[keep]
                xyxy = np.stack([(bb[:, 0] - bb[:, 2] / 2) * w, (bb[:, 1] - bb[:, 3] / 2) * h,
                                 (bb[:, 0] + bb[:, 2] / 2) * w, (bb[:, 1] + bb[:, 3] / 2) * h], 1) \
                    if len(bb) else np.zeros((0, 4))
                out.append((xyxy, s[keep]))
        return out


# ----------------------------------------------------------------------------
# strips
# ----------------------------------------------------------------------------
def make_strip(img, per, axis):
    """axis 'h' -> horizontal strip (counts vertical yarns), 'v' -> vertical strip.
    Returns strip image in native orientation plus geometry in (along, across) terms."""
    S = img.shape[0]
    c = S / 2.0
    if axis == "h":
        p_along, p_perp = per["p_x"], per["p_y"]
        half = max(p_perp, 4.0)
        a, b = int(round(c - half)), int(round(c + half))
        strip = img[a:b, :]
        offset = (0, a)                         # (x, y) of the strip origin in the crop
    else:
        p_along, p_perp = per["p_y"], per["p_x"]
        half = max(p_perp, 4.0)
        a, b = int(round(c - half)), int(round(c + half))
        strip = img[:, a:b]
        offset = (a, 0)
    return strip, p_along, offset


def to_along(arr, axis):
    """image (rows, cols) -> (across, along) view."""
    return arr if axis == "h" else arr.transpose(1, 0, *range(2, arr.ndim))


def strip_exemplars(gray_ac, p_along, along0=0, along1=None):
    """3 exemplar boxes (along0', across0, along1', across1) in window coordinates."""
    along1 = gray_ac.shape[1] if along1 is None else along1
    win = gray_ac[:, along0:along1]
    mx, _ = L.profile_maxima(win, p_along)
    cs = L.pick_exemplar_centres(mx, win.shape[1], p_along)
    T = win.shape[0]
    half = 0.45 * p_along
    return [[max(0.0, c - half), 0.0, min(win.shape[1], c + half), float(T)] for c in cs]


def ac_box_to_xy(box, axis):
    a0, c0, a1, c1 = box
    return [a0, c0, a1, c1] if axis == "h" else [c0, a0, c1, a1]


def run_strip(cg, img, per, axis, mode, prompt, ex):
    strip, p_along, offset = make_strip(img, per, axis)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_ac = to_along(gray, axis)
    length, thick = g_ac.shape[1], g_ac.shape[0]
    if mode == "A":
        starts, win = [0], length
    else:
        starts, win = L.tile_starts(length, int(round(4 * p_along)), int(round(1.25 * p_along)))
    tiles, exs = [], []
    for s in starts:
        t = strip[:, s:s + win] if axis == "h" else strip[s:s + win, :]
        tiles.append(np.ascontiguousarray(t))
        if ex:
            exs.append([ac_box_to_xy(bx, axis) for bx in strip_exemplars(g_ac, p_along, s, s + win)])
        else:
            exs.append(np.zeros((0, 4)))
    res = cg(tiles, exs, prompt)
    dets = []                       # in strip coords: along_centre, across_centre, w, h, score, owned
    n_owned = 0
    for k, (s, (xyxy, sc)) in enumerate(zip(starts, res)):
        for bx, scv in zip(xyxy, sc):
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            along = (cx if axis == "h" else cy) + s
            owned = (L.owner_index(along, starts, win) == k)
            above = scv > BOX_THRESHOLD
            n_owned += int(owned and above)
            gx = bx + np.array([s, 0, s, 0]) if axis == "h" else bx + np.array([0, s, 0, s])
            gx = gx + np.array([offset[0], offset[1], offset[0], offset[1]])     # crop coords
            dets.append([round(float(v), 1) for v in gx] + [round(float(scv), 4), int(owned)])
    ex_crop = []
    for s, e in zip(starts, exs):
        for bx in np.asarray(e).reshape(-1, 4):
            sh = np.array([s, 0, s, 0]) if axis == "h" else np.array([0, s, 0, s])
            ex_crop.append([round(float(v), 1) for v in bx + sh + np.array([offset[0], offset[1]] * 2)])
    return dict(count=n_owned, n_tiles=len(starts), tile_len=int(win), strip_thick=int(thick),
                strip_len=int(length), p_along=per["p_x"] if axis == "h" else per["p_y"],
                detections=dets, exemplars_crop_xyxy=ex_crop)


# ----------------------------------------------------------------------------
# whole crop: interlacing cells
# ----------------------------------------------------------------------------
def run_whole(cg, img, per, prompt, ex, cell_tiles=12):
    S = img.shape[0]
    px, py = per["p_x"], per["p_y"]
    sx, wx = L.tile_starts(S, int(round(cell_tiles * px)), int(round(1.25 * px)))
    sy, wy = L.tile_starts(S, int(round(cell_tiles * py)), int(round(1.25 * py)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    tiles, exs, origins = [], [], []
    for y0 in sy:
        for x0 in sx:
            t = img[y0:y0 + wy, x0:x0 + wx]
            tiles.append(np.ascontiguousarray(t))
            origins.append((x0, y0))
            if ex:
                g = gray[y0:y0 + wy, x0:x0 + wx]
                mxs, _ = L.profile_maxima(g, px)                 # column maxima: warp centres
                mys, _ = L.profile_maxima(g.T, py)               # row maxima: weft centres
                cxs = L.pick_exemplar_centres(mxs, g.shape[1], px, fracs=(0.35, 0.5, 0.65))
                cys = L.pick_exemplar_centres(mys, g.shape[0], py, fracs=(0.35, 0.5, 0.65))
                exs.append([[max(0, cx - 0.5 * px), max(0, cy - 0.5 * py),
                             min(g.shape[1], cx + 0.5 * px), min(g.shape[0], cy + 0.5 * py)]
                            for cx, cy in zip(cxs, cys)])
            else:
                exs.append(np.zeros((0, 4)))
    res = cg(tiles, exs, prompt)
    dets, pts = [], []
    for k, ((x0, y0), (xyxy, sc)) in enumerate(zip(origins, res)):
        ix, iy = sx.index(x0), sy.index(y0)
        for bx, scv in zip(xyxy, sc):
            cx, cy = (bx[0] + bx[2]) / 2 + x0, (bx[1] + bx[3]) / 2 + y0
            owned = L.owner_index(cx, sx, wx) == ix and L.owner_index(cy, sy, wy) == iy
            gx = bx + np.array([x0, y0, x0, y0])
            dets.append([round(float(v), 1) for v in gx] + [round(float(scv), 4), int(owned)])
            if owned and scv > BOX_THRESHOLD:
                pts.append((cx, cy))
    N = len(pts)
    lat = lattice_from_points(np.array(pts, float).reshape(-1, 2), S)
    ex_crop = [[round(float(v), 1) for v in np.asarray(b) + np.array([o[0], o[1], o[0], o[1]])]
               for o, e in zip(origins, exs) for b in np.asarray(e).reshape(-1, 4)]
    return dict(count=N, n_tiles=len(tiles), tile_wh=[int(wx), int(wy)], detections=dets,
                exemplars_crop_xyxy=ex_crop[:30], **lat, sqrt_count=float(np.sqrt(N)))


def lattice_from_points(pts, S, max_angle_deg=30.0):
    """Per-axis lattice spacing from detected cell centres (sqrt-free):
    for each point, nearest neighbour whose direction is within +-30 deg of x (resp. y)."""
    out = dict(lat_dx_px=None, lat_dy_px=None, lat_ends=None, lat_picks=None, lat_n_dx=0, lat_n_dy=0)
    if len(pts) < 4:
        return out
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    kk = min(len(pts), 13)
    d, j = tree.query(pts, k=kk)
    tan = np.tan(np.deg2rad(max_angle_deg))
    dxs, dys = [], []
    for i in range(len(pts)):
        bx = by = None
        for dd, jj in zip(d[i, 1:], j[i, 1:]):
            if not np.isfinite(dd) or dd <= 0:
                continue
            vx, vy = pts[jj] - pts[i]
            if bx is None and abs(vy) <= tan * abs(vx):
                bx = dd
            if by is None and abs(vx) <= tan * abs(vy):
                by = dd
        if bx is not None:
            dxs.append(bx)
        if by is not None:
            dys.append(by)
    if dxs:
        out.update(lat_dx_px=float(np.median(dxs)), lat_ends=float(S / np.median(dxs)), lat_n_dx=len(dxs))
    if dys:
        out.update(lat_dy_px=float(np.median(dys)), lat_picks=float(S / np.median(dys)), lat_n_dy=len(dys))
    return out


# ----------------------------------------------------------------------------
def image_ids(set_name):
    return L.LEGACY_NAMES if set_name == "legacy" else [str(i) for i in range(150)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["legacy", "rect150"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--methods", default=None, help="comma list of method names (default: all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. 'dev' for smoke tests)")
    a = ap.parse_args()
    import torch
    k, n = map(int, a.shard.split("/"))
    ids = a.ids.split(",") if a.ids else image_ids(a.set)
    ids = ids[k::n]
    methods = all_methods()
    if a.methods:
        methods = [m for m in methods if m[0] in a.methods.split(",")]
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"{MODEL_KEY}__{a.set}__shard{k}of{n}{('__' + a.tag) if a.tag else ''}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["image_id"], r["method"], r["part"]))
            except Exception:
                pass
    phys_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    t0 = time.time()
    cg = CountGD()
    load_s = time.time() - t0
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[countgd/{a.set} shard {k}/{n}] loaded in {load_s:.1f}s on GPU {phys_gpu} ({gpu_name}); "
          f"{len(ids)} images, {len(methods)} methods; tied keys not in file: {cg.n_missing_tied}", flush=True)
    # warm-up
    cg([np.full((64, 512, 3), 128, np.uint8)], [np.zeros((0, 4))], "yarn")
    meta = dict(model=MODEL_KEY, repo=HF_REPO, revision=HF_REV, set=a.set, host=socket.gethostname(),
                gpu=phys_gpu, gpu_name=gpu_name, torch=torch.__version__, box_threshold=BOX_THRESHOLD)
    with open(out_path, "a") as f:
        for img_id in ids:
            img = L.load_image(a.set, img_id)
            S = int(img.shape[0])
            tp = time.time()
            per = L.estimate_periods(img)
            per_s = time.time() - tp

            def write(rec):
                f.write(json.dumps({**meta, "image_id": str(img_id), "side_px": S, **per, **rec}) + "\n")
                f.flush()

            if (str(img_id), "PER", "both") not in done:
                write(dict(method="PER", part="both", count_v=S / per["p_x"], count_h=S / per["p_y"],
                           seconds=per_s, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
            for name, mode, prompt, ex in methods:
                parts = ["whole"] if mode == "W" else ["h", "v"]
                for part in parts:
                    if (str(img_id), name, part) in done:
                        continue
                    torch.cuda.synchronize()
                    t = time.time()
                    try:
                        if mode == "W":
                            r = run_whole(cg, img, per, prompt, ex)
                        else:
                            r = run_strip(cg, img, per, part, mode, prompt, ex)
                        err = None
                    except Exception as e:             # logged, the run continues
                        import traceback
                        r, err = dict(count=None), traceback.format_exc()[-2000:]
                    torch.cuda.synchronize()
                    dt = time.time() - t + per_s           # periodicity is part of every method
                    write(dict(method=name, part=part, mode=mode, prompt=prompt, n_exemplars=ex,
                               seconds=dt, error=err, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **r))
                    print(f"  {img_id} {name}/{part}: count={r.get('count')} tiles={r.get('n_tiles')} "
                          f"{dt:.2f}s" + (f" ERROR {err.splitlines()[-1]}" if err else ""), flush=True)
            print(f"[{img_id}] S={S} p_x={per['p_x']:.2f} p_y={per['p_y']:.2f} "
                  f"PER v={S / per['p_x']:.1f} h={S / per['p_y']:.1f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

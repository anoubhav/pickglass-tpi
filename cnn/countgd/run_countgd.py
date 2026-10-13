#!/usr/bin/env python
"""CountGD (Amini-Naieni et al., NeurIPS 2024) as an open-world counting
baseline for thread density, on legacy 1-inch pick-glass crops.

What is counted
---------------
Counting specialists count *objects*, not yarns along a line.  The natural
countable object in a woven fabric is the yarn crossing (one warp x one weft
interlacing cell); crossings per square inch = warp/inch x weft/inch.  So a
crossing count over a known area A gives the GEOMETRIC-MEAN density
sqrt(C / A), which we compare with sqrt(warp x weft) from the hand counts.
Per-axis (warp vs weft) counts cannot be recovered from an object count.

Exemplars (no ground truth used)
--------------------------------
The local intensity lattice is estimated from the crop itself: the mean
magnitude spectrum of the row profiles (column profiles) of the central region
gives the dominant period along x (y) in the band 15-250 cycles/inch.  An
exemplar is a box of one period x one period centred on a local luminance
maximum (smoothed with sigma = period/4); 3 well-separated maxima near the
centre of each tile are used.  The lattice estimate is itself a classical
FFT count; it is reported alongside ("lattice_fft").

Tiling (CountGD returns at most 900 detections per image)
---------------------------------------------------------
Tiles are squares of side t = sqrt(250 / (fx * fy)) inch, i.e. at most ~250
lattice cells per tile (well below the 900-query cap); each tile is resized by
CountGD's own transform (short side 800).  Counting uses a central block of
k x k tiles (k = --block, default 2; the block is clipped to the central
0.70 x 0.70 inch, which avoids the frame); the counted area is exactly
(k t)^2.  (A first run on a2 over the full 0.70-inch region, 3x3 tiles,
exemplars only, is kept in results/run_log_full3x3_a2_partial.txt.)

Oracle-exemplar control (--oracle-cell): exemplar boxes (and tile size) use
the HAND-COUNT cell 1/warp x 1/weft inch instead of the image lattice, to
separate "can CountGD count crossings given the right exemplar?" from "can we
choose the exemplar automatically?".  Not a deployable setting.

Modes: exemplars only ("ex"), text + exemplars ("text+ex", text
"yarn crossing"), text only ("text").  Detection threshold 0.23 (the
authors' app default).

Run (venv_countgd; cwd-independent):
    venv_countgd/bin/python run_countgd.py --space <countgd_space dir> --names a2,a1,r2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WVAQ = os.path.dirname(os.path.dirname(HERE))
LEGACY_DIR = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "legacy2019")), "FR-final attempt", "images")
CONF_THRESH = 0.23
REGION = (0.15, 0.85)        # central region (inches) that is counted
MAX_CELLS_PER_TILE = 250


# ------------------------------------------------------------------ lattice
def dominant_period(gray: np.ndarray, axis: int, S: int, band=(15, 250)):
    """Period in px of the strongest component of the 1-D profiles along
    `axis` (1 = along rows -> x period).  Returns (period_px, freq_per_inch)."""
    g = gray.astype(np.float32)
    prof = g if axis == 1 else g.T
    prof = prof - prof.mean(1, keepdims=True)
    n = prof.shape[1]
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(prof * win, axis=1)).mean(0)
    freqs = np.fft.rfftfreq(n)                  # cycles / px
    cpi = freqs * S                             # cycles / inch
    ok = (cpi >= band[0]) & (cpi <= band[1])
    k = np.argmax(np.where(ok, spec, 0))
    # parabolic refinement
    if 1 <= k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    else:
        d = 0.0
    f = (k + d) / n
    return 1.0 / f, f * S


def exemplar_boxes(gray_tile: np.ndarray, px: float, py: float, n=3):
    """n boxes (x1, y1, x2, y2) of size px x py on well-separated local maxima
    near the tile centre."""
    sm = cv2.GaussianBlur(gray_tile.astype(np.float32), (0, 0), max(0.8, min(px, py) / 4))
    k = max(3, int(round(min(px, py) * 0.8)) | 1)
    mx = cv2.dilate(sm, np.ones((k, k), np.uint8))
    peaks = np.argwhere((sm >= mx - 1e-6) & (sm > np.percentile(sm, 50)))
    H, W = gray_tile.shape
    c = np.array([H / 2, W / 2])
    peaks = peaks[(peaks[:, 0] > py) & (peaks[:, 0] < H - py) & (peaks[:, 1] > px) & (peaks[:, 1] < W - px)]
    peaks = peaks[np.argsort(np.linalg.norm(peaks - c, axis=1))]
    chosen = []
    for p in peaks:
        if all(np.linalg.norm(p - q) > 2.0 * max(px, py) for q in chosen):
            chosen.append(p)
        if len(chosen) == n:
            break
    return [[float(x - px / 2), float(y - py / 2), float(x + px / 2), float(y + py / 2)] for y, x in chosen]


# ------------------------------------------------------------------ CountGD
def build(space: str, device: str):
    sys.path.insert(0, space)
    cwd = os.getcwd()
    os.chdir(space)  # cfg_app.py and checkpoints/ are relative to the space dir
    try:
        import torch
        from util.slconfig import SLConfig
        import datasets.transforms as T
        from models.registry import MODULE_BUILD_FUNCS
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
        cfg = SLConfig.fromfile("cfg_app.py")
        cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
        args = argparse.Namespace(device=device)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        torch.manual_seed(42)
        model, _, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
        ck = torch.load("checkpoint_best_regular.pth", map_location="cpu", weights_only=False)["model"]
        res = model.load_state_dict(ck, strict=False)
        model.eval().to(device)
    finally:
        os.chdir(cwd)
    return model, transform, res


def count_tile(model, transform, tile_rgb: np.ndarray, boxes, text: str, device: str):
    import torch
    from PIL import Image
    from util.misc import nested_tensor_from_tensor_list
    img = Image.fromarray(tile_rgb)
    inp, _ = transform(img, None)
    ex_img, tgt = transform(img, {"exemplars": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)})
    ex = tgt["exemplars"]
    with torch.no_grad():
        out = model(nested_tensor_from_tensor_list([inp.to(device)]),
                    nested_tensor_from_tensor_list([ex_img.to(device)]),
                    [ex.to(device)], [torch.tensor([0]).to(device)],
                    captions=[text + " ."])
    logits = out["pred_logits"].sigmoid()[0].max(-1).values.float().cpu().numpy()
    bx = out["pred_boxes"][0].float().cpu().numpy()
    keep = logits > CONF_THRESH
    return int(keep.sum()), bx[keep], logits


def legacy_crop(name: str):
    sys.path.insert(0, os.path.join(WVAQ, "methods"))
    import legacy as L
    rc, corners, _ = L.rectify_legacy(name)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="local copy of the nikigoli/countgd HF space (code + weights)")
    ap.add_argument("--names", default="a2,a1,r2")
    ap.add_argument("--modes", default="ex,text+ex,text")
    ap.add_argument("--text", default="yarn crossing")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-side", type=int, default=1200, help="crop resampled to this many px per inch")
    ap.add_argument("--block", type=int, default=2, help="count a central block of block x block tiles")
    ap.add_argument("--oracle-cell", action="store_true", help="control: exemplar = hand-count cell size")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = a.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    model, transform, res = build(os.path.abspath(a.space), dev)
    print(f"model built on {dev} in {time.time() - t0:.0f}s; missing keys {len(res.missing_keys)}, "
          f"unexpected {len(res.unexpected_keys)}", flush=True)
    sys.path.insert(0, os.path.dirname(HERE))
    import evaluate as E
    gt = {g.name: g for g in E.load_legacy_gt()}
    rows = []
    for name in a.names.split(","):
        crop = legacy_crop(name)
        S = a.max_side
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        r0, r1 = int(REGION[0] * S), int(REGION[1] * S)
        centre = gray[r0:r1, r0:r1]
        px, fx = dominant_period(centre, 1, S)
        py, fy = dominant_period(centre, 0, S)
        lat_fx, lat_fy = fx, fy
        g = gt[name]
        if a.oracle_cell:   # axis-matched: warp -> vertical yarns -> x
            fx, fy = g.warp, g.weft
            px, py = S / fx, S / fy
        side_in = REGION[1] - REGION[0]
        t_in = min(side_in / a.block, math.sqrt(MAX_CELLS_PER_TILE / (fx * fy)))
        n = a.block
        t_px = int(round(t_in * S))
        start = int(round(S / 2 - n * t_px / 2))
        edges = np.array([start + k * t_px for k in range(n + 1)])
        side_in = n * t_px / S
        g = gt[name]
        print(f"[{name}] GT warp {g.warp} weft {g.weft}; lattice fx {lat_fx:.1f} fy {lat_fy:.1f} /in; exemplar cell "
              f"{'ORACLE ' if a.oracle_cell else ''}{fx:.1f} x {fy:.1f} /in "
              f"(cell {px:.1f} x {py:.1f} px at {S} px/in); {n}x{n} tiles of {t_px} px, "
              f"counted area {side_in:.3f} in square", flush=True)
        vis = crop.copy()
        for mode in a.modes.split(","):
            total, t1, per_tile, nboxes = 0, time.time(), [], []
            for i in range(n):
                for j in range(n):
                    y0, y1, x0, x1 = edges[i], edges[i + 1], edges[j], edges[j + 1]
                    tile = crop[y0:y1, x0:x1]
                    boxes = exemplar_boxes(gray[y0:y1, x0:x1], px, py) if mode != "text" else []
                    text = "" if mode == "ex" else a.text
                    c, bx, _ = count_tile(model, transform, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), boxes, text, dev)
                    total += c
                    per_tile.append(c)
                    nboxes.append(len(boxes))
                    if mode == "ex":
                        th, tw = tile.shape[:2]
                        for b in bx:
                            cx, cy = x0 + b[0] * tw, y0 + b[1] * th
                            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                        for b in boxes:
                            cv2.rectangle(vis, (int(x0 + b[0]), int(y0 + b[1])), (int(x0 + b[2]), int(y0 + b[3])),
                                          (0, 255, 0), 2)
            area = side_in ** 2
            dens = math.sqrt(total / area) if total > 0 else 0.0
            gm = math.sqrt(g.warp * g.weft)
            row = dict(name=name, mode=mode, text="" if mode == "ex" else a.text, warp=g.warp, weft=g.weft,
                       gt_geomean=gm, lattice_fx=lat_fx, lattice_fy=lat_fy,
                       lattice_geomean=math.sqrt(lat_fx * lat_fy), oracle_cell=bool(a.oracle_cell),
                       exemplar_w_px=px, exemplar_h_px=py,
                       tiles=n * n, count=total, max_tile_count=max(per_tile), n_exemplars_per_tile=nboxes[0],
                       area_in2=area, countgd_geomean=dens, ape_countgd=100 * abs(dens - gm) / gm,
                       ape_lattice=100 * abs(math.sqrt(lat_fx * lat_fy) - gm) / gm, seconds=time.time() - t1)
            rows.append(row)
            import pandas as pd
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"countgd_legacy{a.tag}.csv"), index=False)
            print(f"   {mode:8s} count {total:5d} (max/tile {max(per_tile)}) -> sqrt(C/A) {dens:6.1f} /in  "
                  f"vs GT geo-mean {gm:.1f} (APE {row['ape_countgd']:.0f}%)  [{row['seconds']:.0f}s]", flush=True)
        cv2.rectangle(vis, (int(edges[0]), int(edges[0])), (int(edges[-1]), int(edges[-1])), (255, 255, 0), 2)
        cv2.imwrite(os.path.join(a.out, f"countgd_{name}_ex{a.tag}.jpg"),
                    cv2.resize(vis, (900, 900), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"countgd_legacy{a.tag}.csv"), index=False)
    print(df.round(1).to_string())
    with open(os.path.join(a.out, f"countgd_meta{a.tag}.json"), "w") as f:
        json.dump({"device": dev, "missing_keys": len(res.missing_keys), "unexpected_keys": len(res.unexpected_keys),
                   "conf_thresh": CONF_THRESH, "region_in": REGION, "max_cells_per_tile": MAX_CELLS_PER_TILE,
                   "text": a.text, "total_minutes": (time.time() - t0) / 60}, f, indent=1)


if __name__ == "__main__":
    main()

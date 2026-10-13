"""Verification check 2: synthetic h/v swap test (ground truth known exactly).

A fabric plane with VERTICAL yarns at H_TPI per inch and HORIZONTAL yarns at V_TPI per inch
(anisotropic grating) + two asymmetric low-frequency blobs (rotation direction markers) is
photographed through a perspective homography into an 864 x 1152 (portrait, like the dataset)
image; the 1-inch window polygon is stored TL,TR,BR,BL.  Then for every rot90 factor f = 0..3
and several random draws:
    np.rot90(image, f)  + key-point map (x, y) -> (y, W-1-x) applied f times
    -> albumentations-like Perspective (corner jitter 5 %) + ShiftScale (5 %) + GaussNoise
    -> rectify through common/rectify.py's contract (order_corners, homography, warpPerspective)
and we check
  T1  the key-point map is the true pixel map of np.rot90 (single marked pixel)
  T2  source crop measures (H_TPI, V_TPI)  (1-D FFT of column / row means)
  T3  aug crop measures (V_TPI, H_TPI) for f odd, (H_TPI, V_TPI) for f even  -> swap iff k in 90/270
  T4  the aug crop is np.rot90(source crop, f) (CCW): bright-blob position and NCC argmax
  T5  the builder's k recovery (aug_common.k_geometric and the NCC used for manifest k_deg) == 90 f
  T6  propagate_labels.propagate() on a manifest row built like build_aug1016 (swap_hv = k in
      90/270) yields exactly the measured aug (h, v), and swaps warp_dir; group == source_id
  T7  (info) the frozen counter on src / aug crops reports the swapped counts
  T8  line angles: a SHEARED fabric (vertical-yarn family normal at ALPHA, horizontal-yarn family
      normal at BETA, i.e. tool angles h_angle_deg = ALPHA, v_angle_deg = BETA) -> measured tool
      angles of the aug crop are (BETA, ALPHA) for k = 90/270 (swap, signs preserved) and
      (ALPHA, BETA) for k = 0/180, and propagate() produces exactly that
Run:  <venv2>/bin/python test_swap_synthetic.py      exit code 0 = all PASS
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, AUG)
import rectify as R  # noqa: E402
import aug_common as A  # noqa: E402

spec = importlib.util.spec_from_file_location("pl", os.path.join(AUG, "propagate_labels.py"))
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)

H_TPI, V_TPI = 37.0, 23.0       # vertical yarns / inch (crossed by the H line), horizontal yarns / inch
W0, H0 = 864, 1152
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def render_source(rng):
    """Analytic rendering: image pixel -> fabric (u, v) in inches via inverse homography."""
    win = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)            # inches
    c = np.array([W0 / 2, H0 / 2])
    side = 470.0
    ang = np.radians(rng.uniform(-6, 6))
    base = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * side
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    quad = (base @ rot.T + c + rng.uniform(-18, 18, (4, 2))).astype(np.float32)  # mild perspective
    Hm = cv2.getPerspectiveTransform(win, quad)                                  # inch -> px
    Hi = np.linalg.inv(Hm)
    xs, ys = np.meshgrid(np.arange(W0, dtype=np.float64), np.arange(H0, dtype=np.float64))
    P = np.stack([xs, ys, np.ones_like(xs)], -1) @ Hi.T
    u, v = P[..., 0] / P[..., 2], P[..., 1] / P[..., 2]
    g = (0.5 + 0.18 * np.cos(2 * np.pi * H_TPI * u) + 0.18 * np.cos(2 * np.pi * V_TPI * v)
         + 0.35 * np.exp(-((u - 0.22) ** 2 + (v - 0.27) ** 2) / (2 * 0.07 ** 2))      # bright marker
         - 0.30 * np.exp(-((u - 0.78) ** 2 + (v - 0.35) ** 2) / (2 * 0.09 ** 2))      # dark marker
         + 0.10 * u)                                                                  # gradient
    img = np.clip(g * 200 + 20, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, quad.astype(np.float64)        # stored order TL, TR, BR, BL


ALPHA, BETA = 4.0, -3.0


def render_sheared(rng):
    """Like render_source but the yarn families are tilted independently (shear) and the window is
    mapped by a pure similarity, so tool angles in the rectified crop are exactly (ALPHA, BETA)."""
    win = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    quad = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 470 + [200, 300]).astype(np.float32)
    Hi = np.linalg.inv(cv2.getPerspectiveTransform(win, quad))
    xs, ys = np.meshgrid(np.arange(W0, dtype=np.float64), np.arange(H0, dtype=np.float64))
    P = np.stack([xs, ys, np.ones_like(xs)], -1) @ Hi.T
    u, v = P[..., 0] / P[..., 2], P[..., 1] / P[..., 2]
    a, b = np.radians(ALPHA), np.radians(BETA)
    g = (0.5 + 0.2 * np.cos(2 * np.pi * H_TPI * (u * np.cos(a) + v * np.sin(a)))
         + 0.2 * np.cos(2 * np.pi * V_TPI * (-u * np.sin(b) + v * np.cos(b))))
    img = cv2.cvtColor(np.clip(g * 200 + 20, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img, quad.astype(np.float64)


def tool_angles(crop, n=256, pad=2048):
    """Tool convention (core.js): H line direction (cos a, sin a), V line (-sin b, cos b), y down,
    positive = clockwise on screen.  The H line is set perpendicular to the vertical yarns, i.e.
    along the wave vector of the family near the u axis; V likewise for the family near v."""
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64), (n, n), interpolation=cv2.INTER_AREA)
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g, (pad, pad))))
    fr = (np.arange(pad) - pad // 2) * n / pad / n * n      # cycles per crop side
    FX, FY = np.meshgrid(fr, fr)
    mag = np.hypot(FX, FY)
    out = []
    for near_u in (True, False):
        reg = (mag > 8) & (mag < 80) & ((np.abs(FY) < 0.3 * FX) if near_u else (np.abs(FX) < 0.3 * FY))
        idx = np.argmax(np.where(reg, F, 0))
        fx, fy = FX.flat[idx], FY.flat[idx]
        out.append(np.degrees(np.arctan2(fy, fx)) if near_u else np.degrees(np.arctan2(-fx, fy)))
    return out[0], out[1]


def rot90_keypoints(pts, f, W, H):
    p = np.asarray(pts, float).copy()
    for _ in range(f % 4):
        p = np.stack([p[:, 1], W - 1 - p[:, 0]], 1)
        W, H = H, W
    return p


def augment(img, pts, f, rng, noise=8.0):
    H, W = img.shape[:2]
    im = np.ascontiguousarray(np.rot90(img, f))
    p = rot90_keypoints(pts, f, W, H)
    Hh, Ww = im.shape[:2]
    # ShiftScale (no rotation) + Perspective (corner jitter <= 5 % of size), one homography
    s = 1 + rng.uniform(-0.05, 0.05)
    t = rng.uniform(-0.05, 0.05, 2) * [Ww, Hh]
    Aff = np.array([[s, 0, (1 - s) * Ww / 2 + t[0]], [0, s, (1 - s) * Hh / 2 + t[1]], [0, 0, 1]])
    src_c = np.array([[0, 0], [Ww - 1, 0], [Ww - 1, Hh - 1], [0, Hh - 1]], np.float32)
    dst_c = (src_c + rng.uniform(-0.05, 0.05, (4, 2)) * [Ww, Hh]).astype(np.float32)
    G = cv2.getPerspectiveTransform(src_c, dst_c) @ Aff
    im = cv2.warpPerspective(im, G, (Ww, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    q = np.hstack([p, np.ones((4, 1))]) @ G.T
    q = q[:, :2] / q[:, 2:]
    im = np.clip(im.astype(np.float32) + rng.normal(0, noise, im.shape), 0, 255).astype(np.uint8)
    return im, q


def rectify_contract(img, pts):
    c = R.order_corners(pts)
    Hm, S = R.homography(None, corners=c)
    return cv2.warpPerspective(img, Hm, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def measure(crop):
    """(h, v) in cycles per crop side from 1-D FFTs of the column / row means (inner 90 %)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    S = g.shape[0]

    def f1(prof):
        prof = prof - prof.mean()
        n = len(prof)
        spec = np.abs(np.fft.rfft(prof * np.hanning(n), n=8 * n))
        fr = np.fft.rfftfreq(8 * n) * S                    # cycles per S px (= per inch)
        band = (fr > 8) & (fr < 80)
        return float(fr[band][np.argmax(spec[band])])
    return f1(g.mean(0)), f1(g.mean(1))


def blob_pos(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    S = g.shape[0]
    g = cv2.GaussianBlur(cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA), (0, 0), 6)
    y, x = np.unravel_index(np.argmax(g), g.shape)
    return np.array([x, y]) / 128.0                       # fraction of side


def main():
    rng = np.random.default_rng(7)
    # T1: key-point map vs the true pixel motion of np.rot90
    ok = True
    for f in range(4):
        for _ in range(20):
            Wt, Ht = int(rng.integers(5, 40)), int(rng.integers(5, 40))
            x, y = int(rng.integers(0, Wt)), int(rng.integers(0, Ht))
            a = np.zeros((Ht, Wt), np.uint8)
            a[y, x] = 1
            b = np.rot90(a, f)
            yy, xx = np.argwhere(b)[0]
            q = rot90_keypoints([[x, y]], f, Wt, Ht)[0]
            q2 = A.rot90_points([[x, y]], 90 * f, (Wt, Ht))[0]
            ok &= (q[0] == xx and q[1] == yy and q2[0] == xx and q2[1] == yy)
    check("T1 rot90 key-point map == np.rot90 pixel map (mine and aug_common.rot90_points)", ok)

    try:
        CF_spec = importlib.util.spec_from_file_location(
            "cf", os.path.join(ROOT, "methods", "results", "counting_frozen_final.py"))
        CF = importlib.util.module_from_spec(CF_spec)
        CF_spec.loader.exec_module(CF)
    except Exception as e:  # informational only
        CF = None
        print("frozen counter not loadable:", e)

    t2 = t3 = t4 = t5 = t6 = True
    rows = []
    for trial in range(3):
        img, pts = render_source(rng)
        src = rectify_contract(img, pts)
        hs, vs = measure(src)
        t2 &= abs(hs - H_TPI) < 0.6 and abs(vs - V_TPI) < 0.6
        bs = blob_pos(src)
        # builder's NCC machinery on the synthetic source
        canon = R.order_corners(pts)
        rots = A.source_rotations(img, canon, src.shape[0])
        feats = [A.ncc_features(c)[0] for c in rots]
        m = A.ncc_features(src)[1]
        for f in range(4):
            aug_img, aug_pts = augment(img, pts, f, rng)
            crop = rectify_contract(aug_img, aug_pts)
            ha, va = measure(crop)
            exp = (V_TPI, H_TPI) if f % 2 else (H_TPI, V_TPI)
            ok3 = abs(ha - exp[0]) < 0.8 and abs(va - exp[1]) < 0.8
            t3 &= ok3
            # T4: direction -- blob of rot90(src, f) for an S x S array: (x, y) -> (y, 1 - x) per CCW turn
            e = bs.copy()
            for _ in range(f):
                e = np.array([e[1], 1 - e[0]])
            ba = blob_pos(crop)
            nc = [float(np.corrcoef(cv2.resize(np.rot90(src, j), (96, 96)).ravel(),
                                    cv2.resize(crop, (96, 96)).ravel())[0, 1]) for j in range(4)]
            ok4 = np.linalg.norm(ba - e) < 0.08 and int(np.argmax(nc)) == f
            t4 &= ok4
            # T5: builder's estimators
            g = A.k_geometric(pts, aug_pts)
            sc = A.ncc_rot(feats, A.ncc_features(crop)[0], m)
            k_photo = int(np.argmax(sc[:, 2])) * 90
            ok5 = g["k_deg"] == 90 * f and k_photo == 90 * f
            t5 &= ok5
            # T6: propagation on a manifest row built like build_aug1016 (k_deg = k_photo)
            man = pd.DataFrame([dict(aug_id=f"syn{trial}_aug{f}", source_id=1000 + trial, k_deg=k_photo,
                                     swap_hv=k_photo in (90, 270), usable=True, crop_path="x.jpg")])
            srcdf = pd.DataFrame([dict(source_id=1000 + trial, h_tpi=hs, v_tpi=vs, h_count=hs, v_count=vs,
                                       h_angle_deg=2.0, v_angle_deg=-3.0, warp_dir="vertical", weave="plain")])
            lab = PL.propagate(srcdf, man).iloc[0]
            ok6 = (abs(lab.h_tpi - ha) < 0.8 and abs(lab.v_tpi - va) < 0.8 and lab.group == 1000 + trial
                   and lab.warp_dir == ("horizontal" if f % 2 else "vertical")
                   and (lab.h_angle_deg, lab.v_angle_deg) == ((-3.0, 2.0) if f % 2 else (2.0, -3.0)))
            t6 &= ok6
            row = dict(trial=trial, f=f, k_geo=g["k_deg"], k_photo=k_photo, src_h=round(hs, 2), src_v=round(vs, 2),
                       aug_h=round(ha, 2), aug_v=round(va, 2), prop_h=round(lab.h_tpi, 2), prop_v=round(lab.v_tpi, 2),
                       blob_err=round(float(np.linalg.norm(ba - e)), 3), ncc_argmax=int(np.argmax(nc)) * 90,
                       ok3=ok3, ok4=ok4, ok5=ok5, ok6=ok6)
            if CF is not None:
                es, ea = CF.estimate(src, method="ours"), CF.estimate(crop, method="ours")
                row.update(cnt_src=(round(es["h_count"], 1), round(es["v_count"], 1)),
                           cnt_aug=(round(ea["h_count"], 1), round(ea["v_count"], 1)))
            rows.append(row)
    # T8: angle swap with signs preserved
    t8 = True
    img, pts = render_sheared(rng)
    src = rectify_contract(img, pts)
    ah, av = tool_angles(src)
    t8 &= abs(ah - ALPHA) < 0.4 and abs(av - BETA) < 0.4
    print(f"T8 source tool angles h={ah:.2f} v={av:.2f} (truth {ALPHA}, {BETA})")
    for f in range(4):
        H, W = img.shape[:2]
        crop = rectify_contract(np.ascontiguousarray(np.rot90(img, f)), rot90_keypoints(pts, f, W, H))
        bh, bv = tool_angles(crop)
        man = pd.DataFrame([dict(aug_id=f"shear_aug{f}", source_id=2000, k_deg=90 * f, swap_hv=f % 2 == 1,
                                 usable=True, crop_path="x.jpg")])
        lab = PL.propagate(pd.DataFrame([dict(source_id=2000, h_tpi=H_TPI, v_tpi=V_TPI, h_angle_deg=ah,
                                              v_angle_deg=av)]), man).iloc[0]
        ok = abs(lab.h_angle_deg - bh) < 0.5 and abs(lab.v_angle_deg - bv) < 0.5
        t8 &= ok
        print(f"T8 k={90 * f:3d}: measured aug angles h={bh:.2f} v={bv:.2f}; propagated h={lab.h_angle_deg:.2f} "
              f"v={lab.v_angle_deg:.2f} {'ok' if ok else 'MISMATCH'}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(HERE, "swap_synthetic_results.csv"), index=False)
    check("T2 source crop measures (H_TPI, V_TPI)", t2)
    check("T3 aug crop h/v swapped iff k in {90, 270}", t3)
    check("T4 aug crop == np.rot90(source crop, k/90) CCW (blob position + NCC argmax)", t4)
    check("T5 builder's k_geometric and NCC k both == applied 90*f", t5)
    check("T6 propagate(): labels equal measured aug (h, v); warp_dir, angles swapped; group = source", t6)
    check("T8 line angles swap with signs preserved (measured) and propagate() agrees", t8)
    if CF is not None and "cnt_src" in df:
        sw = [(abs(a[0] - (s[1] if f % 2 else s[0])) < 1.5 and abs(a[1] - (s[0] if f % 2 else s[1])) < 1.5)
              for s, a, f in zip(df.cnt_src, df.cnt_aug, df.f)]
        print(f"[INFO] T7 frozen counter follows the swap on {sum(sw)}/{len(sw)} synthetic crops")
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

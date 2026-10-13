"""ruler_calib.py -- pick-glass ruler self-calibration (WVAQ 2027, Job 3; no labels, no API).

The frame of the pick glass carries two engraved scales: a FINE one (26 ticks, long every 5) on
two adjacent edges and a COARSE one (11 ticks, long every other) on the other two.  They are
described as "mm" and "0.1 inch"; measured over the 150 photos their pitch ratio is 2.506
[2.501, 2.510] (ruler_summary.py), i.e. 1 : 2.5, so the code calls them 'mm' (fine, unit
pitch) and 'inch' (coarse, pitch GLASS['r10'] fine divisions) and works in FINE-DIVISION units.
This module finds the ticks in the ORIGINAL photo and turns them into a metric calibration:

    calibrate(photo_bgr, init_corners=None) -> dict
        ok          bool, a calibration was found
        corners     (4, 2) TL, TR, BR, BL window corners in ORIGINAL pixels (same convention as
                    common/rectify.py)
        H_img2mm    3x3 homography, original px -> frame plane in fine divisions (TL = (0, 0),
                    x right, y down); the window is the square [0, W_mm]^2
        W_mm        window side measured in THIS photo, in fine divisions (glass: ~26.0)
        obs         per edge: scale ('mm' fine | 'inch' coarse), tick foot points (px), indices k
        px_per_mm   {'top','right','bottom','left'}: px per fine division along each edge
        quality     no-reference consistency terms + 'score' in [0, 1] (see _quality)
    rectify_metric(photo_bgr, cal, S=None, mode='window'|'inch') -> S x S crop
        'window': the opening, as common/rectify.py (default S = round(mean side in px));
        'inch'  : a 25.4-fine-division square centred in the window (1 inch if fine = 1 mm).

Pipeline
--------
1. Working image: long side 1152 px (the ~1152x864 photos as is; 4608x3456 at x0.25);
   V = max(B, G, R) (the engraved ticks show strong chromatic fringes; the max merges them).
2. Tick mask = white top-hat th = V - open(V) (9-px ellipse) with th > 15 grey levels,
   th > 0.15 x local background and th > 4 x local median top-hat (high on a smooth, possibly
   lit bevel, low on fabric texture).  Connected components -> PCA segments (L/w >= 3).
3. Rows: sequential Hough vote of segment END POINTS for lines whose normal is within +-25 deg
   of the segment direction (a tick is perpendicular to its row); a peak is grown into a row on
   either side (consensus line through the inner ends, TLS refit, re-gather); its segments are
   removed and the vote repeated.  The line through the OTHER ends of the same ticks is kept as
   an alternative so that the quad search can choose the inner one.
4. Per row: foot point of each tick = intersection of its axis with the row line; a 1-D
   projective lattice t = (a k + b)/(c k + 1) gives integer indices k (gaps allowed).
5. Window: the 4 rows (two facing pairs, mutually perpendicular) maximising tick support, with
   each row's lattice spanning the nominal 25.4 fine or 10 coarse divisions (+-5 %, soft) between
   the corners and the glass layout (2 adjacent fine + 2 adjacent coarse edges) as a bonus.
   With init_corners the quad score is additionally weighted by agreement with that guess.
6. Metric fit (fit_metric): feet refined at full resolution (intensity centroid across each
   tick, 8 sections between 15 % and 70 % of its length); robust (Huber, 0.05) least squares
   for H, one free offset per edge and the window side W, with tick k of edge e at
   u = o_e + p_e k along the edge and at 0 or W across it (p = 1 fine, r10 coarse).  Tick
   indices are re-derived in the metric frame and off-lattice ticks dropped (reassign_and_refit).
7. Quality (no labels): RMS tick residual, |W - W_glass|/W_glass, tick coverage, layout.

Glass constants (GLASS) come from the ruler alone: a free fit (r10 and W free per photo) over the
150 photos gives r10 = 2.506 and W = 26.00 fine divisions (ruler_summary.py -> summary.json);
no human polygon or thread label is used.  Mask thresholds (rel_thr, med_k) were chosen among
four settings on the 150 photos by calibration success and the no-reference quality score.

Run as a script for debug overlays:  python ruler_calib.py 0 1 22 57 --out <dir>
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))

INCH_MM = 25.4
WORK_LONG = 1152
SCALES = {"mm": 1.0, "inch": 2.54}            # tick pitch in mm
N_INTERVALS = {"mm": INCH_MM / 1.0, "inch": 10.0}  # intervals across the 1-inch side
SIDES = ("top", "right", "bottom", "left")


# ============================================================================ segments
def _work_image(img):
    H, W = img.shape[:2]
    s = min(1.0, WORK_LONG / max(H, W))
    w = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    return w, s


MASK = {"se": 9, "abs_thr": 15.0, "rel_thr": 0.15, "med_k": 4.0}  # chosen by success + no-reference quality (see ruler_eval.py)


def tick_mask(work_bgr, se=None, abs_thr=None, rel_thr=None, med_k=None):
    """Bright thin marks: white top-hat th = V - open(V) above abs_thr, above rel_thr x the
    local background level, and (if med_k > 0) above med_k x the local MEDIAN top-hat (high on
    smooth, lit bevels; low on fabric texture, where the median top-hat is itself high)."""
    se = se or MASK["se"]
    abs_thr = MASK["abs_thr"] if abs_thr is None else abs_thr
    rel_thr = MASK["rel_thr"] if rel_thr is None else rel_thr
    med_k = MASK["med_k"] if med_k is None else med_k
    g = work_bgr.max(axis=2).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    op = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    th = g - op
    bg = cv2.blur(op, (15, 15))
    m = (th > abs_thr) & (th > rel_thr * bg)
    if med_k > 0:
        th8 = np.clip(th, 0, 255).astype(np.uint8)
        small = cv2.resize(th8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        med = cv2.medianBlur(small, 9)
        med = cv2.resize(med, (th.shape[1], th.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m &= th > med_k * (med + 2.0)
    return m.astype(np.uint8), th, op


def segments(mask, th, min_len=6, max_len=None, min_elong=3.0, bg=None):
    """Connected components -> list of dict(c, d, L, w, e0, e1, amp, bg, contrast)."""
    H, W = mask.shape
    max_len = max_len or 0.25 * min(H, W)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    idx_sorted = np.argsort(lab.ravel(), kind="stable")
    counts = np.bincount(lab.ravel(), minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < 5:
            continue
        bw, bh = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) < min_len or min(bw, bh) > max_len:
            continue
        flat = idx_sorted[starts[j]:starts[j + 1]]
        ys, xs = np.divmod(flat, W)
        P = np.stack([xs, ys], 1).astype(np.float64)
        wts = th[ys, xs].astype(np.float64)
        c = (P * wts[:, None]).sum(0) / wts.sum()
        Q = P - c
        cov = (Q * wts[:, None]).T @ Q / wts.sum()
        ev, evec = np.linalg.eigh(cov)
        d = evec[:, 1]
        if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        proj = Q @ d
        L = proj.max() - proj.min() + 1.0
        wdt = math.sqrt(max(ev[0], 1e-6)) * math.sqrt(12.0)
        if L < min_len or L > max_len or L / max(wdt, 1.0) < min_elong:
            continue
        amp = float(np.median(wts))
        b = float(np.median(bg[ys, xs])) if bg is not None else 0.0
        segs.append({"c": c, "d": d, "L": float(L), "w": float(wdt),
                     "e0": c + d * proj.min(), "e1": c + d * proj.max(), "amp": amp, "bg": b,
                     "contrast": amp / (b + 5.0)})
    return segs


# ============================================================================ rows
def _angle(v):
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def _angdiff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def hough_rows(segs, shape, dth=25.0, rho_bin=1.5, dtheta=0.5, min_votes=5, max_peaks=40):
    """Candidate row lines (theta deg of the NORMAL, rho) from segment end points.
    Coordinates are centred on the image; each end point votes once per theta (step dtheta)
    for theta within +-dth of its segment direction; counts are summed over +-1 rho bin only
    (summing over theta would mix rho values far from the centre), so a peak ~ number of
    collinear end points within ~4 px."""
    if not segs:
        return []
    H, W = shape
    cx, cy = W / 2.0, H / 2.0
    diag = 0.5 * math.hypot(H, W) + 2
    nth = int(round(180 / dtheta))
    nrho = int(2 * diag / rho_bin) + 2
    acc = np.zeros((nth, nrho), np.float32)
    steps = np.arange(-dth, dth + 1e-9, dtheta)
    for s in segs:
        phi = _angle(s["d"])
        ti = (np.round((phi + steps) / dtheta) % nth).astype(int)
        thr = np.radians(ti * dtheta)
        cs, sn = np.cos(thr), np.sin(thr)
        for e in (s["e0"], s["e1"]):
            rho = (e[0] - cx) * cs + (e[1] - cy) * sn
            ri = np.clip(np.round((rho + diag) / rho_bin).astype(int), 0, nrho - 1)
            acc[ti, ri] += 1.0
    acc_s = acc.copy()
    acc_s[:, 1:] += acc[:, :-1]
    acc_s[:, :-1] += acc[:, 1:]
    peaks = []
    a = acc_s
    for _ in range(max_peaks):
        t, r = np.unravel_index(int(np.argmax(a)), a.shape)
        if a[t, r] < min_votes:
            break
        th = t * dtheta
        rho_c = r * rho_bin - diag
        rho_abs = rho_c + cx * math.cos(math.radians(th)) + cy * math.sin(math.radians(th))
        peaks.append((float(th), float(rho_abs), float(a[t, r])))
        for dt in range(-8, 9):  # suppress neighbourhood (+-4 deg, +-6 px)
            a[(t + dt) % nth, max(0, r - 4):r + 5] = 0
    return peaks


def _line_from(theta_deg, rho):
    n = np.array([math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))])
    return n, rho


def _tls_line(P, w=None):
    P = np.asarray(P, np.float64)
    w = np.ones(len(P)) if w is None else np.asarray(w, np.float64)
    c = (P * w[:, None]).sum(0) / w.sum()
    Q = P - c
    cov = (Q * w[:, None]).T @ Q / w.sum()
    ev, evec = np.linalg.eigh(cov)
    n = evec[:, 0]
    return n, float(n @ c)


def _intersect_axis(c, d, n, rho):
    """Intersection of the line c + t d with {x : n.x = rho}."""
    den = n @ d
    if abs(den) < 1e-6:
        return None
    t = (rho - n @ c) / den
    return c + t * d


def _consensus_line(P, tol=3.0):
    """Exhaustive pairwise consensus line (deterministic RANSAC) -> inlier mask."""
    m = len(P)
    best, best_in = -1, None
    for a in range(m):
        for b in range(a + 1, m):
            d = P[b] - P[a]
            L = math.hypot(d[0], d[1])
            if L < 1e-6:
                continue
            nn = np.array([-d[1], d[0]]) / L
            r = np.abs((P - P[a]) @ nn)
            inl = r <= tol
            sc = inl.sum() - 1e-3 * r[inl].sum()
            if sc > best:
                best, best_in = sc, inl
    return best_in


def build_row(segs, ids, inner_is_e0, n, rho):
    """Refine a candidate row; returns row dict or None."""
    P = np.array([segs[j]["e0"] if ie else segs[j]["e1"] for j, ie in zip(ids, inner_is_e0)])
    if len(P) >= 4:
        inl = _consensus_line(P)
        if inl is not None and inl.sum() >= 4:
            ids = [j for j, k in zip(ids, inl) if k]
            inner_is_e0 = [ie for ie, k in zip(inner_is_e0, inl) if k]
    for _ in range(3):
        P = np.array([segs[j]["e0"] if ie else segs[j]["e1"] for j, ie in zip(ids, inner_is_e0)])
        if len(P) < 4:
            return None
        n2, rho2 = _tls_line(P)
        if n2 @ n < 0:
            n2, rho2 = -n2, -rho2
        n, rho = n2, rho2
        res = P @ n - rho
        mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        keep = np.abs(res) <= max(2.5, 3.0 * mad)
        ids = [j for j, k in zip(ids, keep) if k]
        inner_is_e0 = [ie for ie, k in zip(inner_is_e0, keep) if k]
    if len(ids) < 4:
        return None
    # side: direction (+n or -n) in which the ticks extend from the row line
    side = np.sign(np.median([(segs[j]["c"] - (segs[j]["e0"] if ie else segs[j]["e1"])) @ n
                              for j, ie in zip(ids, inner_is_e0)]))
    tdir = np.array([-n[1], n[0]])
    feet, lens, keep_ids = [], [], []
    for j in ids:
        f = _intersect_axis(segs[j]["c"], segs[j]["d"], n, rho)
        if f is None:
            continue
        feet.append(f)
        lens.append(segs[j]["L"])
        keep_ids.append(j)
    if len(feet) < 4:
        return None
    feet = np.array(feet)
    t = feet @ tdir
    o = np.argsort(t)
    row = {"n": n, "rho": rho, "side": float(side), "tdir": tdir, "ids": [keep_ids[i] for i in o],
           "feet": feet[o], "t": t[o], "L": np.array(lens)[o],
           "contrast": float(np.median([segs[keep_ids[i]]["contrast"] for i in o]))}
    lat = fit_lattice(row["t"])
    if lat is None:
        return None
    row.update(lat)
    return row


def fit_lattice(t, max_iter=4):
    """1-D projective lattice t = (a k + b)/(c k + 1) on sorted positions t (integer k)."""
    t = np.asarray(t, np.float64)
    if len(t) < 4:
        return None
    # merge near-duplicates (a tick broken into pieces)
    d = np.diff(t)
    if len(d) == 0:
        return None
    p0 = np.median(d[d > np.percentile(d, 20) * 0.5]) if np.any(d > 0) else 1.0
    keep = np.ones(len(t), bool)
    for i in range(1, len(t)):
        if t[i] - t[i - 1] < 0.35 * p0:
            keep[i] = False
    t = t[keep]
    idx_map = np.nonzero(keep)[0]
    if len(t) < 4:
        return None
    d = np.diff(t)
    # fundamental pitch: smallest-gap cluster
    q = np.sort(d)
    p = np.median(q[: max(2, len(q) // 2)])
    k = np.zeros(len(t), int)
    for i in range(1, len(t)):
        k[i] = k[i - 1] + max(1, int(round(d[i - 1] / p)))
    inl = np.ones(len(t), bool)
    abc = None
    for _ in range(max_iter):
        A = np.stack([k[inl], np.ones(inl.sum()), -k[inl] * t[inl]], 1).astype(np.float64)
        abc, *_ = np.linalg.lstsq(A, t[inl], rcond=None)
        a, b, c = abc
        pred = (a * k + b) / (c * k + 1.0)
        res = t - pred
        loc_p = np.abs(np.gradient(pred, k)) if len(np.unique(k)) > 1 else np.full(len(k), p)
        new = np.abs(res) <= 0.25 * np.maximum(loc_p, 1e-6)
        if new.sum() < 4 or np.array_equal(new, inl):
            inl = new if new.sum() >= 4 else inl
            break
        inl = new
    a, b, c = abc
    pred = (a * k + b) / (c * k + 1.0)
    res = t - pred
    pitch = np.abs((a - b * c) / (c * k + 1.0) ** 2)  # dt/dk
    return {"k": k, "lat_abc": np.array([a, b, c]), "lat_inlier": inl, "lat_res": res,
            "lat_rms_rel": float(np.sqrt(np.mean((res[inl] / pitch[inl]) ** 2))),
            "pitch_med": float(np.median(pitch[inl])), "lat_keep_idx": idx_map}


def t_to_k(row, t):
    a, b, c = row["lat_abc"]
    return (t - b) / (a - c * t)


def _gather(segs, alive, th, rho_, n, tol=3.0, dth=25.0):
    groups = {1.0: ([], []), -1.0: ([], [])}
    for j, sg in enumerate(segs):
        if not alive[j] or _angdiff(_angle(sg["d"]), th) > dth:
            continue
        for ie, e, other in ((True, sg["e0"], sg["e1"]), (False, sg["e1"], sg["e0"])):
            if abs(e @ n - rho_) <= tol:
                sd = 1.0 if (other - e) @ n > 0 else -1.0
                groups[sd][0].append(j)
                groups[sd][1].append(ie)
                break
    return groups


def grow_row(segs, alive, n, rho_, side, tol=3.0, iters=4):
    """Grow a row from a line guess: gather end points on the line whose segment extends to
    `side`, refit (TLS), re-gather with the refined line over all alive segments."""
    ids = []
    for _ in range(iters):
        ids, ies = [], []
        for j, sg in enumerate(segs):
            if not alive[j]:
                continue
            ang = math.degrees(math.atan2(n[1], n[0])) % 180.0
            if _angdiff(_angle(sg["d"]), ang) > 25.0:
                continue
            for ie, e, other in ((True, sg["e0"], sg["e1"]), (False, sg["e1"], sg["e0"])):
                if abs(e @ n - rho_) <= tol and ((other - e) @ n) * side > 0:
                    ids.append(j)
                    ies.append(ie)
                    break
        if len(ids) < 4:
            return None
        P = np.array([segs[j]["e0"] if ie else segs[j]["e1"] for j, ie in zip(ids, ies)])
        n2, r2 = _tls_line(P)
        if n2 @ n < 0:
            n2, r2 = -n2, -r2
        if np.allclose(n2, n, atol=1e-4) and abs(r2 - rho_) < 0.05:
            break
        n, rho_ = n2, r2
    return build_row(segs, ids, ies, n, rho_)


def candidate_rows(segs, shape, max_rows=80, min_ticks=5):
    """Sequential Hough: strongest end-point line -> rows grown on either side of it (and, for
    each, the line through the OTHER end points of the same ticks, so that the quad search can
    pick the inner one) -> remove those segments -> re-vote."""
    alive = np.ones(len(segs), bool)
    rows = []
    tried = 0
    while len(rows) < max_rows and tried < 3 * max_rows and alive.sum() >= min_ticks:
        tried += 1
        live_idx = np.nonzero(alive)[0]
        peaks = hough_rows([segs[j] for j in live_idx], shape, max_peaks=1)
        if not peaks:
            break
        th, rho, votes = peaks[0]
        n, rho_ = _line_from(th, rho)
        used = []
        for side in (1.0, -1.0):
            r = grow_row(segs, alive, n, rho_, side)
            if r is None or len(r["t"]) < min_ticks:
                continue
            rows.append(r)
            used += r["ids"]
            ids2 = r["ids"]
            ies2 = [abs(segs[j]["e0"] @ r["n"] - r["rho"]) > abs(segs[j]["e1"] @ r["n"] - r["rho"]) for j in ids2]
            rho2 = float(np.median([(segs[j]["e0"] if ie else segs[j]["e1"]) @ r["n"] for j, ie in zip(ids2, ies2)]))
            r2 = build_row(segs, ids2, ies2, r["n"], rho2)
            if r2 is not None and len(r2["t"]) >= min_ticks:
                r2["alt_of"] = len(rows) - 1
                rows.append(r2)
        if used:
            alive[np.array(sorted(set(used)), int)] = False
        else:
            # nothing usable at this peak: drop the segments whose end points voted for it
            hit = False
            for j in live_idx:
                sg = segs[j]
                if _angdiff(_angle(sg["d"]), th) <= 25.0 and min(abs(sg["e0"] @ n - rho_), abs(sg["e1"] @ n - rho_)) <= 3.0:
                    alive[j] = False
                    hit = True
            if not hit:
                break
    return rows


# ============================================================================ quad search
def _isect(r1, r2):
    A = np.array([r1["n"], r2["n"]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([r1["rho"], r2["rho"]]))


def _row_span_score(r, pa, pb):
    """Ticks within the edge between corners pa, pb; lattice intervals across the edge."""
    ta, tb = pa @ r["tdir"], pb @ r["tdir"]
    lo, hi = min(ta, tb), max(ta, tb)
    L = hi - lo
    if L <= 0:
        return 0.0, None, 0
    inside = (r["t"] >= lo - 0.04 * L) & (r["t"] <= hi + 0.04 * L)
    n_in = int(inside.sum())
    try:
        nk = abs(t_to_k(r, hi) - t_to_k(r, lo))
    except Exception:
        return 0.0, None, n_in
    best, scale = 0.0, None
    for sc, N in N_INTERVALS.items():
        e = abs(nk - N) / N
        g = math.exp(-0.5 * (e / 0.05) ** 2)
        if g > best:
            best, scale = g, sc
    exp_ticks = N_INTERVALS[scale] + 1 if scale else 1
    cover = min(1.0, n_in / exp_ticks)
    out_frac = 1.0 - n_in / max(1, len(r["t"]))
    return best * cover * (1.0 - 0.5 * out_frac), scale, n_in


def find_quad(rows, shape, init_corners_work=None):
    """Best (top, right, bottom, left) row assignment."""
    H, W = shape
    best = None
    R = len(rows)
    if R < 3:
        return None
    for i in range(R):
        for j in range(i + 1, R):
            a, b = rows[i], rows[j]
            if abs(a["n"] @ b["n"]) < math.cos(math.radians(20)):
                continue
            # facing each other: b lies on the inner side of a, and vice versa
            pb = b["feet"].mean(0)
            pa = a["feet"].mean(0)
            if (pb @ a["n"] - a["rho"]) * a["side"] >= 0 or (pa @ b["n"] - b["rho"]) * b["side"] >= 0:
                continue
            sep = abs(pb @ a["n"] - a["rho"])
            if sep < 0.12 * min(H, W):
                continue
            for k in range(R):
                for l in range(k + 1, R):
                    if len({i, j, k, l}) < 4:
                        continue
                    c, d = rows[k], rows[l]
                    if abs(c["n"] @ d["n"]) < math.cos(math.radians(20)):
                        continue
                    if abs(a["n"] @ c["n"]) > math.sin(math.radians(30)):
                        continue
                    pc, pd = c["feet"].mean(0), d["feet"].mean(0)
                    if (pd @ c["n"] - c["rho"]) * c["side"] >= 0 or (pc @ d["n"] - d["rho"]) * d["side"] >= 0:
                        continue
                    sep2 = abs(pd @ c["n"] - c["rho"])
                    if not (0.6 < sep2 / sep < 1.6):
                        continue
                    pts = {}
                    ok = True
                    for (x, y, key) in ((a, c, "ac"), (a, d, "ad"), (b, c, "bc"), (b, d, "bd")):
                        p = _isect(x, y)
                        if p is None:
                            ok = False
                            break
                        pts[key] = p
                    if not ok:
                        continue
                    sa, sca, na = _row_span_score(a, pts["ac"], pts["ad"])
                    sb, scb, nb = _row_span_score(b, pts["bc"], pts["bd"])
                    sc_, scc, nc = _row_span_score(c, pts["ac"], pts["bc"])
                    sd_, scd, nd = _row_span_score(d, pts["ad"], pts["bd"])
                    score = (sa + sb + sc_ + sd_) * (na + nb + nc + nd)
                    # glass layout: the two mm edges are adjacent (one of a/b and one of c/d)
                    scs = [sca, scb, scc, scd]
                    if scs.count("mm") == 2 and scs.count("inch") == 2 and sca != scb and scc != scd:
                        score *= 1.5
                    if init_corners_work is not None:
                        q = order_quad(np.array([pts["ac"], pts["ad"], pts["bd"], pts["bc"]]))
                        side = np.linalg.norm(init_corners_work - np.roll(init_corners_work, -1, 0), axis=1).mean()
                        err = np.linalg.norm(q - init_corners_work, axis=1).max() / side
                        score *= math.exp(-0.5 * (err / 0.15) ** 2)
                    if best is None or score > best[0]:
                        best = (score, (i, j, k, l), pts, (sca, scb, scc, scd), (na, nb, nc, nd))
    return best


def order_quad(pts):
    import rectify as _R
    return _R.order_corners(pts)


# ============================================================================ metric fit
def _side_of(corners, row):
    """Which window side ('top', ...) a row line belongs to, from the ordered corners."""
    TL, TR, BR, BL = corners
    mids = {"top": (TL + TR) / 2, "right": (TR + BR) / 2, "bottom": (BR + BL) / 2, "left": (BL + TL) / 2}
    return min(mids, key=lambda s: abs(mids[s] @ row["n"] - row["rho"]))


def _refine_feet_fullres(img, row, s, segs):
    """Refine foot points at full resolution: intensity centroid across each tick near its
    inner end (between 15 % and 70 % of the tick length).  Returns feet in FULL-res px."""
    if s >= 0.999:
        return row["feet"].copy()
    g = img.max(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    n = row["n"]
    out = []
    for f, j in zip(row["feet"], row["ids"]):
        sg = segs[j]
        d = sg["d"] * (1 if (sg["c"] - f) @ sg["d"] > 0 else -1)  # pointing outwards
        perp = np.array([-d[1], d[0]])
        L = sg["L"]
        half_w = max(3.0, 2.0 * sg["w"]) / s
        offs = np.linspace(0.15 * L, 0.7 * L, 8) / s
        acr = np.arange(-half_w, half_w + 1e-9, 0.5)
        F = f / s
        cents = []
        for o in offs:
            P = F[None, :] + d[None, :] * o + perp[None, :] * acr[:, None]
            v = cv2.remap(g, P[:, 0].astype(np.float32)[None, :], P[:, 1].astype(np.float32)[None, :],
                          cv2.INTER_LINEAR)[0]
            v = v - np.percentile(v, 20)
            v[v < 0] = 0
            if v.sum() <= 0:
                continue
            cents.append((o, (acr * v).sum() / v.sum()))
        if len(cents) < 3:
            out.append(F)
            continue
        cents = np.array(cents)
        # tick axis in full res: F + d*o + perp*(alpha + beta*o) -> intersect with row line
        beta, alpha = np.polyfit(cents[:, 0], cents[:, 1], 1)
        p0 = F + perp * alpha
        dd = d + perp * beta
        dd /= np.linalg.norm(dd)
        q = _intersect_axis(p0, dd, n, row["rho"] / s)
        out.append(q if q is not None else F)
    return np.array(out)


def _homog_apply(Hm, P):
    P = np.asarray(P, np.float64)
    Q = np.hstack([P, np.ones((len(P), 1))]) @ Hm.T
    return Q[:, :2] / Q[:, 2:3]


# ============================================================================ main entry
# Glass constants, from the ruler itself (free fit over the 150 photos, ruler_summary.py:
# r10 = 2.506 [2.501, 2.510], W = 26.00 [25.98, 26.01]); rounded.  No human polygon used.
GLASS = {"r10": 2.5, "W_mm": 26.0}
GLASS_NOTE = "r10 = pitch of the coarse (10-division) scale in fine divisions; W_mm = window side in fine divisions"


def detect(img, init_corners=None):
    """Stages 1-5: segments -> rows -> quad.  Returns dict with rows assigned to sides."""
    work, s = _work_image(img)
    mask, th, op = tick_mask(work)
    segs = segments(mask, th, bg=cv2.blur(op, (15, 15)))
    rows = candidate_rows(segs, mask.shape)
    rows = _prune_rows(rows)
    det = {"ok": False, "scale_work": s, "n_segments": len(segs), "n_rows_cand": len(rows),
           "segs": segs, "rows_all": rows, "work_shape": mask.shape}
    icw = None if init_corners is None else np.asarray(init_corners, np.float64) * s
    best = find_quad(rows, mask.shape, icw)
    if best is None:
        det["fail"] = "no_quad"
        return det
    score, idx, pts, scs, ns = best
    corners_w = order_quad(np.array([pts["ac"], pts["ad"], pts["bd"], pts["bc"]]))
    sel = {}
    for ii, sc in zip(idx, scs):
        r = dict(rows[ii])
        r["scale"] = sc
        sd = _side_of(corners_w, r)
        r["side_name"] = sd
        sel[sd] = r
    if len(sel) < 4:
        det["fail"] = "side_assignment"
        return det
    det.update({"ok": True, "quad_score": float(score), "rows": sel, "corners_lines": corners_w / s})
    return det


def _prune_rows(rows, keep=28):
    """Keep the most tick-like candidate rows (many, regular, high-contrast ticks)."""
    def q(r):
        return len(r["t"]) * min(2.0, max(0.2, r["contrast"] - 0.6)) * math.exp(-r["lat_rms_rel"] / 0.1)
    return sorted(rows, key=q, reverse=True)[:keep]


EDGE_AX = {"top": (0, "0"), "bottom": (0, "W"), "left": (1, "0"), "right": (1, "W")}


def _row_obs(det, img, refine_fullres=True):
    """Per side: full-res feet + integer lattice index k oriented along the metric axis."""
    s = det["scale_work"]
    Hn = cv2.getPerspectiveTransform(det["corners_lines"].astype(np.float32),
                                     np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])).astype(np.float64)
    obs = {}
    for sd, r in det["rows"].items():
        feet = _refine_feet_fullres(img, r, s, det["segs"]) if refine_fullres else r["feet"] / s
        ax = EDGE_AX[sd][0]
        un = _homog_apply(Hn, feet)[:, ax]          # position along the edge, window units
        k = np.round(t_to_k(r, r["t"])).astype(int)
        if len(np.unique(k)) < 3:
            continue
        if np.corrcoef(k, un)[0, 1] < 0:
            k = -k
        k = k - k.min()
        # drop duplicates in k (keep the one closest to the lattice)
        keep = np.ones(len(k), bool)
        A = np.stack([k, np.ones(len(k))], 1)
        coef, *_ = np.linalg.lstsq(A, un, rcond=None)
        res = un - A @ coef
        for kv in np.unique(k):
            m = np.nonzero(k == kv)[0]
            if len(m) > 1:
                best = m[np.argmin(np.abs(res[m]))]
                keep[m] = False
                keep[best] = True
        # ticks outside the window span (+-6 % of the side) are not part of this scale
        keep &= (un > -0.06) & (un < 1.06)
        dirs, lens, wids = [], [], []
        for f, j in zip(r["feet"], r["ids"]):  # tick direction (pointing away from the window)
            sg = det["segs"][j]
            dirs.append(sg["d"] * (1.0 if (sg["c"] - f) @ sg["d"] > 0 else -1.0))
            lens.append(sg["L"] / s)
            wids.append(sg["w"] / s)
        obs[sd] = {"feet": feet[keep], "k": k[keep], "u_win": un[keep], "scale": r["scale"],
                   "lat_res_rel": r["lat_rms_rel"], "dirs": np.array(dirs)[keep],
                   "tick_len": np.array(lens)[keep], "tick_w": np.array(wids)[keep]}
    return obs


def fit_metric(obs, corners0, r10=None, W=None):
    """Joint robust fit.  Unknowns: homography (8), offsets o_side (one per row), and, unless
    fixed, W (window side, mm-scale units) and r10 (pitch of the 10-division scale).
    Model: H(foot) = (o_side + pitch_side * k, edge) with edge = 0 or W (x for top/bottom ticks,
    y for left/right ticks).  Returns dict(H, W, r10, o, res_mm (per side arrays))."""
    from scipy.optimize import least_squares
    sides = [sd for sd in SIDES if sd in obs]
    W0 = W if W is not None else GLASS["W_mm"]
    r0 = r10 if r10 is not None else GLASS["r10"]
    dst_c = np.array([[0, 0], [W0, 0], [W0, W0], [0, W0]], np.float64)
    H0 = cv2.getPerspectiveTransform(np.float32(corners0), np.float32(dst_c)).astype(np.float64)
    # normalise image coordinates for conditioning
    allp = np.vstack([obs[sd]["feet"] for sd in sides])
    m0, sc0 = allp.mean(0), allp.std() + 1e-9
    T = np.array([[1 / sc0, 0, -m0[0] / sc0], [0, 1 / sc0, -m0[1] / sc0], [0, 0, 1]])
    Hn0 = H0 @ np.linalg.inv(T)
    Hn0 /= Hn0[2, 2]
    o0 = []
    for sd in sides:
        P = _homog_apply(H0, obs[sd]["feet"])
        ax = EDGE_AX[sd][0]
        pitch = 1.0 if obs[sd]["scale"] == "mm" else r0
        o0.append(np.median(P[:, ax] - pitch * obs[sd]["k"]))
    free_W, free_r = W is None, r10 is None
    x0 = np.r_[Hn0.ravel()[:8], o0, [W0] if free_W else [], [r0] if free_r else []]
    feetn = {sd: _homog_apply(T, obs[sd]["feet"]) for sd in sides}

    def unpack(x):
        Hn = np.r_[x[:8], 1.0].reshape(3, 3)
        o = dict(zip(sides, x[8:8 + len(sides)]))
        j = 8 + len(sides)
        Wv = x[j] if free_W else W
        j += int(free_W)
        rv = x[j] if free_r else r10
        return Hn, o, Wv, rv

    def resid(x):
        Hn, o, Wv, rv = unpack(x)
        out = []
        for sd in sides:
            P = _homog_apply(Hn, feetn[sd])
            ax, e = EDGE_AX[sd]
            pitch = 1.0 if obs[sd]["scale"] == "mm" else rv
            u = o[sd] + pitch * obs[sd]["k"]
            ev = 0.0 if e == "0" else Wv
            out.append(P[:, ax] - u)
            out.append(P[:, 1 - ax] - ev)
        return np.concatenate(out)

    sol = least_squares(resid, x0, loss="huber", f_scale=0.05, x_scale="jac", max_nfev=2000)
    Hn, o, Wv, rv = unpack(sol.x)
    Hm = Hn @ T
    Hm /= Hm[2, 2]
    r = resid(sol.x)
    res = {}
    j = 0
    for sd in sides:
        n = len(obs[sd]["k"])
        res[sd] = np.hypot(r[j:j + n], r[j + n:j + 2 * n])
        j += 2 * n
    return {"H": Hm, "W": float(Wv), "r10": float(rv), "o": {k: float(v) for k, v in o.items()},
            "res_mm": res, "cost": float(sol.cost), "success": bool(sol.success)}


def reassign_and_refit(obs, fit, corners0, r10v, Wv, iters=3, out_frac=0.3):
    """Re-derive every tick's integer index in the metric frame of the current fit (fixes
    lattice gaps mis-rounded in 1-D), drop ticks off the lattice (> out_frac pitch) and
    ticks beyond the window (u < -0.6 or u > W + 0.6), refit.  Repeats until stable."""
    for _ in range(iters):
        changed = False
        new = {}
        for sd, o in obs.items():
            ax = EDGE_AX[sd][0]
            pitch = 1.0 if o["scale"] == "mm" else fit["r10"]
            u = _homog_apply(fit["H"], o["feet"])[:, ax]
            kf = (u - fit["o"][sd]) / pitch
            k = np.round(kf).astype(int)
            good = (np.abs(kf - k) < out_frac) & (u > -0.6) & (u < fit["W"] + 0.6)
            # one tick per index
            for kv in np.unique(k[good]):
                m = np.nonzero(good & (k == kv))[0]
                if len(m) > 1:
                    keep = m[np.argmin(np.abs(kf[m] - k[m]))]
                    good[m] = False
                    good[keep] = True
            if good.sum() < 4:
                good = np.ones(len(k), bool)
                k = o["k"]
            if not np.array_equal(k[good], o["k"]) or good.sum() != len(o["k"]):
                changed = True
            new[sd] = dict(o, **{key: v[good] for key, v in o.items()
                                 if isinstance(v, np.ndarray) and len(v) == len(good)})
            new[sd]["k"] = k[good]
        obs = new
        if not changed:
            break
        fit = fit_metric(obs, corners0, r10=r10v, W=Wv)
    return obs, fit


def calibrate(img, init_corners=None, refine_fullres=True, r10=None, W=None, free_W=True):
    """Full calibration (see module docstring).  By default r10 is fixed to the glass constant
    and W (window side in mm-scale units) is measured per photo."""
    det = detect(img, init_corners)
    out = {"ok": False, "scale_work": det["scale_work"], "n_segments": det["n_segments"],
           "n_rows_cand": det["n_rows_cand"], "_det": det}
    if not det["ok"]:
        out["fail"] = det.get("fail")
        return out
    obs = _row_obs(det, img, refine_fullres)
    if len(obs) < 4:
        out["fail"] = "rows_lost"
        return out
    r10v = GLASS["r10"] if r10 is None else r10
    Wv = None if free_W else (GLASS["W_mm"] if W is None else W)
    fit = fit_metric(obs, det["corners_lines"], r10=r10v, W=Wv)
    obs, fit = reassign_and_refit(obs, fit, det["corners_lines"], r10v, Wv)
    Wm = fit["W"]
    dst_c = np.array([[0, 0], [Wm, 0], [Wm, Wm], [0, Wm]], np.float64)
    corners = _homog_apply(np.linalg.inv(fit["H"]), dst_c)
    out.update({"ok": True, "corners": corners, "corners_lines": det["corners_lines"], "H_img2mm": fit["H"],
                "W_mm": Wm, "r10": fit["r10"], "offsets": fit["o"], "obs": obs, "res_mm": fit["res_mm"],
                "scales": {sd: obs[sd]["scale"] for sd in obs}})
    out["px_per_mm"] = {sd: local_px_per_mm(fit["H"], sd, Wm) for sd in obs}
    out["quality"] = _quality(out)
    out.update(tick_blur(img, out))
    return out


def local_px_per_mm(Hm, side, side_mm):
    """Length in px of 1 mm (mm-scale unit) along the given edge, at its midpoint."""
    Hi = np.linalg.inv(Hm)
    h = side_mm / 2
    mid = {"top": ((h - .5, 0), (h + .5, 0)), "bottom": ((h - .5, side_mm), (h + .5, side_mm)),
           "left": ((0, h - .5), (0, h + .5)), "right": ((side_mm, h - .5), (side_mm, h + .5))}[side]
    P = _homog_apply(Hi, np.array(mid, np.float64))
    return float(np.linalg.norm(P[1] - P[0]))


def _quality(cal):
    """No-reference consistency of one calibration (no annotation, no thread labels):
    rms_mm      RMS tick residual of the joint fit (mm-scale units)
    W_dev       |W_photo - W_glass| / W_glass: the window measured in this photo's ruler vs. the
                glass constant (a wrong tick assignment, a missed scale or strong distortion
                shows up here)
    coverage    detected ticks / expected ticks on the four edges
    layout_ok   two adjacent mm edges and two adjacent 10-division edges
    score       exp(-rms/0.06) * exp(-W_dev/0.01) * coverage^0.5 * (1 if layout_ok else 0.5)"""
    res = np.concatenate([v for v in cal["res_mm"].values()])
    rms = float(np.sqrt(np.mean(res ** 2)))
    n_ticks = int(sum(len(o["k"]) for o in cal["obs"].values()))
    exp_ticks = sum((26 if o["scale"] == "mm" else 11) for o in cal["obs"].values())
    cover = min(1.0, n_ticks / exp_ticks)
    W_dev = abs(cal["W_mm"] - GLASS["W_mm"]) / GLASS["W_mm"]
    sc = cal["scales"]
    layout_ok = sorted(sc.values()) == ["inch", "inch", "mm", "mm"] and sc.get("top") != sc.get("bottom") \
        and sc.get("left") != sc.get("right")
    score = math.exp(-rms / 0.06) * math.exp(-W_dev / 0.01) * math.sqrt(cover) * (1.0 if layout_ok else 0.5)
    return {"rms_mm": rms, "n_ticks": n_ticks, "coverage": cover, "W_dev": W_dev,
            "layout_ok": bool(layout_ok), "score": float(score)}


def tick_blur(img, cal, n_sec=6, step=0.25):
    """Blur meter from the engraved ticks (a known-sharp target in every photo).  For every
    detected tick, intensity cross-sections (V = max channel, full resolution) at n_sec positions
    between 25 % and 65 % of its length; FWHM by half-maximum interpolation; sigma = FWHM/2.355.
    Returns the median sigma in px and in u = 1e-3 window side (degrade.py's unit), plus the
    number of ticks used.  The value includes the engraving width and the optics (chromatic
    fringes); it is a relative, no-reference sharpness measure, not a PSF estimate."""
    g = img.max(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    C = np.asarray(cal["corners"], np.float64)
    S = float(np.linalg.norm(C - np.roll(C, -1, 0), axis=1).mean())
    sig = []
    for sd, o in cal["obs"].items():
        if "dirs" not in o:
            continue
        for F, d, L, w in zip(o["feet"], o["dirs"], o["tick_len"], o["tick_w"]):
            perp = np.array([-d[1], d[0]])
            hw = max(6.0, 4.0 * w)
            acr = np.arange(-hw, hw + 1e-9, step)
            vals = []
            for off in np.linspace(0.25 * L, 0.65 * L, n_sec):
                P = F[None, :] + d[None, :] * off + perp[None, :] * acr[:, None]
                v = cv2.remap(g, P[:, 0].astype(np.float32)[None, :], P[:, 1].astype(np.float32)[None, :],
                              cv2.INTER_LINEAR)[0]
                base = np.percentile(v, 10)
                v = v - base
                i0 = int(np.argmax(v))
                pk = v[i0]
                if pk < 10 or i0 == 0 or i0 == len(v) - 1:
                    continue
                h = pk / 2
                lft = i0
                while lft > 0 and v[lft] > h:
                    lft -= 1
                rgt = i0
                while rgt < len(v) - 1 and v[rgt] > h:
                    rgt += 1
                if v[lft] > h or v[rgt] > h:
                    continue  # profile does not fall to half maximum inside the window
                xl = lft + (h - v[lft]) / (v[lft + 1] - v[lft] + 1e-9)
                xr = rgt - 1 + (v[rgt - 1] - h) / (v[rgt - 1] - v[rgt] + 1e-9)
                vals.append((xr - xl) * step / 2.3548)
            if len(vals) >= 3:
                sig.append(float(np.median(vals)))
    if not sig:
        return {"blur_px": np.nan, "blur_u": np.nan, "n_ticks_blur": 0}
    sp = float(np.median(sig))
    return {"blur_px": sp, "blur_u": 1000.0 * sp / S, "n_ticks_blur": len(sig)}


def rectify_metric(img, cal, S=None, mode="window"):
    """mode='window': S x S crop of the window opening (same convention as common/rectify.py,
    default S = round(mean side length in px)); counts across it are per window side.
    mode='inch': a 25.4 mm-scale-unit square centred in the window (true 1 inch if the
    mm scale is millimetres), S px per inch."""
    C = np.asarray(cal["corners"], np.float64)
    if S is None:
        S = int(round(float(np.linalg.norm(C - np.roll(C, -1, 0), axis=1).mean())))
    Wm = cal["W_mm"]
    if mode == "window":
        A = np.array([[S / Wm, 0, 0], [0, S / Wm, 0], [0, 0, 1]]) @ cal["H_img2mm"]
    else:
        off = (Wm - INCH_MM) / 2
        A = np.array([[S / INCH_MM, 0, -off * S / INCH_MM], [0, S / INCH_MM, -off * S / INCH_MM], [0, 0, 1]]) \
            @ cal["H_img2mm"]
    return cv2.warpPerspective(img, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ============================================================================ debug overlay
def overlay(img, cal, scale=None, gt_corners=None, thickness=None):
    H, W = img.shape[:2]
    s = scale or min(1.0, 1400 / max(H, W))
    vis = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    vis = (vis * 0.6).astype(np.uint8)
    t = thickness or 1
    cols = {"mm": (60, 220, 255), "inch": (255, 120, 60)}
    if gt_corners is not None:
        cv2.polylines(vis, [np.int32(np.asarray(gt_corners) * s).reshape(-1, 1, 2)], True, (0, 0, 255), t)
    if cal.get("ok"):
        for sd, o in cal["obs"].items():
            for P in o["feet"]:
                cv2.circle(vis, tuple(np.int32(P * s)), 3, cols[o["scale"]], -1)
        cv2.polylines(vis, [np.int32(cal["corners"] * s).reshape(-1, 1, 2)], True, (0, 255, 0), t)
    return vis


if __name__ == "__main__":
    import argparse
    import rectify as R
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int, default=[0, 1, 22, 57])
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i in a.ids:
        img = R.load_image(i)
        gt = R.load_corners(i)
        cal = calibrate(img)
        side_gt = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
        if cal["ok"]:
            err = np.linalg.norm(cal["corners"] - gt, axis=1) / side_gt * INCH_MM
            q = cal["quality"]
            print(i, img.shape[:2], "rows", {k: (o["scale"], len(o["k"])) for k, o in cal["obs"].items()},
                  "W", round(cal["W_mm"], 3), "corner err mm", err.round(2),
                  "q", {k: round(v, 3) if isinstance(v, float) else v for k, v in q.items()})
        else:
            print(i, "FAIL", cal.get("fail"), cal["n_segments"], cal["n_rows_cand"])
        cv2.imwrite(os.path.join(a.out, f"ov_{i}.jpg"), overlay(img, cal, gt_corners=gt))

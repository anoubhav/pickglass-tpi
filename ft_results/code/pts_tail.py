# =====================================================================================================
# [pts] EXACT YARN-CROSSING POINT LABELS (appended to the v2 generator by make_synth_pts.py)
# =====================================================================================================
# Every Molmo2 input in this project is a square view resized to VLM_SIDE = 644 px
# (vlm/data_vlm.resize_square) with a red guide line drawn by vlm/data_vlm.guide_line_view at row /
# column VLM_SIDE // 2.  For a view = crop[y0:y0+h, x0:x0+w] we map every sample of the guide line
# back through the generator's exact pixel chain
#     view px -> final crop px -> native crop px -> (inverse rectification) sensor px
#             -> render px (cv2.resize pixel-centre convention) -> (inverse perspective) render grid
# and read the continuous yarn coordinates U (warp) / V (weft) that render_fabric used to draw the
# yarns.  Warp yarn j's centre line is U = j + 0.5, so the vertical yarns crossing the horizontal
# guide line are the solutions of U(line(t)) = j + 0.5 (linear interpolation between samples taken
# every 1/8 view pixel).  Crossings outside the 1-inch window (frame border) are dropped.
VLM_SIDE = 644
POINT_LINE = VLM_SIDE // 2


def _bilinear(F, x, y):
    h, w = F.shape
    x = np.clip(x, 0, w - 1.000001)
    y = np.clip(y, 0, h - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    f = lambda a, b: F[a, b].astype(np.float64)  # noqa: E731
    return ((f(y0, x0) * (1 - fx) + f(y0, x0 + 1) * fx) * (1 - fy)
            + (f(y0 + 1, x0) * (1 - fx) + f(y0 + 1, x0 + 1) * fx) * fy)


class CropGeometry:
    """Maps final-crop pixel coordinates to the render grid and samples U, V there."""

    def __init__(self, U, V, Hm, H, hs, Hr, S0, S, ppi_r, x0_in):
        self.U, self.V = U, V
        self.Hm_inv = np.linalg.inv(Hm)
        self.Hr_inv = np.linalg.inv(Hr)
        self.H, self.hs, self.S0, self.S, self.ppi_r, self.x0_in = H, hs, S0, S, ppi_r, x0_in

    def to_render(self, px, py):
        px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
        if self.S != self.S0:                              # undo the final INTER_AREA resize
            px = (px + .5) * self.S0 / self.S - .5
            py = (py + .5) * self.S0 / self.S - .5
        s = cv2.perspectiveTransform(np.stack([px, py], -1)[None], self.Hr_inv)[0]
        r = (s + .5) * (self.H / self.hs) - .5               # sensor grid -> render grid
        pre = cv2.perspectiveTransform(r[None], self.Hm_inv)[0]
        return pre[:, 0], pre[:, 1]

    def fields(self, px, py):
        rx, ry = self.to_render(px, py)
        xin = self.x0_in + (rx + .5) / self.ppi_r
        yin = self.x0_in + (ry + .5) / self.ppi_r
        h, w = self.U.shape
        inside = (xin >= 0) & (xin <= 1) & (yin >= 0) & (yin <= 1) & (rx >= 0) & (ry >= 0) \
            & (rx <= w - 1) & (ry <= h - 1)
        return _bilinear(self.U, rx, ry), _bilinear(self.V, rx, ry), inside


def _crossings(g, t, inside):
    """Positions t at which g (sampled at increasing t) crosses an integer, both samples inside."""
    k = np.floor(g)
    out = []
    for i in np.nonzero((k[1:] != k[:-1]) & inside[1:] & inside[:-1])[0]:
        a, b = g[i], g[i + 1]
        lo, hi = min(a, b), max(a, b)
        for m in range(int(np.floor(lo)) + 1, int(np.floor(hi)) + 1):
            out.append(t[i] + (m - a) / (b - a) * (t[i + 1] - t[i]))
    return sorted(out)


def view_lines(geo, rect, side=VLM_SIDE, oversample=8):
    """rect = (x0, y0, w, h): the view is crop[y0:y0+h, x0:x0+w] resized to side x side.
    Returns h = x positions (view-normalised, 0..1) where VERTICAL yarns cross the horizontal guide
    line, v = y positions where HORIZONTAL yarns cross the vertical guide line, and line = the
    normalised position of both guide lines."""
    x0, y0, w, h = rect
    u = np.linspace(-0.5, side - 0.5, side * oversample + 1)
    c = side // 2
    cx = x0 + (u + .5) * w / side - .5
    cy = np.full_like(cx, y0 + (c + .5) * h / side - .5)
    U, _, ins = geo.fields(cx, cy)
    hx = _crossings(U - 0.5, u, ins)
    cy2 = y0 + (u + .5) * h / side - .5
    cx2 = np.full_like(cy2, x0 + (c + .5) * w / side - .5)
    _, V2, ins2 = geo.fields(cx2, cy2)
    vy = _crossings(V2 - 0.5, u, ins2)
    return dict(h=[round((q + .5) / side, 5) for q in hx], v=[round((q + .5) / side, 5) for q in vy],
                line=(c + .5) / side, inside_h=float(ins.mean()), inside_v=float(ins2.mean()))


def quarter_rect(S, which):
    """Identical arithmetic to vlm/data_vlm.quarter_patch (returns x0, y0, w, h)."""
    q = S / 4.0
    if which == "c":
        a = int(round(1.5 * q))
        n = int(round(q))
        return (a, a, n, n)
    k = int(which[1])
    a, b = int(round(k * q)), int(round((k + 1) * q))
    return (a, a, b - a, b - a)


def generate_pts(seed: int, max_crop_side: int = 1600, density_range=(30, 200), frame_margin: float = 0.1,
                 p_expand: float = 0.3, n_rand_views: int = 6):
    """synth.generate() (same RNG stream) + exact crossing labels for a fixed set of views.
    Extra randomness (frame-border expansion, random views) comes from a SEPARATE rng, so with
    p_expand=0 the crop equals synth.generate(seed, max_crop_side, density_range, frame_margin)."""
    rng = np.random.default_rng(seed)
    p = sample_params(rng, seed, density_range)
    p.frame_margin = frame_margin
    p.persp = min(p.persp, 0.6 * frame_margin / (1 + 2 * frame_margin))
    ppi_r = float(np.clip(max(5.0 * max(p.epi, p.ppi), 1.35 * p.r_eff), 450, 1200))
    side_in = 1 + 2 * p.frame_margin
    side_px = int(round(side_in * ppi_r))
    x0_in = -p.frame_margin
    refl, height, U, V = render_fabric(p, rng, ppi_r, side_px, x0_in, return_fields=True)
    scene, win, cg = camera(p, rng, refl, height, ppi_r, x0_in, return_geo=True)
    aux = np.random.default_rng([seed, 20261007])
    expand = None
    if aux.random() < p_expand:   # imperfect frame trimming: 0-3 % of frame visible on some sides
        expand = tuple(float(x) for x in aux.uniform(0.0, 0.03, 4) * (aux.random(4) < 0.6))
    crop, Hr, S0 = rectify_window(scene, win, rng, p.seg_jitter, expand=expand, return_h=True)
    S = S0
    if crop.shape[0] > max_crop_side:
        crop = cv2.resize(crop, (max_crop_side, max_crop_side), interpolation=cv2.INTER_AREA)
        S = max_crop_side
    geo = CropGeometry(U, V, cg["Hm"], cg["H"], cg["hs"], Hr, S0, S, ppi_r, x0_in)
    views = {"full": (0, 0, S, S), "qc": quarter_rect(S, "c")}
    for k in range(4):
        views[f"qd{k}"] = quarter_rect(S, f"d{k}")
    for k in range(n_rand_views):   # random square windows, 15-60 % of the side (zoom augmentation)
        n = int(max(48, round(float(np.exp(aux.uniform(np.log(0.15), np.log(0.6)))) * S)))
        n = min(n, S)
        views[f"r{k}"] = (int(aux.integers(0, S - n + 1)), int(aux.integers(0, S - n + 1)), n, n)
    lines = {k: dict(rect=list(map(int, r)), **view_lines(geo, r)) for k, r in views.items()}
    lab = asdict(p)
    lab.pop("extra", None)
    lab.update(h_count=p.epi, v_count=p.ppi, weave_coarse=COARSE[p.weave], gen_profile=PROFILE,
               crop_native_side=int(S0), crop_side=int(S), ppi_render=ppi_r, expand=expand,
               n_h_full=len(lines["full"]["h"]), n_v_full=len(lines["full"]["v"]), views=lines)
    return crop, lab


def _work_pts(args):
    seed, out_dir, max_side, dr, p_expand = args
    cv2.setNumThreads(1)
    t = time.time()
    crop, lab = generate_pts(seed, max_crop_side=max_side, density_range=dr, p_expand=p_expand)
    fn = f"s{seed:07d}.jpg"
    cv2.imwrite(os.path.join(out_dir, "img", fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lab["file"] = fn
    lab["gen_seconds"] = round(time.time() - t, 2)
    return lab


def write_pts_dataset(out_dir, n, seed0, workers=16, max_side=1600, density_range=(30, 200), p_expand=0.3):
    """Resumable: seeds whose JSON line already exists in labels.jsonl are skipped."""
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    lp = os.path.join(out_dir, "labels.jsonl")
    done = set()
    if os.path.exists(lp):
        for line in open(lp):
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    jobs = [(seed0 + k, out_dir, max_side, density_range, p_expand) for k in range(n) if seed0 + k not in done]
    print(f"{len(jobs)} to generate ({len(done)} done) -> {out_dir}", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool, open(lp, "a") as f:
        for k, lab in enumerate(pool.imap_unordered(_work_pts, jobs, chunksize=2)):
            f.write(json.dumps(lab) + "\n")
            f.flush()
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--p-expand", type=float, default=0.3)
    a = ap.parse_args()
    write_pts_dataset(a.out, a.n, a.seed0, a.workers, a.max_side, p_expand=a.p_expand)

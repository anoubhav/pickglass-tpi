"""camera_sim.py -- smartphone-capture simulator for rectified 1-inch pick-glass crops
(WVAQ 2027; the renderer behind PickGlass-Stress).

    import camera_sim as CS
    out_bgr, P, info = CS.simulate(crop_bgr, "budget_binned_12mp", seed=0)
    # P    = every sampled parameter (JSON-serialisable, enough to re-render bit-exactly)
    # info = sizes, px/inch, LABEL FACTORS, predicted noise, clipped fraction, timings
    out_bgr, info = CS.render(crop_bgr, P, enabled=CS.ALL_STAGES)   # stage toggles for the audit

The input crop (uint8 BGR, S_src x S_src px = S_src px per inch, sRGB) is taken as the scene.
Profiles come from camera_profiles.json (RESEARCH.md section 6).  Every random choice is drawn
from a seeded numpy Generator, in independent sub-streams (camera / overlay / transport / noise),
so e.g. 'budget + messaging' re-uses exactly the camera of 'budget' with the same seed.

PIPELINE (order of Brooks et al. 2019 / Rawgment 2023: optics on linear scene light, then sensor,
then ISP).  Stage names are the keys of ``enabled``; 'unprocess', 'rectify', 'store' always run.
 unprocess  sRGB -> inverse smoothstep -> x^2.2 -> inverse CCM -> / WB gains -> x 0.8 x exposure
 distortion radial k1 about the window centre, rectified with the 4 exact window corners (the
            homography of common/rectify.py cannot undo it) -> LABEL FACTOR != 1 (computed exactly)
 chromatic  lateral CA: R magnified (1+eps), B (1-eps) about the optical axis (G exact; |dR|<=eps)
 vignette   natural cos^4 fall-off, optical axis jittered +-15 % of the side
 defocus    sigma(r) = sqrt((s0*sigma_c + extra*(r/r_corner)^2)^2 + sigma_af^2)  [u = 1e-3 inch]
            (centre defocus + single-lens Petzval field curvature + optional AF miss)
 motion     linear shake kernel, length in u, random direction (loupe magnifies phone rotation)
 sample     lens PSF (sigma px) + 100 %-fill pixel aperture (k x k supersampled box) on a sensor
            grid rotated by theta and shifted by a sub-pixel phase w.r.t. the window, at the
            profile's px/inch.  Where the profile is finer than the source, the scene is bicubically
            up-sampled (no detail is invented; the camera's CFA, noise, ISP and JPEG keep their
            physical scale) -- v2; v1 ran such profiles at the source pitch (CAP_TO_SOURCE).
 cfa        RGGB Bayer mosaic (quad-Bayer + normalised-convolution remosaic for the 50 MP mode)
            and demosaic: bilinear | Malvar-He-Cutler | OpenCV edge-aware
 noise      y ~ N(x, (lambda_read + lambda_shot x) / n_frames)   (Brooks Eq. 2-3; ISO-scaled)
 isp        WB gains (+ AWB error) on the mosaic, digital gain, CCM, gamma 1/2.2, smoothstep tone
            blend, local tone mapping (log-luma base flattening)
 denoise    luma bilateral (sigma_colour = k x predicted output noise) + chroma Gaussian
 zoom       digital zoom = ISP bicubic up-sampling (no new detail)
 sharpen    unsharp mask on luma with threshold (overshoot / halos)
 jpeg       native camera JPEG, quality q, 4:2:0
 transport  none | messaging_standard | messaging_hd | resave_q75 | resave_q75_x0.25
 rectify    exact known corners -> S_out x S_out, bilinear (common/rectify.py convention)
 store      JPEG q92 4:2:0 = annotation_tool/prepare.py's crop file (the benchmark file)

LABEL TRANSFORM (verified in label_audit.py / selftest()): the output crop always covers the same
1-inch window, so every resolution / CFA / noise / ISP / JPEG / transport stage is the IDENTITY
for threads per inch.  The one exception is radial distortion under corner-based rectification:
row y of the output spans x_in(S, y) - x_in(0, y) of fabric, i.e. a factor
g(sqrt(1+y_n^2)) / g(sqrt 2) (g = 1 + k1 r^2), exact at any yarn tilt (the y-displacements of the
two row ends are equal).  info['label_factor_h'] / ['label_factor_v'] = area-mean factor (what a
global estimator measures), info['label_factor_h_centre'] / ['_v_centre'] = on the centre guide
lines (what the annotation tool's guide-line count measures); new label = old label x factor.
Lateral CA moves R/B counts by <= eps (info['label_bound_ca']).  Observability is NOT guaranteed:
below ~2.4 px per yarn (ours' 0.42 cycles/px search limit; luma Nyquist 2 px, Bayer R/B Nyquist
4 px) the count can alias -- label_audit.py measures where.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC_PATH = os.path.join(HERE, "camera_profiles.json")
SPEC = json.load(open(SPEC_PATH))
PROFILES = SPEC["profiles"]
PROFILE_NAMES = list(PROFILES)
TRANSPORTS = ("none", "messaging_standard", "messaging_hd", "resave_q75", "resave_q75_x0.25")
OVERLAYS = (None, "low_light")

STAGES = ("distortion", "chromatic", "vignette", "defocus", "motion", "sample", "cfa", "noise", "isp",
          "denoise", "zoom", "sharpen", "jpeg", "transport")
ALL_STAGES = frozenset(STAGES)
ALWAYS = ("unprocess", "rectify", "store")
STORE_Q = 92                       # annotation_tool/prepare.py writes the rectified crops at q92
SIM_VERSION = "camera_sim 2026-10-07 v2"
# v1   : profiles finer than the source were simulated at the source pitch ("capped"), with binned noise and
#        scaled ISP radii -- but the CFA / quad-Bayer remosaic / JPEG blocks were then physically too coarse
#        (e.g. a 50 MP sensor's Bayer pattern simulated at 490 px/inch), exaggerating CFA artefacts.
# v1.1 : transport target fixed for capped pitches.
# v2   : the sensor and ISP always run at the profile's own px/inch; where that is finer than the source,
#        the scene is bicubically up-sampled (no detail is invented, the camera's artefacts get their
#        physical scale).  CAP_TO_SOURCE = True restores v1.1.
CAP_TO_SOURCE = False

# Brooks 2019 constants
BROOKS_HEADROOM = 0.8              # mean of Brooks' inverse digital gain N(0.8, 0.1)
LAMBDA_SHOT_MAX = 1.2e-2           # Brooks' upper bound of lambda_shot
# camera RGB -> linear sRGB colour-correction matrix of a generic phone sensor (rows sum to 1, so
# grey is preserved; typical magnitude of published phone CCMs).  ASSUMPTION: one fixed matrix.
CAM2RGB = np.array([[1.65, -0.45, -0.20],
                    [-0.30, 1.55, -0.25],
                    [0.05, -0.55, 1.50]], np.float64)
RGB2CAM = np.linalg.inv(CAM2RGB)
WINDOW_ROTATION_DEG = 5.0          # ASSUMPTION: sensor grid vs window, U(-5, 5) deg
# Radial distortion RELATIVE to the source crop.  The spec range (global.optics_loupe.
# radial_distortion_k1 = [-0.08, 0.08]) means up to 16 % distortion at the window corners and a
# +-5 % area-mean count change.  measure_distortion.py measured the real phone + pick glass of our
# 150 photos on the frame edges after corner rectification: k1 = -0.028 (IQR -0.036 .. -0.021, clean
# sides), i.e. mild pincushion that is already baked into every source crop.  Default: another phone /
# loupe differs from ours by at most +-0.03 (calibrated, ASSUMPTION on the spread); k1_range='spec'
# restores the spec range.
K1_RANGES = {"calibrated": (-0.03, 0.03), "spec": tuple(SPEC["global"]["optics_loupe"]["radial_distortion_k1"]),
             "none": (0.0, 0.0)}
UNIT_INCH = 1e-3                   # 'u' of methods/degrade.py


# =============================================================================================
# small helpers
# =============================================================================================
def stable_seed(*parts) -> int:
    """Process-independent 31-bit seed from any parts (str() of each)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _u(r, lo_hi):
    lo, hi = float(lo_hi[0]), float(lo_hi[-1])
    return float(r.uniform(lo, hi)) if hi > lo else lo


def _logu(r, lo_hi):
    lo, hi = float(lo_hi[0]), float(lo_hi[-1])
    return float(math.exp(r.uniform(math.log(lo), math.log(hi)))) if hi > lo else lo


def _choice(r, d: dict):
    keys = [k for k in d if k != "note"]
    p = np.array([float(d[k]) for k in keys])
    return str(keys[int(r.choice(len(keys), p=p / p.sum()))])


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def inv_smoothstep(y):
    y = np.clip(y, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0)


def jpeg_bytes(img_u8_bgr, q, sampling="420", progressive=False) -> bytes:
    fl = [cv2.IMWRITE_JPEG_QUALITY, int(q)]
    if sampling == "420":
        fl += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
    elif sampling == "444":
        fl += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if progressive:
        fl += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    ok, buf = cv2.imencode(".jpg", img_u8_bgr, fl)
    if not ok:
        raise IOError("JPEG encode failed")
    return buf.tobytes()


def jpeg_roundtrip(img_u8_bgr, q, sampling="420", progressive=False):
    b = jpeg_bytes(img_u8_bgr, q, sampling, progressive)
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), b


def to_u8(x01):
    return np.clip(x01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def resize(img, size_wh, method):
    """Resampling used by the transport stage.  'area' = cv2 INTER_AREA (box, anti-aliased),
    'bilinear' = cv2 INTER_LINEAR (NO anti-aliasing on down-scaling: aliases), 'lanczos' = cv2
    INTER_LANCZOS4 (fixed 8-tap kernel, NOT anti-aliased for large factors), 'pil_bicubic' =
    Pillow BICUBIC (support scaled with the factor = anti-aliased; Pillow wrote our q75 files)."""
    W, H = int(size_wh[0]), int(size_wh[1])
    if method == "area":
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if method == "bilinear":
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if method == "lanczos":
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if method == "pil_bicubic":
        from PIL import Image
        im = Image.fromarray(img[..., ::-1].copy())
        return np.asarray(im.resize((W, H), Image.BICUBIC))[..., ::-1].copy()
    raise ValueError(method)


# =============================================================================================
# parameter sampling
# =============================================================================================
def sample_params(profile: str, seed: int, overlay: str | None = None, transport: str = "none",
                  scene_effects: bool = False, k1_range: str = "calibrated") -> dict:
    """Draw every parameter of one capture from the profile's ranges (camera_profiles.json).

    Independent sub-streams [seed, j, crc(profile)]: j = 1 camera, 2 low-light overlay, 3 transport,
    4 scene effects (LED torch; off in the benchmark), 5 -> noise seed, 6 'auto' transport choice.
    The same seed therefore gives independent draws for different profiles, and identical camera
    draws for the same profile with / without an overlay or a transport.
    """
    if profile not in PROFILES:
        raise KeyError(profile)
    if transport == "auto":
        rt = np.random.default_rng([int(seed), 6, stable_seed(profile)])
        tr = {k: v["p"] for k, v in SPEC["transport"].items()}
        t = _choice(rt, tr)
        if t == "pipeline_resave_q75":   # spec resize_factor [1.0, 1.0, 0.25]: x0.25 with p = 1/3
            t = "resave_q75_x0.25" if rt.random() < 1 / 3 else "resave_q75"
        transport = t
    if transport not in TRANSPORTS:
        raise ValueError(transport)
    pr = PROFILES[profile]
    G = SPEC["global"]
    r = np.random.default_rng([int(seed), 1, stable_seed(profile)])
    P = dict(sim_version=SIM_VERSION, profile=profile, seed=int(seed), overlay=overlay, transport=transport)
    # ---- output geometry ----------------------------------------------------------------
    P["output_long_side_px"] = int(round(_u(r, pr["output_long_side_px"])))
    P["ppi_profile"] = _u(r, pr["ppi"])                      # window px/inch on the camera output
    dz = G["digital_zoom"]
    P["zoom"] = _u(r, dz["range"]) if r.random() < dz["p_zoomed"] else 1.0
    P["theta_deg"] = float(r.uniform(-WINDOW_ROTATION_DEG, WINDOW_ROTATION_DEG))
    P["phase"] = [float(r.uniform(0, 1)), float(r.uniform(0, 1))]
    P["lens_sigma_px"] = _u(r, pr["lens_psf_sigma_px"])
    # ---- loupe optics ------------------------------------------------------------------------
    op = G["optics_loupe"]
    P["k1_range"] = k1_range
    P["k1"] = _u(r, K1_RANGES[k1_range]) if k1_range != "none" else (r.uniform(0, 1) * 0.0)
    P["ca_eps"] = _u(r, op["lateral_chromatic_aberration_eps"])
    P["ca_sign"] = int(r.choice([-1, 1]))
    # spec range is a multiplier on the natural cos^4 fall-off at the window corner (21 deg field
    # angle for a 41.6 % window on a 26 mm-eq. lens -> 25 % fall-off): corner loss 5-25 %
    P["vignette_corner_loss"] = 0.25 * _u(r, op["vignetting_cos4_strength"])
    P["vignette_centre"] = [float(0.5 + r.uniform(-0.15, 0.15)), float(0.5 + r.uniform(-0.15, 0.15))]
    P["defocus_centre_u"] = _u(r, pr["defocus_center_sigma_u"])
    P["defocus_sign"] = int(r.choice([-1, 1]))
    P["defocus_corner_extra_u"] = _u(r, pr["defocus_corner_extra_sigma_u"])
    af = pr["af_miss"]
    P["af_miss"] = bool(r.random() < af["p"])
    s_af = _u(r, af["sigma_u"])
    P["af_sigma_u"] = s_af if P["af_miss"] else 0.0
    ois = pr.get("ois", False)
    P["ois"] = bool(r.random() < 0.5) if isinstance(ois, str) else bool(ois)   # 'mixed' -> coin flip
    P["motion_u"] = _u(r, pr["motion_blur_len_u"])
    P["motion_angle_deg"] = float(r.uniform(0, 180))
    # ---- sensor ---------------------------------------------------------------------------------
    P["iso"] = _logu(r, pr["iso"])
    P["lambda_shot_iso100"] = _logu(r, pr["lambda_shot_at_iso100"])
    z_read = float(r.normal(0.0, 0.26))
    fr = pr["multi_frame_merge_frames"]
    P["frames"] = int(r.integers(int(fr[0]), int(fr[-1]) + 1))
    dem = _choice(r, pr["demosaic"])
    P["cfa"] = "quad" if dem.startswith("quad") else "bayer"
    P["demosaic"] = {"quad_bayer_remosaic_then_bilinear": "bilinear",
                     "quad_bayer_remosaic_then_malvar": "malvar_he_cutler"}.get(dem, dem)
    # ---- ISP --------------------------------------------------------------------------------------
    wb = pr["white_balance_gains"]
    P["wb_r"], P["wb_b"] = _u(r, wb["r"]), _u(r, wb["b"])
    P["awb_err_r"] = float(r.normal(0.0, wb["awb_error_rel_sd"]))
    P["awb_err_b"] = float(r.normal(0.0, wb["awb_error_rel_sd"]))
    tn = pr["tone"]
    P["exposure"] = _u(r, tn["exposure_gain"])
    P["tone_blend"] = _u(r, tn["smoothstep_blend"])
    P["ltm"] = _u(r, tn["local_tone_map_strength"])
    dn = pr["denoise"]
    P["denoise_k"] = _u(r, dn["luma_strength_x_noise_sigma"])
    P["denoise_space_px"] = float(r.uniform(1.0, 2.0))
    P["chroma_sigma_px"] = _u(r, dn["chroma_blur_sigma_px"])
    sh = pr["sharpen_usm"]
    P["sharpen_sigma_px"] = _u(r, sh["radius_sigma_px"])
    P["sharpen_amount"] = _u(r, sh["amount"])
    P["sharpen_threshold_dn"] = _u(r, sh["threshold_dn"])
    jq = pr["jpeg_native"]["q"]
    P["jpeg_q"] = int(r.integers(int(jq[0]), int(jq[-1]) + 1))
    # ---- low-light overlay -------------------------------------------------------------------
    if overlay == "low_light":
        ro = np.random.default_rng([int(seed), 2, stable_seed(profile)])
        ll = SPEC["overlays"]["low_light"]
        P["iso"] = _logu(ro, ll["iso"])
        P["frames"] = int(ro.integers(int(ll["night_mode_merge_frames"][0]),
                                      int(ll["night_mode_merge_frames"][1]) + 1))
        m = _u(ro, ll["motion_blur_len_u_no_ois"])
        P["motion_u"] = m * (0.3 if P["ois"] else 1.0)     # ASSUMPTION: OIS removes ~70 % of shake
        P["denoise_k"] = float(dn["luma_strength_x_noise_sigma"][-1])
        P["chroma_sigma_px"] = float(dn["chroma_blur_sigma_px"][-1])
    elif overlay is not None:
        raise ValueError(overlay)
    P["lambda_shot"] = float(min(P["lambda_shot_iso100"] * P["iso"] / 100.0, LAMBDA_SHOT_MAX))
    P["lambda_read"] = float(math.exp(2.18 * math.log(P["lambda_shot"]) + 1.20 + z_read))
    # ---- transport ---------------------------------------------------------------------------------
    rt = np.random.default_rng([int(seed), 3, stable_seed(profile)])
    T = SPEC["transport"]
    P["transport_q"], P["transport_progressive"], P["transport_filter"] = None, False, None
    P["transport_long_side_px"], P["transport_factor"] = None, 1.0
    if transport == "messaging_standard":
        t = T["messaging_standard"]
        P["transport_long_side_px"] = int(round(_u(rt, t["resize_long_side_px"])))
        P["transport_filter"] = str(rt.choice(t["resize_filter"]))
        P["transport_q"] = int(rt.integers(t["jpeg_q"][0], t["jpeg_q"][1] + 1))
        P["transport_progressive"] = True
    elif transport == "messaging_hd":
        t = T["messaging_hd"]
        P["transport_long_side_px"] = int(round(_u(rt, t["resize_long_side_px"])))
        P["transport_filter"] = str(rt.choice(["area", "bilinear", "lanczos"]))
        P["transport_q"] = int(rt.integers(t["jpeg_q"][0], t["jpeg_q"][1] + 1))
        P["transport_progressive"] = True
    elif transport in ("resave_q75", "resave_q75_x0.25"):
        P["transport_q"] = 75
        if transport.endswith("x0.25"):
            P["transport_factor"] = 0.25
            # the resampler of our real quarter-resolution copies is unknown (ASSUMPTION)
            P["transport_filter"] = str(rt.choice(["pil_bicubic", "area"]))
    if P["transport_long_side_px"]:
        P["transport_factor"] = float(min(1.0, P["transport_long_side_px"] / P["output_long_side_px"]))
    # ---- scene effects (training mode only) ----------------------------------------------------
    rs = np.random.default_rng([int(seed), 4, stable_seed(profile)])
    P["led"] = bool(scene_effects and rs.random() < (0.6 if overlay == "low_light" else pr["led_torch_p"]))
    P["led_amp"] = float(rs.uniform(0.2, 1.2))
    P["led_r0"] = float(rs.uniform(0.08, 0.2))
    P["led_centre"] = [float(rs.uniform(0.25, 0.75)), float(rs.uniform(0.25, 0.75))]
    if P["led"]:  # torch on -> more light -> lower ISO (ASSUMPTION: x 1/4)
        P["iso"] = max(float(pr["iso"][0]), P["iso"] / 4.0)
        P["lambda_shot"] = float(min(P["lambda_shot_iso100"] * P["iso"] / 100.0, LAMBDA_SHOT_MAX))
        P["lambda_read"] = float(math.exp(2.18 * math.log(P["lambda_shot"]) + 1.20 + z_read))
    P["noise_seed"] = int(np.random.default_rng([int(seed), 5, stable_seed(profile)]).integers(0, 2 ** 31 - 1))
    return P


# =============================================================================================
# optics (linear scene, source grid)
# =============================================================================================
def distortion_maps(S, k1):
    """Inverse map of the corner-rectified radial distortion (optical axis = window centre) and the
    exact label factors.  Continuous coords: pixel i covers [i, i+1); the window is [0, S]."""
    c = S / 2.0
    R = S / 2.0
    G = 1.0 + k1 * 2.0                                    # g(r_corner = sqrt 2): corners stay fixed
    xs = np.arange(S, dtype=np.float32) + 0.5
    dx = (xs - c)[None, :]
    dy = (xs - c)[:, None]
    g = 1.0 + np.float32(k1 / R ** 2) * (dx * dx + dy * dy)
    mx = (c + dx * g / np.float32(G) - 0.5).astype(np.float32)   # -> OpenCV index coords
    my = (c + dy * g / np.float32(G) - 0.5).astype(np.float32)
    xs = xs.astype(np.float64)
    # label factors: row y spans x_in(S, y) - x_in(0, y) = S * g(r_end) / G  (both ends same r)
    yn = (xs - c) / R
    f_rows = (1.0 + k1 * (1.0 + yn ** 2)) / G
    lab = dict(label_factor_h=float(f_rows.mean()), label_factor_v=float(f_rows.mean()),
               label_factor_h_centre=float((1.0 + k1) / G), label_factor_v_centre=float((1.0 + k1) / G),
               label_factor_h_min=float(f_rows.min()), label_factor_h_max=float(f_rows.max()))
    return mx, my, lab


def apply_distortion_ca(x, k1, eps, sign, do_dist=True, do_ca=True):
    """x: linear RGB float32 (S x S x 3).  Radial distortion (all channels) + lateral CA
    (R x (1 + s eps), B x (1 - s eps) magnification relative to G, about the window centre)."""
    S = x.shape[0]
    out = np.empty_like(x)
    if do_dist:
        mx, my, lab = distortion_maps(S, k1)
    else:
        xs = np.arange(S, dtype=np.float32)
        mx, my = np.meshgrid(xs, xs)
        lab = dict(label_factor_h=1.0, label_factor_v=1.0, label_factor_h_centre=1.0,
                   label_factor_v_centre=1.0, label_factor_h_min=1.0, label_factor_h_max=1.0)
    c = (S - 1) / 2.0
    for ch, m in ((0, 1.0 + sign * eps), (1, 1.0), (2, 1.0 - sign * eps)):
        if not do_ca:
            m = 1.0
        if m == 1.0 and not do_dist:
            out[..., ch] = x[..., ch]
            continue
        # a channel magnified by m samples the scene at (p - c) / m
        cx = (c + (mx - c) / m).astype(np.float32)
        cy = (c + (my - c) / m).astype(np.float32)
        out[..., ch] = cv2.remap(x[..., ch], cx, cy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0, None), lab


def apply_vignette(x, corner_loss, centre):
    S = x.shape[0]
    cx, cy = centre[0] * S, centre[1] * S
    xs = np.arange(S, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, xs)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    rc2 = max((cx - a) ** 2 + (cy - b) ** 2 for a in (0, S) for b in (0, S))
    s = float(np.clip(corner_loss, 0.0, 0.95))
    a = ((1 - s) ** -0.5 - 1.0) / rc2 if s > 0 else 0.0
    return x * ((1.0 + a * r2) ** -2)[..., None].astype(np.float32)


def defocus_sigma_map_u(S, P):
    xs = (np.arange(S, dtype=np.float32) + 0.5) / S - 0.5
    X, Y = np.meshgrid(xs, xs)
    rr = (X * X + Y * Y) / 0.5                            # (r / r_corner)^2, 1 at the corners
    s = P["defocus_sign"] * P["defocus_centre_u"] + P["defocus_corner_extra_u"] * rr
    return np.sqrt(s * s + P["af_sigma_u"] ** 2)


def apply_defocus(x, sig_px, n_levels=7):
    """Spatially varying Gaussian blur by linear interpolation in a blur stack."""
    smax = float(sig_px.max())
    if smax <= 0.05:
        return x
    lev = np.linspace(0.0, smax, n_levels)
    pos = np.clip(sig_px / (lev[1] - lev[0]), 0, n_levels - 1).astype(np.float32)
    out = np.zeros_like(x)
    for k in range(n_levels):                             # piecewise-linear interpolation weights,
        w = np.maximum(0.0, 1.0 - np.abs(pos - k))         # one blur level in memory at a time
        if w.max() > 0:
            b = x if lev[k] <= 0.05 else cv2.GaussianBlur(x, (0, 0), float(lev[k]))
            out += b * w[..., None]
            del b
    return out


def motion_kernel(L_px, angle_deg):
    """Bilinear splat of a centred line segment (same construction as methods/degrade.py)."""
    L = max(1.0, float(L_px))
    n = int(math.ceil(L)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    t = np.linspace(-L / 2, L / 2, max(2, int(4 * L)))
    xs = c + t * math.cos(math.radians(angle_deg))
    ys = c - t * math.sin(math.radians(angle_deg))
    for x, y in zip(xs, ys):
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for ddx, ddy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy),
                            (1, 1, fx * fy)):
            if 0 <= x0 + ddx < n and 0 <= y0 + ddy < n:
                k[y0 + ddy, x0 + ddx] += w
    return k / k.sum()


def apply_led(x, P):
    S = x.shape[0]
    xs = np.arange(S, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, xs)
    cx, cy = P["led_centre"][0] * S, P["led_centre"][1] * S
    lobe = P["led_amp"] * (1.0 + ((X - cx) ** 2 + (Y - cy) ** 2) / (P["led_r0"] * S) ** 2) ** -1.5
    m = x.mean((0, 1), keepdims=True)
    v = np.clip(0.15 * lobe, 0, 0.9)[..., None]
    return x * (1 - v) + m * v + lobe[..., None] * np.array([0.95, 1.0, 1.04], np.float32)


# =============================================================================================
# sensor
# =============================================================================================
def cfa_masks(H, W, kind="bayer"):
    """(mR, mG, mB) float32 masks.  'bayer' = RGGB; 'quad' = 2x2 same-colour cells in a 4x4 RGGB
    super-cell (Tetra / Quad-Bayer)."""
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == "quad":
        yy, xx = yy // 2, xx // 2
    r = ((yy % 2) == 0) & ((xx % 2) == 0)
    b = ((yy % 2) == 1) & ((xx % 2) == 1)
    g = ~(r | b)
    return r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)


def mosaic(rgb, kind="bayer"):
    mR, mG, mB = cfa_masks(*rgb.shape[:2], kind)
    return rgb[..., 0] * mR + rgb[..., 1] * mG + rgb[..., 2] * mB


def _f2(x, k):
    return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101)


def demosaic_bilinear(m):
    mR, mG, mB = cfa_masks(*m.shape, "bayer")
    kG = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0
    kRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
    return np.stack([_f2(m * mR, kRB), _f2(m * mG, kG), _f2(m * mB, kRB)], -1)


_MHC = None


def _mhc_kernels():
    global _MHC
    if _MHC is None:
        g_at_rb = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0],
                            [0, 0, -1, 0, 0]], np.float32) / 8
        rb_at_g_row = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0],
                                [0, 0, .5, 0, 0]], np.float32) / 8     # R at G in an R row (B col)
        rb_at_g_col = rb_at_g_row.T.copy()                               # R at G in a B row (R col)
        rb_at_br = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0],
                             [0, 0, -1.5, 0, 0]], np.float32) / 8
        _MHC = (g_at_rb, rb_at_g_row, rb_at_g_col, rb_at_br)
    return _MHC


def demosaic_mhc(m):
    """Malvar-He-Cutler 2004 linear demosaic of an RGGB mosaic (5x5 gradient-corrected filters)."""
    H, W = m.shape
    g_at_rb, k_row, k_col, k_diag = _mhc_kernels()
    yy, xx = np.mgrid[0:H, 0:W]
    ev_r, ev_c = (yy % 2) == 0, (xx % 2) == 0
    isR, isB = ev_r & ev_c, ~ev_r & ~ev_c
    isG_Rrow, isG_Brow = ev_r & ~ev_c, ~ev_r & ev_c
    G_rb, A_row, A_col, A_diag = _f2(m, g_at_rb), _f2(m, k_row), _f2(m, k_col), _f2(m, k_diag)
    R = np.where(isR, m, np.where(isG_Rrow, A_row, np.where(isG_Brow, A_col, A_diag)))
    B = np.where(isB, m, np.where(isG_Brow, A_row, np.where(isG_Rrow, A_col, A_diag)))
    Gc = np.where(isR | isB, G_rb, m)
    return np.stack([R, Gc, B], -1).astype(np.float32)


_EA_CODE = None


def demosaic_ea(m):
    """OpenCV edge-aware (VNG-like) demosaic on a 16-bit mosaic; pattern code auto-detected."""
    global _EA_CODE
    if _EA_CODE is None:
        test = np.zeros((16, 16, 3), np.float32)
        test[..., 0] = 1.0                                   # pure red scene (RGB order)
        mm = (mosaic(test) * 65535).astype(np.uint16)
        for name in ("COLOR_BayerRG2BGR_EA", "COLOR_BayerBG2BGR_EA", "COLOR_BayerGR2BGR_EA",
                     "COLOR_BayerGB2BGR_EA"):
            o = cv2.cvtColor(mm, getattr(cv2, name)).astype(np.float32) / 65535
            if o[4:12, 4:12, 2].mean() > 0.8 and o[4:12, 4:12, 0].mean() < 0.2:
                _EA_CODE = getattr(cv2, name)
                break
        assert _EA_CODE is not None
    u16 = (np.clip(m, 0, 1) * 65535 + 0.5).astype(np.uint16)
    bgr = cv2.cvtColor(u16, _EA_CODE).astype(np.float32) / 65535
    return bgr[..., ::-1].copy()


def remosaic_quad(m):
    """Quad-Bayer -> Bayer remosaic by per-channel normalised convolution (a simple stand-in for
    the ISP's directional remosaic; produces the soft 'colour mesh' of full-resolution modes)."""
    H, W = m.shape
    k = cv2.getGaussianKernel(7, 1.4)
    k = (k @ k.T).astype(np.float32)
    full = []
    for mk in cfa_masks(H, W, "quad"):
        full.append(_f2(m * mk, k) / np.maximum(_f2(mk, k), 1e-6))
    return mosaic(np.stack(full, -1), "bayer")


def demosaic(m, method, cfa="bayer"):
    if cfa == "quad":
        m = remosaic_quad(m)
    if method == "bilinear":
        return demosaic_bilinear(m)
    if method == "malvar_he_cutler":
        return demosaic_mhc(m)
    if method == "edge_aware":
        return demosaic_ea(m)
    raise ValueError(method)


# =============================================================================================
# ISP
# =============================================================================================
_UNPROC_LUT = np.power(inv_smoothstep(np.arange(256) / 255.0), 2.2).astype(np.float32)


def tone_forward(lin, blend):
    g = cv2.pow(np.clip(lin, 1e-8, 1.0).astype(np.float32), 1 / 2.2)
    return np.float32(1 - blend) * g + np.float32(blend) * smoothstep(g)


def tone_slope(v, blend, h=1e-3):
    v = float(np.clip(v, 2e-3, 1 - 2e-3))
    a = tone_forward(np.array([v - h]), blend)[0]
    b = tone_forward(np.array([v + h]), blend)[0]
    return (b - a) / (2 * h)


def local_tone_map(rgb, strength):
    if strength <= 0:
        return rgb
    S = rgb.shape[0]
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lY = np.log(np.clip(Y, 1e-4, None)).astype(np.float32)
    f = max(1, S // 128)                                   # large-scale base layer on a decimated grid
    small = cv2.resize(lY, (max(4, S // f), max(4, S // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(2.0, S / 8.0) / f)
    base = cv2.resize(small, (S, S), interpolation=cv2.INTER_LINEAR)
    gain = np.exp(-strength * (base - float(base.mean())))
    return rgb * gain[..., None].astype(np.float32)


def denoise_ycc(rgb01, sigma_dn, k, space_px, chroma_sigma):
    """rgb01 float RGB in [0,1] -> bilateral on luma (sigma_colour = k * sigma_dn grey levels) and
    Gaussian chroma blur, in YCrCb."""
    ycc = cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    sc = float(k * sigma_dn) / 255.0
    if sc > 1e-4:
        ycc[..., 0] = cv2.bilateralFilter(ycc[..., 0], -1, sc, float(space_px))
    if chroma_sigma > 0.05:
        for c in (1, 2):
            ycc[..., c] = cv2.GaussianBlur(ycc[..., c], (0, 0), float(chroma_sigma))
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def sharpen_ycc(rgb01, sigma, amount, thr_dn):
    ycc = cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    Y = ycc[..., 0]
    d = Y - cv2.GaussianBlur(Y, (0, 0), float(sigma))
    if thr_dn > 0:
        d = np.where(np.abs(d) * 255.0 > thr_dn, d, 0.0)
    ycc[..., 0] = Y + float(amount) * d
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


# =============================================================================================
# geometry of the sensor patch
# =============================================================================================
def _affine_cont_to_index(A, b):
    """x_src_cont = A x_dst_cont + b  ->  2x3 matrix on OpenCV index coords (centre = integer)."""
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    bb = A @ np.array([0.5, 0.5]) + b - 0.5
    return np.hstack([A, bb[:, None]])


def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


# =============================================================================================
# main renderer
# =============================================================================================
def render(src_bgr, P: dict, enabled=ALL_STAGES, src_is_linear_rgb=None, scene_degrade=None,
           return_bytes=False, timings=False):
    """Render one capture of the source crop with the parameters P (see sample_params).

    enabled          : iterable of stage names (STAGES); 'unprocess', 'rectify', 'store' always run.
    src_is_linear_rgb: optional float32 linear RGB scene (S x S x 3, [0, 1]) instead of src_bgr
                       (synthetic fabrics: no unprocessing needed).
    scene_degrade    : optional (kind, level, seed) from methods/degrade.py applied to the sRGB
                       scene before capture (glare / gblur combos of the stress benchmark).
    Returns (out_bgr uint8 S_out x S_out, info) [+ store-JPEG bytes if return_bytes].
    """
    en = set(enabled)
    bad = en - set(STAGES)
    if bad:
        raise ValueError(f"unknown stages {bad}")
    t0 = time.time()
    tt = {}

    def tick(name):
        if timings:
            tt[name] = round(time.time() - t0, 3)

    info = dict(stages=sorted(en))
    # ---------------------------------------------------------------- 0. unprocess -----------------
    isp_on = "isp" in en
    expo = P["exposure"] if isp_on else 1.0
    gains = np.array([P["wb_r"], 1.0, P["wb_b"]], np.float64)
    if src_is_linear_rgb is not None:
        lin = np.clip(src_is_linear_rgb.astype(np.float32), 0, 1)
        S = lin.shape[0]
    else:
        img = src_bgr
        if scene_degrade is not None:
            sys.path.insert(0, os.path.join(ROOT, "methods"))
            import degrade as DG  # noqa: E402
            kind, level, dseed = scene_degrade
            img = DG.apply(img, kind, level, seed=dseed)
        if img.shape[0] != img.shape[1]:
            s = max(img.shape[:2])
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        S = img.shape[0]
        lin = _UNPROC_LUT[img[..., ::-1]]                            # RGB; Brooks inverse tone + gamma
    info["S_src"] = int(S)
    tick("unprocess")
    # ---------------------------------------------------------------- 1. optics (scene, linear sRGB primaries)
    lab = dict(label_factor_h=1.0, label_factor_v=1.0, label_factor_h_centre=1.0, label_factor_v_centre=1.0,
               label_factor_h_min=1.0, label_factor_h_max=1.0)
    if "distortion" in en or "chromatic" in en:
        lin, lab = apply_distortion_ca(lin, P["k1"], P["ca_eps"], P["ca_sign"], "distortion" in en,
                                       "chromatic" in en)
    info.update(lab)
    info["label_bound_ca"] = float(P["ca_eps"]) if "chromatic" in en else 0.0
    if "vignette" in en:
        lin = apply_vignette(lin, P["vignette_corner_loss"], P["vignette_centre"])
    px_per_u = S * UNIT_INCH
    if "defocus" in en:
        sig = defocus_sigma_map_u(S, P) * px_per_u
        sc0 = P["defocus_sign"] * P["defocus_centre_u"]
        info["defocus_sigma_u_centre_corner"] = [math.hypot(sc0, P["af_sigma_u"]),
                                                 math.hypot(sc0 + P["defocus_corner_extra_u"], P["af_sigma_u"])]
        lin = apply_defocus(lin, sig.astype(np.float32))
    if "motion" in en and P["motion_u"] * px_per_u > 1.0:
        lin = cv2.filter2D(lin, -1, motion_kernel(P["motion_u"] * px_per_u, P["motion_angle_deg"]),
                           borderType=cv2.BORDER_REFLECT_101)
    if P.get("led"):
        lin = apply_led(lin, P)
    tick("optics")
    # camera space: inverse CCM, inverse WB, headroom x exposure
    cam = cv2.transform(lin, RGB2CAM.astype(np.float32))
    cam = cam * (np.float32(BROOKS_HEADROOM * expo) / gains.astype(np.float32))
    cam = np.clip(cam, 0.0, None)
    # ---------------------------------------------------------------- 2. sample onto the sensor grid
    ppi_target = float(P["ppi_profile"])
    if "sample" in en:
        ppi_s = min(ppi_target, float(S)) if CAP_TO_SOURCE else ppi_target
        th = math.radians(P["theta_deg"])
        phase = P["phase"]
        lens = P["lens_sigma_px"]                             # scaled by psc below
    else:                                                      # isolated stages run at the source pitch
        ppi_s, th, phase, lens = float(S), 0.0, [0.0, 0.0], 0.0
    info["ppi_profile"] = ppi_target
    info["ppi_sensor"] = float(ppi_s)
    info["ppi_capped"] = bool("sample" in en and CAP_TO_SOURCE and ppi_target > S)
    info["scene_upsampled"] = bool("sample" in en and ppi_s > S)
    # noise is specified per camera output pixel: on a coarser (capped / source) pitch its variance is
    # binned (n_bin > 1), on a finer pitch it grows (n_bin < 1) -- same noise per unit fabric area
    n_bin = (ppi_target / ppi_s) ** 2
    # spatial ISP / optics parameters are defined in the camera's output px: rescaled to the pitch used
    psc = ppi_s / ppi_target
    info["pitch_scale"] = float(psc)
    scale = S / ppi_s                                          # source px per sensor px
    if "sample" in en:
        span = ppi_s * (abs(math.cos(th)) + abs(math.sin(th)))
        M = int(math.ceil(span)) + 6
        M += M % 2
        k = int(min(6, max(1, math.ceil(scale - 1e-9))))
        cs = np.array([M / 2.0 + phase[0], M / 2.0 + phase[1]])
        # sub-grid t (k per sensor px) -> sensor s = t / k -> window w -> source x = S w
        # x = S (0.5 + R(-th) (s - cs) / ppi_s)
        A = (S / ppi_s) * _rot(-th) / k
        b = S * 0.5 - (S / ppi_s) * _rot(-th) @ cs
        Mi = M * k
        interp = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_CUBIC
        Mx = _affine_cont_to_index(A, b)
        sens = np.empty((M, M, 3), np.float32)
        for ch in range(3):                                     # per channel: bounded memory
            pre = cam[..., ch]
            if scale / k > 1.0:                                 # (k capped) avoid aliasing in the sub-grid
                pre = cv2.GaussianBlur(pre, (0, 0), 0.5 * scale / k)
            sub = cv2.warpAffine(pre, Mx, (Mi, Mi), flags=interp | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if lens * psc > 0.05:
                sub = cv2.GaussianBlur(sub, (0, 0), lens * psc * k)
            sens[..., ch] = cv2.resize(sub, (M, M), interpolation=cv2.INTER_AREA) if k > 1 else sub
            del sub
        sens = np.clip(sens, 0, None)
    else:
        M, k = S, 1
        cs = np.array([S / 2.0, S / 2.0])
        sens = cam
    info["M"], info["k_sub"] = int(M), int(k)
    tick("sample")
    clip_frac = float((sens >= 1.0).any(-1).mean())
    sens = np.minimum(sens, 1.0)                               # sensor saturation
    info["clipped_frac"] = clip_frac
    # ---------------------------------------------------------------- 3-5. CFA, noise, demosaic
    rng = np.random.default_rng(P["noise_seed"])
    lam_s = P["lambda_shot"] / n_bin / max(1, P["frames"])
    lam_r = P["lambda_read"] / n_bin / max(1, P["frames"])
    info["lambda_shot_eff"], info["lambda_read_eff"] = float(lam_s), float(lam_r)
    dg = 1.0 / (BROOKS_HEADROOM * expo)
    awb = np.array([1.0 + P["awb_err_r"], 1.0, 1.0 + P["awb_err_b"]]) if isp_on else np.ones(3)
    wbg = (gains * awb).astype(np.float32)
    if "cfa" in en:
        m = mosaic(sens, P["cfa"])
        if "noise" in en:
            m = m + rng.standard_normal(m.shape).astype(np.float32) * np.sqrt(lam_r + lam_s * np.clip(m, 0, 1))
        m = np.clip(m, 0.0, 1.0)
        mR, mG, mB = cfa_masks(M, M, P["cfa"])
        m = m * (mR * wbg[0] + mG * wbg[1] + mB * wbg[2])      # WB on the mosaic (ISP order)
        rawG = float(np.median(sens[..., 1]))
        camw = demosaic(m, P["demosaic"], P["cfa"])
    else:
        x = sens
        if "noise" in en:
            x = x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(lam_r + lam_s * np.clip(x, 0, 1))
        x = np.clip(x, 0.0, 1.0)
        rawG = float(np.median(sens[..., 1]))
        camw = x * wbg
    tick("cfa_noise")
    # ---------------------------------------------------------------- 6. ISP colour + tone
    rgb = cv2.transform(camw.astype(np.float32), CAM2RGB.astype(np.float32)) * np.float32(dg)
    rgb = np.clip(rgb, 0.0, 1.0)
    blend = P["tone_blend"] if isp_on else 1.0
    if isp_on and P["ltm"] > 0:
        rgb = np.clip(local_tone_map(rgb, P["ltm"]), 0.0, 1.0)
    out = tone_forward(rgb, blend).astype(np.float32)
    # predicted output noise (grey levels) at the median G level: what the ISP's noise profile knows
    v = float(np.clip(rawG * dg, 0, 1))
    sig_raw = math.sqrt(P["lambda_read"] / n_bin / max(1, P["frames"])
                        + P["lambda_shot"] / n_bin / max(1, P["frames"]) * rawG)
    sigma_dn = 255.0 * tone_slope(v, blend) * dg * sig_raw
    info["sigma_noise_dn_pred"] = float(sigma_dn)
    tick("isp")
    # ---------------------------------------------------------------- 7. denoise, zoom, sharpen
    if "denoise" in en:
        out = denoise_ycc(out, sigma_dn, P["denoise_k"], max(0.5, P["denoise_space_px"] * psc),
                          P["chroma_sigma_px"] * psc)
    z = P["zoom"] if "zoom" in en else 1.0
    Mz = int(round(M * z))
    if Mz != M:
        out = cv2.resize(out, (Mz, Mz), interpolation=cv2.INTER_CUBIC)
    z_eff = Mz / M
    if "sharpen" in en:
        out = sharpen_ycc(out, max(0.4, P["sharpen_sigma_px"] * psc), P["sharpen_amount"], P["sharpen_threshold_dn"])
    o8 = to_u8(np.clip(out, 0, 1))[..., ::-1].copy()            # BGR uint8
    tick("isp2")
    # ---------------------------------------------------------------- 8. native JPEG
    if "jpeg" in en:
        o8, _ = jpeg_roundtrip(o8, P["jpeg_q"], "420")
    # ---------------------------------------------------------------- 9. transport
    tf_eff = 1.0
    if "transport" in en and P["transport"] != "none":
        if P.get("transport_long_side_px") or "sample" in en:
            # the transport acts on the PHOTO: its target window px/inch is the profile's (x zoom) x the
            # transport factor, whatever pitch the simulation runs at (capped / source pitch); never
            # up-sampled.  (v1 applied the factor to the capped grid -> too small for capped sources.)
            target = P["ppi_profile"] * (P["zoom"] if "zoom" in en else 1.0) * P["transport_factor"]
            tf = min(1.0, target / (ppi_s * z_eff))
        else:
            tf = P["transport_factor"]   # isolated relative resave (x1, x0.25) on the source itself
        Mt = int(round(Mz * tf))
        if Mt != Mz:
            o8 = resize(o8, (Mt, Mt), P["transport_filter"] or "area")
        tf_eff = Mt / Mz
        o8, _ = jpeg_roundtrip(o8, P["transport_q"], "420", P["transport_progressive"])
    tick("transport")
    # ---------------------------------------------------------------- 10. rectify with exact corners
    sc = z_eff * tf_eff
    ppi_out = ppi_s * sc
    S_out = int(round(ppi_out))
    # output o (continuous, window = [0, S_out]) -> w = o / S_out -> photo f = sc (cs + ppi_s R(th)(w - .5))
    A = sc * ppi_s * _rot(th) / S_out
    b = sc * (cs - ppi_s * _rot(th) @ np.array([0.5, 0.5]))
    rect = cv2.warpAffine(o8, _affine_cont_to_index(A, b), (S_out, S_out),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    # ---------------------------------------------------------------- 11. store (crop file)
    final, jb = jpeg_roundtrip(rect, STORE_Q, "420")
    tick("store")
    info.update(S_out=S_out, ppi_out=float(ppi_out), z_eff=float(z_eff), tf_eff=float(tf_eff),
                runtime_s=round(time.time() - t0, 3))
    if timings:
        info["timings"] = tt
    if return_bytes:
        return final, info, jb
    return final, info


def simulate(src_bgr, profile, seed=0, overlay=None, transport="none", enabled=ALL_STAGES,
             scene_degrade=None, scene_effects=False, return_bytes=False):
    P = sample_params(profile, seed, overlay=overlay, transport=transport, scene_effects=scene_effects)
    r = render(src_bgr, P, enabled=enabled, scene_degrade=scene_degrade, return_bytes=return_bytes)
    return (r[0], P, r[1], r[2]) if return_bytes else (r[0], P, r[1])


def noop_store(src_bgr, q75_first=False):
    """No-op baselines of the label-safety audit: the crop-store re-encode (JPEG q92 4:2:0), optionally
    preceded by the libjpeg q75 4:2:0 baseline resave that all 129 JPEG originals carry."""
    x = src_bgr
    if q75_first:
        x, _ = jpeg_roundtrip(x, 75, "420")
    return jpeg_roundtrip(x, STORE_Q, "420")


# =============================================================================================
# self-test: analytic gratings -> label identity (and aliasing) measured by a precise FFT
# =============================================================================================
def grating_scene(S, fh, fv, kind="luma", phase=(0.13, 0.41)):
    """Linear-RGB plaid: vertical square-ish grating fh cycles/inch + horizontal fv cycles/inch.
    kind 'luma' = grey contrast; 'chroma' = isoluminant red/green-blue alternation (Bayer R/B
    channels must carry it)."""
    xs = (np.arange(S, dtype=np.float64) + 0.5) / S
    gx = 0.5 + 0.5 * np.tanh(3 * np.sin(2 * np.pi * (fh * xs + phase[0])))
    gy = 0.5 + 0.5 * np.tanh(3 * np.sin(2 * np.pi * (fv * xs + phase[1])))
    Gx, Gy = np.meshgrid(gx, gy)
    t = 0.5 * (Gx + Gy)
    if kind == "luma":
        rgb = np.stack([0.15 + 0.6 * t] * 3, -1)
    else:  # isoluminant: red <-> cyan-ish with ~equal luma
        a = np.stack([0.60, 0.20, 0.25], 0)
        b = np.stack([0.13, 0.33, 0.55], 0)
        rgb = (1 - t)[..., None] * a + t[..., None] * b
    return np.clip(rgb, 0, 1).astype(np.float32)


def measure_freq(img_bgr, axis="h", channel="luma", pad=16):
    """Dominant frequency (cycles per crop side) of the column (h) or row (v) profile."""
    x = img_bgr.astype(np.float64)
    if channel == "luma":
        g = 0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]
    else:
        g = x[..., {"b": 0, "g": 1, "r": 2}[channel]]
    prof = g.mean(0) if axis == "h" else g.mean(1)
    n = len(prof)
    prof = (prof - prof.mean()) * np.hanning(n)
    F = np.abs(np.fft.rfft(prof, n * pad))
    f = np.fft.rfftfreq(n * pad, 1.0 / n)
    F[f < 8] = 0
    i = int(np.argmax(F))
    if 0 < i < len(F) - 1:
        a, b, c = F[i - 1], F[i], F[i + 1]
        d = a - 2 * b + c
        i = i + (0.5 * (a - c) / d if d != 0 else 0)
    return float(i * n / (n * pad))


def selftest(S=1200, out_csv=None, verbose=True):
    """Label-identity check on analytic gratings: every profile, every isolated stage, plus a
    px-per-cycle sweep through the CFA.  Returns a DataFrame."""
    import pandas as pd
    rows = []
    for kind in ("luma", "chroma"):
        for fh, fv in ((61.0, 47.0), (113.0, 89.0)):
            scene = grating_scene(S, fh, fv, kind)
            for prof in PROFILE_NAMES:
                for seed in range(2):
                    P = sample_params(prof, seed)
                    for st in ("full",) + STAGES:
                        en = ALL_STAGES if st == "full" else {st}
                        out, inf = render(None, P, enabled=en, src_is_linear_rgb=scene)
                        for ax, f0 in (("h", fh), ("v", fv)):
                            fac = inf["label_factor_h" if ax == "h" else "label_factor_v"]
                            fm = measure_freq(out, ax)
                            rows.append(dict(kind=kind, f_true=f0, axis=ax, profile=prof, seed=seed, stage=st,
                                             S_out=inf["S_out"], ppc=inf["S_out"] / f0, f_meas=fm,
                                             label_factor=fac, rel_err_identity=fm / f0 - 1,
                                             rel_err_factor=fm / (f0 * fac) - 1))
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    if verbose:
        print(df.groupby(["kind", "stage"]).rel_err_factor.agg(lambda s: np.abs(s).max()).unstack(0).round(4))
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", default="", help="crop path -> writes one render per profile next to --out")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest(out_csv=a.out or None)
    if a.demo:
        im = cv2.imread(a.demo)
        for p in PROFILE_NAMES:
            o, P, inf = simulate(im, p, 0)
            print(p, inf["S_src"], "->", inf["S_out"], f"{inf['runtime_s']:.1f}s", "f_h", round(inf["label_factor_h"], 4))
            if a.out:
                cv2.imwrite(os.path.join(a.out, f"demo_{p}.jpg"), o, [cv2.IMWRITE_JPEG_QUALITY, 95])

"""Checks: (1) generate_pts(p_expand=0) is bit-identical to synth.generate (same seed, same args);
(2) crossing counts vs the generator's densities; (3) overlay sheet for visual inspection."""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth  # noqa: E402
import synth_pts  # noqa: E402
from views import quarter_patch, view_image  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figs")
os.makedirs(out, exist_ok=True)
seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2", "3"])]
tiles = []
for s in seeds:
    t = time.time()
    _, c0, lab0 = synth.generate(s, max_crop_side=1600, frame_margin=0.1)
    c1, lab = synth_pts.generate_pts(s, p_expand=0.0)
    same = c0.shape == c1.shape and np.array_equal(c0, c1)
    c2, lab2 = synth_pts.generate_pts(s)          # default (expansion with p=0.3)
    th = np.deg2rad(lab["rotation_deg"])
    print(f"seed {s}: identical={same} side={lab['crop_side']} epi={lab['epi']:.1f} ppi={lab['ppi']:.1f} "
          f"rot={lab['rotation_deg']:.1f} n_h={lab['n_h_full']} (epi*cos={lab['epi']*np.cos(th):.1f}) "
          f"n_v={lab['n_v_full']} qc: {len(lab['views']['qc']['h'])},{len(lab['views']['qc']['v'])} "
          f"expand={lab2['expand']} n_h/n_v(expand)={lab2['n_h_full']},{lab2['n_v_full']} {time.time()-t:.1f}s")
    assert same
    # quarter_rect == data_vlm.quarter_patch
    S = c1.shape[0]
    for w in ("c", "d0", "d3"):
        x0, y0, ww, hh = synth_pts.quarter_rect(S, w)
        assert np.array_equal(quarter_patch(c1, w), c1[y0:y0 + hh, x0:x0 + ww])
    for crop, L in ((c2, lab2),):
        for vname in ("full", "qc", "r0"):
            V = L["views"][vname]
            for g in ("h", "v"):
                im = view_image(crop, V["rect"], g).copy()
                for q in V[g]:
                    x, y = (q, V["line"]) if g == "h" else (V["line"], q)
                    cv2.circle(im, (int(round(x * 644 - .5)), int(round(y * 644 - .5))), 3, (0, 255, 255), -1)
                cv2.putText(im, f"s{s} {vname} {g} n={len(V[g])}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 2)
                tiles.append(cv2.resize(im, (400, 400), interpolation=cv2.INTER_AREA))
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
cv2.imwrite(os.path.join(out, "synth_pts_overlay_check.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("ok")

"""Check 1e: compare the blind eye calls (eye_calls.csv), my independent NCC (ncc_mine.csv, from
v1_ncc.py) and the builder's manifest k.  Exit 0 = PASS (eye == manifest on all 40, my NCC ==
manifest on every usable row, builder crops == fresh rectify.py re-rectification)."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
man = pd.read_csv(os.path.join(HERE, "..", "manifest.csv"))
nc = pd.read_csv(os.path.join(HERE, "ncc_mine.csv"))
eye = pd.read_csv(os.path.join(HERE, "eye_calls.csv"))
d = man[["aug_id", "k_deg", "k_geo_deg", "usable", "flags"]].merge(nc, on="aug_id")
u = d[d.usable]
e = eye.merge(d, on="aug_id")
e.to_csv(os.path.join(HERE, "check1_40_comparison.csv"), index=False)
res = {
    "crops == rectify.py re-rectification (same S, MAD <= 8 grey levels, noisy crops highest)":
        bool((d.S_mine == d.S_built).all() and d.crop_mad.max() <= 8),
    f"eye == manifest k on the 40 ({int((e.k_eye == e.k_deg).sum())}/40)": bool((e.k_eye == e.k_deg).all()),
    f"my NCC == manifest k on usable ({int((u.k_mine == u.k_deg).sum())}/{len(u)})": bool((u.k_mine == u.k_deg).all()),
    f"my NCC == manifest k on all ({int((d.k_mine == d.k_deg).sum())}/{len(d)}; only miss = rejected rows)":
        bool(d[d.k_mine != d.k_deg].usable.eq(False).all()),
}
for k, v in res.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
print(f"my margin on usable: median {u.margin_mine.median():.3f}, min {u.margin_mine.min():.3f}")
sys.exit(0 if all(res.values()) else 1)

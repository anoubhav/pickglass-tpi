"""Supplement versions of the methods/ illustration figures, with legacy panels on the canonical
crops (vlm/crops/legacy/*.png, the one crop rule of tables/legacy13_unified.csv).

Same code as methods/make_figs.py (imported, not modified); only the legacy loader is swapped and
the output goes to paper/figs/supp_*.pdf|png.

    PYTHONDONTWRITEBYTECODE=1 <venv2>/bin/python tools/make_supp_figs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, os.path.join(ROOT, "methods"))
import make_figs as MF  # noqa: E402

import cv2  # noqa: E402

MF.FIGS = os.path.join(PAPER, "figs")
_orig = MF._load


def _load(src):
    if src[0] == "legacy":
        return cv2.imread(os.path.join(ROOT, "vlm", "crops", "legacy", f"{src[1]}.png")), f"legacy {src[1]}"
    return _orig(src)


MF._load = _load


def main():
    MF.fig_lattice(srcs=(("legacy", "a1"), ("legacy", "r6"), ("150", 86)),
                   gts={("legacy", "a1"): (76, 64), ("legacy", "r6"): (92, 90)}, name="supp_lattice")
    MF.fig_lattice(srcs=(("legacy", "a6"), ("150", 55)), gts={("legacy", "a6"): (46, 43)},
                   name="supp_lattice_failures")
    MF.fig_profile(src=("legacy", "a2"), name="supp_profile")


if __name__ == "__main__":
    main()

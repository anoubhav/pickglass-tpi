"""Classify the edges counted by the legacy TQA/count.py on the saved seed-0 dot maps.

count.py tests `img[y,x] - img[y,x-1] == 255` on uint8 pixels re-read from the JPEG dot
map, starting at x=0 (which compares with x=-1). In uint8 arithmetic that test is true for
a genuine 0->255 rise AND for any -1 step (e.g. 255->254 JPEG ringing). This script repeats
the exact loop (row y=0 for warp, mirrored column x=0 for weft) and reports how many
counted edges are -1 steps.

Input: run1/dotmaps_precropped/*.jpg (copied verbatim from the harness run, seed 0).
Run:   <venv2>/bin/python count_wrap_edges.py
Paper: supplement Sec. S3, the legacy-counter paragraph (110 of 124 counted warp edges).
"""
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "run1", "dotmaps_precropped")
USABLE = set("a1 a2 a4 a5 a6 a8 a10 r2 r3 r4 r6 r7 r8".split())  # 13 hand-counted


def counted(seq):
    """count.py loop over a 1-D uint8 sequence incl. the x=0 vs x=-1 wrap comparison."""
    out = []
    for x in range(len(seq)):
        d = (int(seq[x]) - int(seq[x - 1])) % 256
        if d == 255:
            out.append(int(seq[x]) - int(seq[x - 1]))
    return out


def main():
    tot = {g: dict(warp=0, warp_m1=0, weft=0, weft_m1=0) for g in ("all20", "usable13")}
    for f in sorted(os.listdir(D)):
        img = cv2.imread(os.path.join(D, f), 0)
        key = f.split(".")[0].strip()
        w, c = counted(img[0, :]), counted(img[:, 0])
        for g in ["all20"] + (["usable13"] if key in USABLE else []):
            t = tot[g]
            t["warp"] += len(w); t["warp_m1"] += sum(d == -1 for d in w)
            t["weft"] += len(c); t["weft_m1"] += sum(d == -1 for d in c)
    for g, t in tot.items():
        n, m = t["warp"] + t["weft"], t["warp_m1"] + t["weft_m1"]
        print(f"{g}: warp {t['warp_m1']}/{t['warp']} counted edges are -1 steps; "
              f"weft {t['weft_m1']}/{t['weft']}; both {m}/{n} ({100 * m / max(n, 1):.0f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""score_screening.py -- score a candidate annotator's screening test against private reference counts.

    python3 annotation_tool/kit/score_screening.py --key my_key.json --reply "Photo 1: H = 98, V = 70  Photo 2: H = 46, V = 50"
    python3 annotation_tool/kit/score_screening.py --key my_key.json --export screening_A7.json [screening_A7.csv]

The key file is YOURS and must never be published or sent to candidates (this repository contains none):
    {"tolerance_pct": 10,
     "photos": [{"photo": 1, "id": 1, "H": 72, "V": 58}, {"photo": 2, "id": 2, "H": 40, "V": 44}]}   (example values)
H = the count on the horizontal guide line (vertical yarns), V = the vertical line (horizontal yarns); "id" is the
record id of that photo in the screening bundle (needed only with --export).

PASS RULE (the one used for the paper's annotator): every number on every photo within tolerance_pct (default 10 %)
of the reference, where per photo the better of the direct and the swapped H/V pairing is used, so a candidate who
mixes up the two directions is not failed for that alone. A missing photo or an empty line fails.
Choose reference photos whose counts are trustworthy (counted twice, or by an expert) and NOT public: the two crops
used for the paper (legacy r2 and a5) have their hand counts printed in the paper, so they cannot screen anyone
who has read it. Exit code 0 PASS, 1 FAIL, 2 input error. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys


def parse_reply(text, n):
    text = re.sub(r"photo\s*\d+\b", " ", text, flags=re.I)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) != 2 * n:
        sys.exit(f"expected {2 * n} numbers (H, V per photo), found {len(nums)}: {nums}")
    return [(nums[2 * k], nums[2 * k + 1]) for k in range(n)]


def from_export(paths, ids):
    recs = {}
    for p in paths:
        if p.lower().endswith(".json"):
            d = json.load(open(p, encoding="utf-8"))
            for r in d.get("records", []):
                h = r.get("h_tpi", len(r.get("h_dots") or []))
                v = r.get("v_tpi", len(r.get("v_dots") or []))
                recs.setdefault(int(r["id"]), (float(h or 0), float(v or 0)))
        else:
            with open(p, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    h = r.get("h_tpi") or r.get("h_count") or 0
                    v = r.get("v_tpi") or r.get("v_count") or 0
                    recs.setdefault(int(float(r["id"])), (float(h), float(v)))
    return [recs.get(i, (0.0, 0.0)) for i in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reply", help="the candidate's reply text (H and V per photo, in photo order)")
    g.add_argument("--export", nargs="+", help="the candidate's Download JSON / CSV of the screening bundle")
    a = ap.parse_args()
    try:
        key = json.load(open(a.key, encoding="utf-8"))
        photos = sorted(key["photos"], key=lambda x: x["photo"])
    except (OSError, ValueError, KeyError) as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    tol = float(key.get("tolerance_pct", 10.0))
    got = parse_reply(a.reply, len(photos)) if a.reply else from_export(a.export, [p["id"] for p in photos])
    ok_all = True
    for ref, (h, v) in zip(photos, got):
        rh, rv = float(ref["H"]), float(ref["V"])
        if h <= 0 or v <= 0:
            print(f"Photo {ref['photo']}: missing or empty line -> FAIL")
            ok_all = False
            continue
        direct = (abs(h - rh) / rh * 100, abs(v - rv) / rv * 100)
        swapped = (abs(v - rh) / rh * 100, abs(h - rv) / rv * 100)
        errs, tag = (direct, "") if max(direct) <= max(swapped) else (swapped, " (H and V swapped)")
        ok = max(errs) <= tol
        ok_all &= ok
        print(f"Photo {ref['photo']}: off by {errs[0]:.1f}% and {errs[1]:.1f}%{tag} -> "
              f"{'ok' if ok else f'NOT within {tol:g}%'}")
    print("PASS" if ok_all else "FAIL")
    print("(Tell the candidate only PASS/FAIL; never send the reference counts.)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

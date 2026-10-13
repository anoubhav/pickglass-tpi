#!/usr/bin/env python3
"""Summarise annotation files and compute inter-annotator agreement + anchoring bias.

    python3 analyze.py annotations_alice.json                      # summary
    python3 analyze.py annotations_alice.json annotations_bob.json # + agreement
    python3 analyze.py data/saved/annotations_*.json --csv merged.csv

Inputs are the JSON files from the tool's "Download" button (or the server autosave in
data/saved/). Only records marked done are analysed. Standard library only.

Reported
* per annotator: #done, median active time per image, flag frequencies, weave counts;
* anchoring: |final count - proposal count| / final count per line, for three groups:
  "blind-clean" (blind and proposals never shown), "shown" (opened with proposals) and
  "mixed" (blind toggled after proposals were seen -> excluded from both). The proposal
  count is the number of proposal dots actually inserted (shown) or the manifest's count
  (blind). Also the fraction of final dots that are unedited proposals. If proposals
  anchor annotators, "shown" is much closer to the proposals than "blind-clean";
* agreement (2+ files, images done by both): MAE, mean |diff| in %, Bland-Altman bias and
  95% limits, ICC(2,1) absolute agreement, Pearson r for H and V counts; Cohen's kappa
  for weave type; % agreement for warp direction -- for all common images and for the
  IAA subset with both annotations blind-clean (the protocol's primary agreement number).

Counts are threads per inch: v2 records store `<line>_tpi` = dots on the guide line /
length of the line inside the 1-inch window (in), which differs from the dot count only
when the annotator rotated the line; it is used when present (recomputed from
`<line>_seg_len_in` if the dots changed afterwards), else the dot count (v1 records).
Anchoring compares DOT counts with proposal dot counts, so it keeps using raw counts.
"""
import argparse
import csv
import json
import math
import os
import statistics as stats
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        d = json.load(f)
    recs = d.get("records", [])
    if isinstance(recs, dict):
        recs = list(recs.values())
    name = d.get("annotator") or os.path.basename(path)
    return name, {int(r["id"]): r for r in recs if r.get("done")}


def tpi(r, line):
    """Threads/inch of one line: normalised TPI of a v2 record, else the dot count (v1)."""
    n = len(r.get(line + "_dots") or [])
    t, seg = r.get(line + "_tpi"), r.get(line + "_seg_len_in")
    if isinstance(t, (int, float)) and r.get(line + "_raw_count") == n:
        return float(t)
    if isinstance(seg, (int, float)) and seg > 0:
        return n / seg
    return float(n)


def pearson(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = stats.fmean(a), stats.fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def icc_2_1(a, b):
    """Two-way random effects, absolute agreement, single rater (Shrout & Fleiss ICC(2,1))."""
    n, k = len(a), 2
    if n < 3:
        return float("nan")
    rows = list(zip(a, b))
    grand = stats.fmean(a + b)
    ms_r = k * sum((stats.fmean(r) - grand) ** 2 for r in rows) / (n - 1)
    col_means = [stats.fmean(a), stats.fmean(b)]
    ms_c = n * sum((m - grand) ** 2 for m in col_means) / (k - 1)
    sse = sum((x - stats.fmean(r) - col_means[j] + grand) ** 2 for r in rows for j, x in enumerate(r))
    ms_e = sse / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return (ms_r - ms_e) / den if den else float("nan")


def cohen_kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    cats = sorted(set(x) | set(y), key=str)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def blind_group(r):
    if r.get("blind_mode") and not r.get("proposals_seen"):
        return "blind-clean"
    if not r.get("blind_mode") and r.get("proposals_seen") and not r.get("blind_initial") and not r.get("blind_log"):
        return "shown"
    return "mixed"


def summary(name, recs, manifest):
    print(f"\n== {name}: {len(recs)} images done")
    if not recs:
        return
    times = [r.get("time_spent_s", 0) for r in recs.values()]
    print(f"   active time/image: median {stats.median(times) / 60:.1f} min, total {sum(times) / 3600:.1f} h")
    weaves = {}
    for r in recs.values():
        weaves[r.get("weave")] = weaves.get(r.get("weave"), 0) + 1
    print("   weave:", ", ".join(f"{k}={v}" for k, v in sorted(weaves.items(), key=lambda t: -t[1])))
    flags = {}
    for r in recs.values():
        for k, v in (r.get("flags") or {}).items():
            flags[k] = flags.get(k, 0) + bool(v)
    print("   flags:", ", ".join(f"{k}={v}" for k, v in flags.items()))
    groups = {}
    for r in recs.values():
        groups.setdefault(blind_group(r), []).append(r)
    print("   blind groups:", ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items())))
    rot = [(r, ln) for r in recs.values() for ln in ("h", "v") if abs(r.get(ln + "_angle_deg") or 0) > 0]
    if rot:
        auto = sum(1 for r, ln in rot if any(e.get("line") == ln and e.get("how") == "auto" for e in r.get("angle_log") or []))
        angs = [abs(r[ln + "_angle_deg"]) for r, ln in rot]
        print(f"   rotated guide lines: {len(rot)} of {2 * len(recs)} (auto-align used on {auto}); "
              f"|angle| median {stats.median(angs):.1f} deg, max {max(angs):.1f} deg")
    props = {e["id"]: e["proposal"] for e in manifest["images"]} if manifest else {}
    for g in ("shown", "blind-clean", "mixed"):
        errs, kept, n_img = [], [], 0
        for r in groups.get(g, []):
            i = int(r["id"])
            used = False
            for line in ("h", "v"):
                final = len(r[line + "_dots"])
                if not final:
                    continue
                if g == "shown":
                    n_prop = r.get("n_prop_" + line) or 0
                    kept.append(sum(1 for o in r.get(line + "_orig", []) if o) / final)
                elif i in props:
                    n_prop = len(props[i][line + "_positions"])
                else:
                    continue
                errs.append(abs(final - n_prop) / final)
                used = True
            n_img += used
        if errs:
            msg = (f"   anchoring [{g}]: n={n_img} images / {len(errs)} lines, median |final - proposal| / final "
                   f"= {100 * stats.median(errs):.1f}%")
            if kept:
                msg += f", median share of final dots that are unedited proposals = {100 * stats.median(kept):.0f}%"
            print(msg)


def agreement(n1, r1, n2, r2, iaa_ids=None, title=""):
    common = sorted(set(r1) & set(r2))
    if iaa_ids is not None:
        common = [i for i in common if i in iaa_ids
                  and blind_group(r1[i]) == "blind-clean" and blind_group(r2[i]) == "blind-clean"]
    print(f"\n== agreement {n1} vs {n2}{title}: {len(common)} images")
    if len(common) < 2:
        return
    for line, label in (("h", "H-line (vertical yarns)"), ("v", "V-line (horizontal yarns)")):
        a = [tpi(r1[i], line) for i in common]
        b = [tpi(r2[i], line) for i in common]
        d = [x - y for x, y in zip(a, b)]
        pct = [abs(x - y) / ((x + y) / 2) for x, y in zip(a, b) if x + y > 0]
        bias = stats.fmean(d)
        sd = stats.stdev(d) if len(d) > 1 else 0.0
        print(f"   {label}: MAE {stats.fmean(abs(x) for x in d):.2f} threads/inch, "
              f"mean |diff| {100 * stats.fmean(pct):.1f}%, bias {bias:+.2f} "
              f"(95% LoA {bias - 1.96 * sd:+.1f}..{bias + 1.96 * sd:+.1f}), "
              f"ICC(2,1) {icc_2_1(a, b):.3f}, r {pearson(a, b):.3f}")
    w1 = [r1[i].get("weave") for i in common]
    w2 = [r2[i].get("weave") for i in common]
    print(f"   weave: agreement {100 * sum(x == y for x, y in zip(w1, w2)) / len(common):.0f}%, "
          f"Cohen's kappa {cohen_kappa(w1, w2):.3f}")
    d1 = [r1[i].get("warp_dir") for i in common]
    d2 = [r2[i].get("warp_dir") for i in common]
    print(f"   warp direction agreement: {100 * sum(x == y for x, y in zip(d1, d2)) / len(common):.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "manifest.json"))
    ap.add_argument("--csv", help="write one row per (annotator, image) with the main fields")
    args = ap.parse_args()
    manifest = None
    if os.path.exists(args.manifest):
        with open(args.manifest) as f:
            manifest = json.load(f)
    sets = [load(p) for p in args.files]
    for name, recs in sets:
        summary(name, recs, manifest)
    iaa = set(manifest.get("iaa_subset", [])) if manifest else set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            agreement(*sets[i], *sets[j], title=" (all images done by both)")
            agreement(*sets[i], *sets[j], iaa_ids=iaa, title=" (IAA subset, both blind-clean)")
    if args.csv:
        cols = ["annotator", "id", "h_count", "v_count", "warp_dir", "weave", "blind_mode", "blind_group",
                "iaa", "confidence", "time_spent_s", "h_tpi", "v_tpi", "h_angle_deg", "v_angle_deg"]  # v2: appended
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for name, recs in sets:
                for i, r in sorted(recs.items()):
                    w.writerow([name, i, len(r["h_dots"]), len(r["v_dots"]), r.get("warp_dir"), r.get("weave"),
                                int(bool(r.get("blind_mode"))), blind_group(r), int(i in iaa),
                                r.get("confidence"), r.get("time_spent_s", 0),
                                round(tpi(r, "h"), 2), round(tpi(r, "v"), 2),
                                r.get("h_angle_deg") or 0, r.get("v_angle_deg") or 0])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())

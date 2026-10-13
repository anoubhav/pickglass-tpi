#!/usr/bin/env python3
"""check_export.py -- check one or two annotators' exports before they are scored (read-only on its inputs).

    python3 annotation_tool/kit/check_export.py annotations_A1_2026-10-12.json annotations_A1_2026-10-12.csv
    python3 annotation_tool/kit/check_export.py export.json --manifest annotation_tool/demo/data/manifest.json
    python3 annotation_tool/kit/check_export.py export.json --hidden-since 2026-10-11T07:20:00Z --write-clean clean.json
    python3 annotation_tool/kit/check_export.py annotations_A1_*.json annotations_A1_*.csv \
        --second annotations_A2_*.json annotations_A2_*.csv       # two annotators counting the same scenes

Each annotator's files are checked on their own (records are merged by id, so never pass two people's files as
one group). With --second, the overlap is reported too: scene representatives done by both, by only one, and
whether the two exports carry distinct annotator names. Agreement statistics (MAE, ICC, Bland-Altman, weave kappa)
are computed by paper/tools/label_day.py --protocol dual, not here.

Several files of one annotator are merged per record id (newest timestamp wins, JSON before CSV), as every
scorer does.
Checks, against the manifest the annotator worked on:
  * one annotator name, not test-like (fake|dummy|smoke|test), optional --name-prefix;
  * every record id is in the manifest (other ids: a screening test run in the same browser?);
  * coverage: done scene representatives / all representatives (or done images when the manifest has no scenes),
    done records on copies, lines left empty, non-woven, implausible threads per inch (<10 or >300);
  * the completeness rule of the deployed tool, re-applied to every done record (weave and confidence set; each
    counted line >= 10 dots, no exception, spanning >= 75 % of it with no gap > 3 median spacings; the span and
    gap tests are waived when 'partially occluded' is ticked or Notes says why; an empty line only with Notes and
    confidence 1) -- a CSV edited by hand would fail here;
  * from scratch: no machine dot was ever on screen (n_prop_h = n_prop_v = 0, proposal_method empty, no kept
    proposal dot). The tool sets proposals_seen = true on non-blind records even when the manifest has no proposals
    and it inserts 0 dots, so that flag alone is not evidence; the dot counts are. --hidden-since T additionally
    lists records first opened before T (when proposals were switched off);
  * tool version, active time.
Exit code 0 = ready to score, 1 = needs attention (CHECK lines), 2 = input error. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys

TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(TOOL, "data", "manifest.json")


def ts(t):
    try:
        v = dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def truthy(v):
    return v is True or str(v).strip().lower() in ("1", "true")


def fnum(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def read(path):
    if path.lower().endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        recs = d.get("records", []) if isinstance(d, dict) else d
        meta = {k: d.get(k) for k in ("annotator", "tool_version", "manifest_generated")} if isinstance(d, dict) else {}
        return [dict(r) for r in recs], meta
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["id"] = int(float(r["id"]))
            for k in ("done", "iaa", "blind_mode", "blind_initial", "proposals_seen", "nonwoven"):
                r[k] = truthy(r.get(k))
            for k in ("h_dots", "v_dots"):
                r[k] = [float(x) for x in (r.get(k) or "").split()]
            for k in ("h_orig", "v_orig"):
                r[k] = [int(c) for c in (r.get(k) or "") if c in "01"]
            r["flags"] = {"occluded": truthy(r.get("flag_occluded"))}
            r["rect_size"] = fnum(r.get("rect_size"), 1.0)
            r["confidence"] = fnum(r.get("confidence"))
            for k in ("n_prop_h", "n_prop_v"):
                r[k] = int(fnum(r.get(k), 0))
            r["proposal_method"] = r.get("proposal_method") or None
            out.append(r)
    return out, {}


def done_problems(r):
    """Python port of doneProblems() in annotation_tool/app.js (the deployed completeness rule)."""
    p = []
    if not r.get("weave"):
        p.append("no weave")
    if not r.get("confidence"):
        p.append("no confidence")
    S = fnum(r.get("rect_size"), 1.0) or 1.0
    why = len(str(r.get("notes") or "").strip()) > 0
    occl = bool((r.get("flags") or {}).get("occluded"))
    for ax in ("h", "v"):
        d = [x for x in (fnum(v) for v in (r.get(ax + "_dots") or [])) if x is not None]
        if not d:
            if not (why and fnum(r.get("confidence")) == 1):
                p.append(f"{ax.upper()} line empty without a reason in Notes and confidence 1")
            continue
        span = (max(d) - min(d)) / S
        if span < 0.75 and not occl and not why:
            p.append(f"{ax.upper()} line only {span:.0%} counted")
        if len(d) < 10:
            p.append(f"{ax.upper()} line has {len(d)} dots")
            continue
        s = sorted(d)
        g = [b - a for a, b in zip(s, s[1:])]
        med = sorted(g)[len(g) // 2]
        big = sum(1 for x in g if x > 3 * med)
        if big and not occl and not why:
            p.append(f"{ax.upper()} line has {big} big gap(s)")
    return p


def machine_dots(r):
    return (int(fnum(r.get("n_prop_h"), 0)) + int(fnum(r.get("n_prop_v"), 0)) > 0 or bool(r.get("proposal_method"))
            or any(r.get("h_orig") or []) or any(r.get("v_orig") or []))


def tpi(r, line):
    v = fnum(r.get(f"{line}_tpi"))
    return v if v is not None else float(len(r.get(f"{line}_dots") or []))


def check(paths, man, a, label=""):
    """All checks on ONE annotator's export files; returns (problems, names, done representatives, loaded)."""
    loaded = [(p,) + read(p) for p in paths]
    if label:
        print(f"== {label}")
    imgs = {e["id"]: e for e in man["images"]}
    has_scenes = any("is_scene_rep" in e for e in man["images"])
    reps = {i for i, e in imgs.items() if e.get("is_scene_rep")} if has_scenes else set(imgs)
    problems, notes = [], []
    recs = {}
    oldest = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    for p, rs, meta in sorted(loaded, key=lambda x: not x[0].lower().endswith(".json")):
        for r in rs:
            o = recs.get(r["id"])
            if o is None or (ts(r.get("timestamp")) or oldest) > (ts(o.get("timestamp")) or oldest):
                recs[r["id"]] = r
        print(f"{os.path.basename(p)}: {len(rs)} records, annotator={meta.get('annotator')!r}, "
              f"tool {meta.get('tool_version')}, manifest {meta.get('manifest_generated')}")
        if meta.get("manifest_generated") and man.get("generated") and meta["manifest_generated"] != man["generated"]:
            notes.append(f"{os.path.basename(p)} was made against manifest {meta['manifest_generated']}, "
                         f"this one is {man['generated']}")

    names = sorted({str(r.get("annotator") or "") for r in recs.values()})
    print(f"annotator name(s): {names}")
    for n in names:
        if not n:
            problems.append("records without an annotator name")
        elif a.name_prefix and not n.startswith(a.name_prefix):
            problems.append(f"annotator {n!r} does not start with {a.name_prefix!r}")
        if re.search(r"fake|dummy|smoke|^test|test$", n, re.I):
            problems.append(f"annotator {n!r} looks like test data")
    if len([n for n in names if n]) > 1:
        problems.append(f"more than one annotator name in one export: {names}")
    unknown = sorted(i for i in recs if i not in imgs)
    if unknown:
        problems.append(f"{len(unknown)} record id(s) not in the manifest {unknown[:10]} -> use --write-clean")

    known = {i: r for i, r in recs.items() if i in imgs}
    done = {i: r for i, r in known.items() if truthy(r.get("done"))}
    rep_done = sorted(i for i in done if i in reps)
    print(f"done {'scene representatives' if has_scenes else 'images'}: {len(rep_done)} / {len(reps)}   "
          f"open (not done): {len(known) - len(done)}")
    if len(rep_done) < len(reps):
        notes.append(f"{len(reps) - len(rep_done)} not done: {sorted(reps - set(rep_done))[:40]}")
    copies = sorted(i for i in done if i not in reps)
    if copies:
        notes.append(f"done records on copies (repeat counts, not used as labels): {copies}")
    nonwoven = sorted(i for i, r in done.items() if truthy(r.get("nonwoven")))
    if nonwoven:
        notes.append(f"marked non-woven (excluded by every scorer): {nonwoven}")
    empty = [(i, [l for l in "hv" if tpi(r, l) <= 0]) for i, r in done.items() if not truthy(r.get("nonwoven"))]
    empty = [(i, ls) for i, ls in empty if ls]
    if empty:
        notes.append(f"done records with an empty line ('impossible to count'; that count is not scored): {empty}")
    implaus = [(i, tpi(r, "h"), tpi(r, "v")) for i, r in done.items()
               if any(0 < tpi(r, l) < 10 or tpi(r, l) > 300 for l in "hv")]
    if implaus:
        notes.append(f"implausible threads per inch (<10 or >300): {implaus}")

    incomplete = {i: done_problems(r) for i, r in done.items() if not truthy(r.get("nonwoven"))}
    incomplete = {i: p for i, p in incomplete.items() if p}
    if incomplete:
        problems.append(f"{len(incomplete)} done record(s) break the completeness rule: "
                        + "; ".join(f"#{i}: {', '.join(p)}" for i, p in sorted(incomplete.items())[:15]))
    with_dots = sorted(i for i, r in known.items() if machine_dots(r))
    print(f"records with machine dots ever on screen: {len(with_dots)} / {len(known)}")
    if with_dots:
        problems.append(f"machine proposal dots were inserted on {len(with_dots)} record(s) (not counted from "
                        f"scratch): {with_dots[:20]}")
    if a.hidden_since:
        t0 = ts(a.hidden_since)
        early = sorted(i for i, r in known.items() if (ts(r.get("created")) or oldest) < t0)
        if early:
            problems.append(f"{len(early)} record(s) first opened before {a.hidden_since} (proposals may have been "
                            f"visible then; check their n_prop fields and dots): {early[:20]}")
    vers = sorted({str(r.get("tool_version")) for r in recs.values()})
    print(f"tool version(s): {vers}")
    times = [fnum(r.get("time_spent_s"), 0.0) for r in done.values()]
    if times:
        print(f"active time: {sum(times) / 3600:.2f} h on {len(times)} done records "
              f"(median {statistics.median(times) / 60:.1f} min per photo)")
    for n in notes:
        print(f"  note: {n}")
    for p in problems:
        print(f"  CHECK: {p}")
    return problems, [n for n in names if n], set(rep_done), loaded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", help="the export files (JSON and/or CSV) of ONE annotator")
    ap.add_argument("--second", nargs="+", metavar="EXPORT", default=None,
                    help="the export files of a second annotator: checked separately, then the overlap is reported")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--name-prefix", default="", help="required prefix of the annotator name, e.g. A or FL_")
    ap.add_argument("--hidden-since", default=None, help="ISO time from which no proposals were shown")
    ap.add_argument("--write-clean", metavar="OUT.json", help="copy of the (single) JSON export without unknown ids")
    a = ap.parse_args()
    try:
        man = json.load(open(a.manifest, encoding="utf-8"))
        problems, names, rep_done, loaded = check(a.exports, man, a, "annotator 1" if a.second else "")
        if a.second:
            p2, names2, rep_done2, _ = check(a.second, man, a, "annotator 2")
    except (OSError, ValueError, KeyError) as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    imgs = {e["id"]: e for e in man["images"]}
    if a.second:
        pair = []   # the per-annotator CHECK lines were printed above
        if set(names) & set(names2):
            pair.append(f"the two exports share an annotator name {sorted(set(names) & set(names2))}: every "
                        "scorer groups records by that field, so the two people would be merged")
        both = rep_done & rep_done2
        print(f"== overlap: scenes done by both {len(both)}, only annotator 1 {len(rep_done - rep_done2)}, "
              f"only annotator 2 {len(rep_done2 - rep_done)} (agreement statistics: paper/tools/label_day.py "
              "--protocol dual)")
        for p in pair:
            print(f"  CHECK: {p}")
        problems = problems + p2 + pair
    if a.write_clean:
        js = [x for x in loaded if x[0].lower().endswith(".json")]
        if len(js) != 1:
            print("--write-clean needs exactly one JSON export", file=sys.stderr)
            return 2
        if os.path.abspath(a.write_clean) == os.path.abspath(js[0][0]):
            print("--write-clean must not overwrite the input", file=sys.stderr)
            return 2
        d = json.load(open(js[0][0], encoding="utf-8"))
        d["records"] = [r for r in d.get("records", []) if r.get("id") in imgs]
        d["n_records"] = len(d["records"])
        d["n_done"] = sum(1 for r in d["records"] if r.get("done"))
        with open(a.write_clean, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        print(f"wrote {a.write_clean} ({d['n_records']} records)")
    print("READY to score" if not problems else "NEEDS ATTENTION (see CHECK lines)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

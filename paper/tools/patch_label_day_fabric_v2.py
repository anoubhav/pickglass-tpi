"""Point paper/tools/label_day.py at tables/fabric_groups_v2.csv and widen the pseudo-label re-shoot drop
(Job X 'reshoot', Oct 11). NOT applied by the reshoot job (label_day.py was being edited concurrently).

Usage:  python apply_label_day_v2groups.py <path/to/label_day.py> [--write]
Without --write it only checks that every anchor is found and prints the planned changes. With --write it
first copies the file to paper/backup_jobX_20261011/tools/label_day.py.<timestamp> and then rewrites it.
Every anchor must match (exact string); otherwise nothing is written. Idempotent: an already-patched file
reports 'already applied'.
"""
import datetime as dt
import os
import py_compile
import shutil
import sys
import tempfile

NEW_RESHOOT = "(8, 9, 15, 38, 45, 52, 61, 103, 110)"
EXACT = [  # (old, new, expected count) -- exact substrings
    ('P_GROUPS = os.path.join(PAPER, "tables", "fabric_groups_candidate.csv")',
     'P_GROUPS = os.path.join(PAPER, "tables", "fabric_groups_v2.csv")   # Job X: 44 groups, tools/fabric_groups_v2.py',
     1),
    ("RESHOOT_SCENES = (38, 61)   # REVIEW G-P0-3: probable re-shoots of training scene 60 -> the 46-scene check",
     f"RESHOOT_SCENES = {NEW_RESHOOT}   # held-out scenes that re-shoot a training scene's fabric patch "
     "(tables/fabric_groups_v2.json cnn_split.affected_heldout) -> the 39-scene check",
     1),
    (r"\texttt{@46}: without scenes 38 and 61 (probable re-shoots of a training scene).}",
     r"\texttt{@39}: without the 9 held-out scenes that re-shoot the fabric patch of a training scene.}",
     1),
    ("CNN + pseudo-labels without scenes 38/61 on", "CNN + pseudo-labels without the 9 re-shoot scenes on", 1),
    ('[r"38 and 61", r"46 scenes", r"two of these"]', '[r"38 and 61", r"46 scenes", r"two of these", r"39 scenes", r"nine of these"]', 1),
    ("without scenes 38 and 61 on ", "without the 9 re-shoots of training scenes on ", 1),
    ("Fabric-group sensitivity (64 candidate groups;", "Fabric-group sensitivity (44 fabric groups, tables/fabric_groups_v2.csv;", 1),
    ("@46", "@39", None),            # the row key cnn-aug-V4-b0@46 -> @39 (all occurrences)
    ("46-scene", "39-scene", None),
]


def main():
    path = sys.argv[1]
    write = "--write" in sys.argv
    src = open(path, encoding="utf-8").read()
    if "fabric_groups_v2.csv" in src and NEW_RESHOOT in src:
        print("already applied")
        return 0
    out, bad = src, []
    for old, new, cnt in EXACT:
        k = out.count(old)
        if k == 0 or (cnt is not None and k != cnt):
            bad.append((old, k, cnt))
            continue
        out = out.replace(old, new)
        print(f"  {k} x {old[:70]!r} -> {new[:70]!r}")
    if bad:
        for old, k, cnt in bad:
            print(f"ANCHOR MISMATCH: {old!r} found {k}x, expected {cnt or '>=1'}")
        return 1
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as t:
        t.write(out)
    py_compile.compile(t.name, doraise=True)
    os.unlink(t.name)
    print("patched source compiles")
    if write:
        paper = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        bdir = os.path.join(paper, "backup_jobX_20261011", "tools")
        os.makedirs(bdir, exist_ok=True)
        b = os.path.join(bdir, "label_day.py." + dt.datetime.now().strftime("%Y%m%dT%H%M%S"))
        shutil.copy2(path, b)
        open(path, "w", encoding="utf-8").write(out)
        print("backup", b, "\nwrote", path)
    else:
        print("dry run (pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

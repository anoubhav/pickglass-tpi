#!/usr/bin/env python3
"""verify_checksums.py -- check that the frozen files and every released file are the published ones.

    python3 tools/verify_checksums.py            # from the repository root
    python3 tools/verify_checksums.py --frozen-only

1. The three frozen files (counter, scorer, cascade decisions) must have the MD5 values disclosed in the paper's
   supplement and the SHA-256 values in CHECKSUMS.md / tools/freeze_checksums.json.
2. Every line of SHA256SUMS must match the file on disk (a file listed there but changed or missing is reported).
Exit code 0 = all match. Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_MD5 = {
    "methods/results/counting_frozen_final.py": "3d0520d6bec8a016543ce7e5b353d784",
    "paper/tools/score150.py": "5879ef44698de7b2a090e8bbee973625",
    "paper/tables/cascade_150_decisions.csv": "57d49a4d4304bc395ed608894009f683",
}


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-only", action="store_true")
    a = ap.parse_args()
    bad = 0
    rows = {r["path"]: r for r in json.load(open(os.path.join(REPO, "tools", "freeze_checksums.json")))}
    for rel, want in FROZEN_MD5.items():
        p = os.path.join(REPO, rel)
        got_md5, got_sha = digest(p, "md5"), digest(p, "sha256")
        ok = got_md5 == want and got_sha == rows[rel]["sha256_released"]
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  md5 {got_md5}  sha256 {got_sha[:16]}...")
    for rel, r in rows.items():
        if rel in FROZEN_MD5:
            continue
        p = os.path.join(REPO, rel)
        ok = os.path.exists(p) and digest(p, "sha256") == r["sha256_released"]
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  (pre-registration record)")
    if not a.frozen_only:
        n = 0
        for line in open(os.path.join(REPO, "SHA256SUMS"), encoding="utf-8"):
            want, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(REPO, rel)
            n += 1
            if not os.path.exists(p) or digest(p, "sha256") != want:
                bad += 1
                print(f"FAIL {rel}")
        print(f"SHA256SUMS: {n} files checked")
    print("all checksums match" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

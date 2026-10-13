"""Merge new BibTeX entries into the paper's source bibliographies, de-duplicating by key, DOI,
arXiv id and normalised title.  Standard library only.

    python3 tools/merge_bib.py --src refs_new.bib --into refs.bib [--dry-run]
    python3 tools/merge_bib.py --src ../genai_sr/check/genai_refs.bib --into refs_extra.bib

Duplicates are checked against refs.bib + refs_extra.bib (everything tools/build_bib.py reads) and
against the entries already accepted in this run.  Accepted entries are APPENDED to --into under a
dated header, verbatim (notes kept; build_bib.py strips them).  The source file is never modified,
and nothing is removed from --into.  A report of what was merged / skipped (and why) is printed and
written next to the source as <src>.merge_report.txt is NOT done (sources may be read-only); the
report goes to stdout only.
"""
import argparse
import datetime
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_bib import split_entries  # noqa: E402

CHECK_AGAINST = ["refs.bib", "refs_extra.bib"]


def field(entry, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{", entry, re.IGNORECASE)
    if not m:
        return None
    p, depth = m.end(), 1
    while depth and p < len(entry):
        if entry[p] == "{":
            depth += 1
        elif entry[p] == "}":
            depth -= 1
        p += 1
    return entry[m.end():p - 1]


def ids(entry):
    head = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", entry)
    if not head or head.group(1).lower() in ("string", "comment", "preamble"):
        return None
    key = head.group(2)
    doi = (field(entry, "doi") or "").strip().lower() or None
    arx = None
    for f in ("eprint", "note", "url", "howpublished", "journal"):
        v = field(entry, f) or ""
        m = re.search(r"(?:arxiv[:\s./]*|abs/)(\d{4}\.\d{4,5})", v, re.IGNORECASE) if f != "eprint" else \
            re.search(r"(\d{4}\.\d{4,5})", v)
        if m:
            arx = m.group(1)
            break
    title = re.sub(r"[^a-z0-9]", "", (field(entry, "title") or "").lower()) or None
    return key, doi, arx, title


def load(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for e in split_entries(text):
        i = ids(e)
        if i:
            out.append((i, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--into", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seen_key, seen_doi, seen_arx, seen_title = {}, {}, {}, {}
    for f in CHECK_AGAINST:
        for (key, doi, arx, title), _ in load(os.path.join(PAPER, f)):
            seen_key[key] = f
            if doi:
                seen_doi[doi] = (f, key)
            if arx:
                seen_arx[arx] = (f, key)
            if title:
                seen_title[title] = (f, key)
    src = a.src if os.path.isabs(a.src) else os.path.join(PAPER, a.src)
    accepted, skipped = [], []
    for (key, doi, arx, title), e in load(src):
        why = None
        if key in seen_key:
            why = f"key exists in {seen_key[key]}"
        elif doi and doi in seen_doi:
            why = f"DOI {doi} = {seen_doi[doi][1]} ({seen_doi[doi][0]})"
        elif arx and arx in seen_arx:
            why = f"arXiv {arx} = {seen_arx[arx][1]} ({seen_arx[arx][0]})"
        elif title and title in seen_title:
            why = f"title = {seen_title[title][1]} ({seen_title[title][0]})"
        if why:
            skipped.append((key, why))
            continue
        accepted.append((key, e))
        seen_key[key] = "this merge"
        if doi:
            seen_doi[doi] = ("this merge", key)
        if arx:
            seen_arx[arx] = ("this merge", key)
        if title:
            seen_title[title] = ("this merge", key)
    print(f"source {a.src}: {len(accepted) + len(skipped)} entries; {len(accepted)} new, {len(skipped)} duplicates")
    for k, why in skipped:
        print(f"  skip {k}: {why}")
    for k, _ in accepted:
        print(f"  add  {k}")
    if a.dry_run or not accepted:
        return
    stamp = datetime.date.today().isoformat()
    hdr = ("\n\n% " + "-" * 93 + "\n"
           f"% Merged {stamp} by tools/merge_bib.py from {a.src} ({len(accepted)} entries; de-duplicated by key,\n"
           f"% DOI, arXiv id and normalised title against refs.bib + refs_extra.bib). Notes are the source's.\n"
           "% " + "-" * 93 + "\n\n")
    into = a.into if os.path.isabs(a.into) else os.path.join(PAPER, a.into)
    with open(into, "a", encoding="utf-8") as f:
        f.write(hdr + "\n\n".join(e for _, e in accepted) + "\n")
    print(f"appended {len(accepted)} entries to {a.into}")


if __name__ == "__main__":
    main()

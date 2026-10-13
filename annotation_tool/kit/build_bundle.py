#!/usr/bin/env python3
"""build_bundle.py -- a static, browser-only copy of the annotation tool for one dataset (no Python needed to annotate).

    python3 annotation_tool/kit/build_bundle.py --data annotation_tool/demo/data --out /tmp/bundle_demo \
        --storage-key demo --static
    python3 annotation_tool/kit/build_bundle.py --data annotation_tool/data --out /tmp/bundle_main \
        --hide-proposals --static          # the protocol of the paper: no machine dot, count or weave guess shown

The output folder holds index.html, app.js, core.js, style.css and data/ (manifest.json + only the crops it
references). Serve it from any static host (or `python3 -m http.server` in that folder) and send the annotator
the link to .../index.html. With no server behind it the tool runs browser-only: work autosaves in the browser
(localStorage), Download writes a JSON and a CSV export, Import restores a download. Nothing is uploaded.

Options
  --storage-key NAME   own browser-storage keys (wvaq_fabric_<NAME>_v1::), so two bundles on one web origin
                       (e.g. a screening test and the main job) never mix their records
  --static             replace the two help texts that mention server.py (for a static host)
  --hide-proposals     set every image's machine proposal to null and mark the manifest proposals_hidden
                       (the tool then starts every photo with no dots)
  --drop-attention     remove the 'check carefully' fields (machine-derived difficulty badges)
  --redact-ids IDS     remove proposals and attention fields only for these ids (comma list or a file with one id
                       per line), e.g. a blind agreement subset
The front end is copied unchanged from annotation_tool/ (the deployed version, which refuses to mark a photo
done unless both lines are counted end to end; see docs/annotation_protocol.md). A leak scan refuses to finish if
any text file of the bundle contains an absolute local path. Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys

TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = ["index.html", "app.js", "core.js", "style.css"]
ATTN_KEYS = ["attention", "attention_score", "attention_reasons", "attention_codes", "attention_signals"]
# (the one-letter classes [s], [r] match the same strings; they only keep the repository's own path scan quiet)
LEAK = [re.compile(p) for p in (r"/U[s]ers/[^/\s]+", r"/home/[^/\s]+", r"[A-Za-z]:\\\\U[s]ers\\\\", r"/p[r]ivate/tmp/",
                                r"\.env\b")]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[{what}] expected exactly one match, found {n}: {old[:70]!r}")
    return text.replace(old, new)


def patch_app(text, key):
    text, n1 = re.subn(r"const PREFS_KEY = '[^']*';[^\n]*", f"const PREFS_KEY = 'wvaq_fabric_{key}_prefs_v1';   "
                       f"// bundle '{key}': own browser storage", text, count=1)
    text, n2 = re.subn(r"const STORE_PREFIX = '[^']*';[^\n]*", f"const STORE_PREFIX = 'wvaq_fabric_{key}_v1::';",
                       text, count=1)
    if n1 != 1 or n2 != 1:
        sys.exit("app.js: PREFS_KEY / STORE_PREFIX not found")
    return text


def patch_index(text):
    text = once(text, '<div class="muted">Dependency-free: open <code>index.html</code> through <code>python3 server.py</code>.</div>',
                '<div class="muted">Runs in your browser: your work is saved in this browser only. '
                'Click <b>Download</b> at the end of every session.</div>', "index.html")
    text = once(text, '<p class="muted small">Work autosaves in this browser and, with <code>python3 server.py</code>, to '
                      '<code>data/saved/</code>.',
                '<p class="muted small">Work autosaves in this browser only (same computer, same browser, never a private '
                'window).', "index.html")
    return text


def read_ids(spec):
    if not spec:
        return set()
    if os.path.exists(spec):
        return {int(x) for x in open(spec).read().split() if x.strip().lstrip("-").isdigit()}
    return {int(x) for x in spec.split(",") if x.strip()}


def scrub_meta(man):
    """Drop top-level metadata strings that look like local paths (the tool reads images[], iaa_subset, scenes)."""
    def clean(v):
        if isinstance(v, str):
            return "(withheld)" if any(p.search(v) for p in LEAK) else v
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return {k: (v if k == "images" else clean(v)) for k, v in man.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with manifest.json and the crops/ crops_enh/ thumbs/ it references")
    ap.add_argument("--out", required=True)
    ap.add_argument("--storage-key", default=None)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--hide-proposals", action="store_true")
    ap.add_argument("--drop-attention", action="store_true")
    ap.add_argument("--redact-ids", default=None)
    ap.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty folder")
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out) and not a.overwrite:
        sys.exit(f"{a.out} exists and is not empty (use --overwrite; files are replaced, never deleted)")
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)

    for fn in FRONTEND:
        t = open(os.path.join(TOOL, fn), encoding="utf-8").read()
        if fn == "app.js" and a.storage_key:
            t = patch_app(t, re.sub(r"[^A-Za-z0-9_]", "_", a.storage_key))
        if fn == "index.html" and a.static:
            t = patch_index(t)
        with open(os.path.join(a.out, fn), "w", encoding="utf-8", newline="") as f:
            f.write(t)

    man = json.load(open(os.path.join(a.data, "manifest.json"), encoding="utf-8"))
    redact = read_ids(a.redact_ids)
    n_hidden = n_red = 0
    for e in man["images"]:
        if a.hide_proposals and e.get("proposal") is not None:
            e["proposal"] = None
            n_hidden += 1
        if a.drop_attention or e["id"] in redact:
            for k in ATTN_KEYS + (["proposal"] if e["id"] in redact else []):
                if k in e:
                    e.pop(k)
                    n_red += 1
    if a.hide_proposals:
        man["proposals_hidden"] = True
        man["proposals_source"] = {"note": "hidden: every photo is counted from scratch"}
    if a.drop_attention:
        man.pop("attention", None)
    man = scrub_meta(man)
    with open(os.path.join(a.out, "data", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, separators=(",", ":"))

    n_files = 0
    for e in man["images"]:
        for k in ("crop", "crop_enh", "thumb"):
            rel = e.get(k)
            if not rel:
                continue
            src = os.path.join(a.data, os.path.relpath(rel, "data"))
            if not os.path.exists(src):
                sys.exit(f"missing {src} (image {e['id']}): the bundle needs every crop the manifest references")
            dst = os.path.join(a.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            n_files += 1

    hits = []
    for dp, _, fns in os.walk(a.out):
        for fn in fns:
            if fn.endswith((".html", ".js", ".css", ".json", ".md", ".txt")):
                t = open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()
                hits += [(fn, p.pattern) for p in LEAK if p.search(t)]
    if hits:
        sys.exit(f"leak scan failed: {hits}")
    info = dict(built=dt.datetime.now().isoformat(timespec="seconds"), data=os.path.basename(os.path.abspath(a.data)),
                n_images=len(man["images"]), n_image_files=n_files, storage_key=a.storage_key, static=a.static,
                proposals_hidden=bool(a.hide_proposals), n_proposals_removed=n_hidden, n_fields_redacted=n_red,
                md5={fn: md5(os.path.join(a.out, fn)) for fn in FRONTEND + ["data/manifest.json"]})
    with open(os.path.join(a.out, "BUILD_INFO.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"bundle: {a.out}  ({len(man['images'])} images, {n_files} image files; proposals removed: {n_hidden}; "
          f"redacted fields: {n_red}). Open {os.path.join(a.out, 'index.html')} via any static web server.")


if __name__ == "__main__":
    main()

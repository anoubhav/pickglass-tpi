#!/usr/bin/env python3
"""Optional local server for the annotator: static files + autosave to disk.

    python3 server.py                 # http://127.0.0.1:8765/
    python3 server.py --open          # ... and open it in the default browser
    python3 server.py --port 9000     # (browser storage is per port; the disk autosave is not)
    python3 server.py --saved-dir /tmp/x   # write autosaves elsewhere (testing)
    python3 server.py --demo --open   # the 3 synthetic practice fabrics in demo/data (port 8766)
    python3 server.py --data-dir DIR  # any folder with manifest.json + crops/ crops_enh/ thumbs/

Behaves exactly like `python3 -m http.server` for static files (the app works with
either), and additionally implements:
    GET  /api/ping                      -> {"ok": true}
    POST /api/save   (export JSON body) -> MERGED into data/saved/annotations_<annotator>.json
                                           per record, newer timestamp wins, so a stale
                                           second tab or a fresh browser can never
                                           overwrite newer work (+ a timestamped copy in
                                           data/saved/backups/ at most every 10 minutes)
    GET  /api/load?annotator=<name>     -> the saved JSON (empty record list if none)
so a cleared browser cache or a different port never loses work. Standard library only;
binds to 127.0.0.1 so it is not reachable from other machines.
"""
import argparse
import datetime as dt
import errno
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # served as /data/ (--data-dir / --demo)
SAVED = os.path.join(HERE, "data", "saved")
BACKUPS = os.path.join(SAVED, "backups")
BACKUP_EVERY_S = 600
MAX_BODY = 50 * 1024 * 1024
_last_backup = {}
_lock = threading.Lock()


def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s or "anonymous").strip())[:40]
    return s or "anonymous"


def _ts(rec):
    """Record timestamp (ISO 8601 from JS toISOString) -> epoch seconds; 0 if missing."""
    t = rec.get("timestamp") if isinstance(rec, dict) else None
    if not t:
        return 0.0
    try:
        return dt.datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _pristine(rec):
    """Opened but never edited (same rule as core.js isPristine; legacy records: False)."""
    return rec.get("n_edits") == 0 and not rec.get("done")


def merge_saved(old, new):
    """Per-record merge of two export dicts, same rule as core.js mergeRecords: an untouched
    record never replaces an edited one; otherwise the newer timestamp wins, ties go to `new`."""
    by_id = {}
    for r in (old or {}).get("records", []) or []:
        if isinstance(r, dict) and "id" in r:
            by_id[str(r["id"])] = r
    for r in new.get("records", []):
        if not (isinstance(r, dict) and "id" in r):
            continue
        k = str(r["id"])
        o = by_id.get(k)
        if o is None:
            by_id[k] = r
        elif _pristine(r) != _pristine(o):
            if _pristine(o):
                by_id[k] = r
        elif _ts(r) >= _ts(o):
            by_id[k] = r
    out = dict(new)
    out["records"] = [by_id[k] for k in sorted(by_id, key=lambda x: (len(x), x))]
    out["n_records"] = len(out["records"])
    out["n_done"] = sum(1 for r in out["records"] if r.get("done"))
    return out


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        """/data/... comes from DATA (default <tool>/data), everything else from the tool folder."""
        p = urllib.parse.urlparse(path).path
        if p.startswith("/data/") and os.path.abspath(DATA) != os.path.join(HERE, "data"):
            rel = os.path.normpath(urllib.parse.unquote(p[len("/data/"):])).lstrip(os.sep)
            if rel.startswith(".."):
                return os.path.join(DATA, "__forbidden__")
            return os.path.join(DATA, rel)
        return super().translate_path(path)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/ping":
            return self._json(200, {"ok": True})
        if u.path == "/api/load":
            name = safe_name(urllib.parse.parse_qs(u.query).get("annotator", [""])[0])
            p = os.path.join(SAVED, f"annotations_{name}.json")
            if not os.path.exists(p):  # 200 + empty set keeps the browser console clean
                return self._json(200, {"annotator": name, "records": []})
            with open(p) as f:
                return self._json(200, json.load(f))
        return super().do_GET()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/save":
            return self._json(404, {"error": "unknown endpoint"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BODY:
            return self._json(400, {"error": "bad body size"})
        try:
            data = json.loads(self.rfile.read(n))
            assert isinstance(data, dict) and isinstance(data.get("records"), list)
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"invalid JSON: {e}"})
        name = safe_name(data.get("annotator"))
        with _lock:
            os.makedirs(BACKUPS, exist_ok=True)
            p = os.path.join(SAVED, f"annotations_{name}.json")
            old = None
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        old = json.load(f)
                except Exception:  # noqa: BLE001 -- corrupt file: keep it aside, never lose it
                    os.replace(p, p + f".corrupt-{int(time.time())}")
            data = merge_saved(old, data)
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, p)
            if time.time() - _last_backup.get(name, 0) > BACKUP_EVERY_S:
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(os.path.join(BACKUPS, f"annotations_{name}_{stamp}.json"), "w") as f:
                    json.dump(data, f)
                _last_backup[name] = time.time()
        return self._json(200, {"ok": True, "n_records": data["n_records"], "n_done": data["n_done"]})

    def end_headers(self):
        # never cache the app or the manifest (refreshed proposals must show up)
        if not self.path.startswith("/data/crops") and not self.path.startswith("/data/thumbs"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ""
        if "/api/" in first or "favicon.ico" in first:
            return  # keep the console quiet during autosaves
        super().log_message(fmt, *args)


def main():
    global SAVED, BACKUPS, DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--saved-dir", default=SAVED, help="where autosaves go (default: data/saved)")
    ap.add_argument("--open", action="store_true", help="open the app in the default browser")
    ap.add_argument("--data-dir", default=None, help="serve /data/ from this folder (manifest.json, crops/, ...)")
    ap.add_argument("--demo", action="store_true", help="the synthetic demo set in demo/data (own port 8766)")
    a = ap.parse_args()
    if a.demo:
        a.data_dir = a.data_dir or os.path.join(HERE, "demo", "data")
        a.port = 8766 if a.port == 8765 else a.port   # browser storage is per port: demo never mixes
    if a.data_dir:
        DATA = os.path.abspath(a.data_dir)
        if a.saved_dir == SAVED:
            a.saved_dir = os.path.join(DATA, "saved")
    SAVED = os.path.abspath(a.saved_dir)
    BACKUPS = os.path.join(SAVED, "backups")
    if not os.path.exists(os.path.join(DATA, "manifest.json")):
        print(f"{DATA}/manifest.json is missing -- run prepare.py first (or use --demo)", file=sys.stderr)
        return 1
    if not os.path.isdir(os.path.join(DATA, "crops")):
        print(f"note: {DATA}/crops/ is missing: the photos are released after review; "
              "try --demo for the synthetic practice set", file=sys.stderr)
    try:
        httpd = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=HERE))
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"Port {a.port} is busy (is the annotator already running? then just open "
                  f"http://{a.host}:{a.port}/). Otherwise use --port 8766; your disk autosave is "
                  f"restored on any port.", file=sys.stderr)
            return 1
        raise
    url = f"http://{a.host}:{a.port}/"
    print(f"Annotator: {url}   (autosave -> {SAVED})   Ctrl-C to stop", flush=True)
    if a.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

# annotation_tool/kit: run an annotation job with the tool

Generic, standard-library helpers (Pillow / OpenCV only where stated) for hiring, screening and checking annotators. They contain no names, hosting details, answer keys or pay. The protocol they implement is described in [docs/annotation_protocol.md](../../docs/annotation_protocol.md).

| Script | What it does |
|---|---|
| `build_bundle.py` | static, browser-only copy of the tool for one dataset (`--hide-proposals` for from-scratch counting, `--storage-key` to keep bundles apart on one web origin, `--static` for a static host, `--redact-ids` for a blind subset); refuses to finish if a text file contains a local path |
| `make_practice_set.py` | (re)generates the synthetic practice fabrics of `annotation_tool/demo/data` (or any other seeds) with their true counts (numpy, OpenCV) |
| `score_screening.py` | scores a candidate's screening reply (or the screening bundle's export) against YOUR private reference counts; pass rule: every number within 10 % (best H/V pairing per photo) |
| `check_export.py` | checks each annotator's export before scoring: one name per export, known ids, coverage of the scene representatives, the tool's completeness rule, no machine dots on screen, records opened before a protocol switch, active time; `--second` checks a second annotator's export on its own and reports the overlap (scenes done by both, distinct names) |
| `show_dots.py` | draws an annotator's exact dots on the crops and zooms in on every probable missed yarn and double click, for feedback (Pillow) |

Typical job:

```bash
# 1. public practice bundle and screening bundle (two photos with trusted, non-public counts)
python3 annotation_tool/kit/build_bundle.py --data annotation_tool/demo/data --out bundles/practice --storage-key practice --static
python3 annotation_tool/kit/build_bundle.py --data my_screening_data --out bundles/screen --storage-key screen --static --hide-proposals
# 2. score a candidate (key file stays private)
python3 annotation_tool/kit/score_screening.py --key private/screen_key.json --reply "Photo 1: H = 71, V = 58  Photo 2: H = 41, V = 44"
# 3. main bundles, counted from scratch: one per annotator, each under its own storage key and unguessable path
python3 annotation_tool/kit/build_bundle.py --data annotation_tool/data --out bundles/main_A1 --hide-proposals --static --storage-key A1
python3 annotation_tool/kit/build_bundle.py --data annotation_tool/data --out bundles/main_A2 --hide-proposals --static --storage-key A2
# 4. after the first 10 photos and at delivery (each annotator's files on their own; --second adds the overlap)
python3 annotation_tool/kit/check_export.py annotations_A1_*.json annotations_A1_*.csv --second annotations_A2_*.json annotations_A2_*.csv
python3 annotation_tool/kit/show_dots.py annotations_A1_*.json annotation_tool/data/crops feedback_A1.jpg
```

Host each bundle as static files (any web server or object store; `python3 -m http.server` in the bundle folder for a local test) and always link to `.../index.html`. Bundles on one web origin share the browser's storage unless their storage keys differ, so give every annotator their own key (and never open two annotators' links in one browser profile when testing). If the photos are confidential, keep the screening and main bundles under separate unguessable paths or behind an access control, and switch them off after the job.

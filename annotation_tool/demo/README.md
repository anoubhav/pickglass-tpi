# Demo dataset: three synthetic practice fabrics

`data/` is a complete annotation-tool dataset (manifest + rectified crops) of three **synthetic** fabrics made with
`cnn/synth.py` (seeds 986514, 969471, 830348; ids 801-803). No real photo is involved. It is the practice set of the
paper's job post.

```bash
python3 annotation_tool/server.py --demo --open      # http://127.0.0.1:8766/ ; autosave -> demo/data/saved/
```

Regenerate it with `python3 annotation_tool/kit/make_practice_set.py --out <dir>` (JPEG bytes can differ across
OpenCV builds; the counts cannot).

True counts (rounded; within 1 to 2 threads is fine, the generator varies the spacing slightly across the window):

| Photo | Fabric | H line (vertical yarns) | V line (horizontal yarns) | Weave |
| --- | --- | --- | --- | --- |
| 1 of 3 (#801) | plain weave, blue | 63 | 67 | plain (key 1) |
| 2 of 3 (#802) | twill, teal | 64 | 64 | twill 2/1 (key 2) |
| 3 of 3 (#803) | two-colour plain weave, gold and dark blue | 49 | 66 | plain (key 1) |

Typical mistakes: about half on #803 (one dot per light-and-dark pair: they are two yarns); a half or a third on #802
(one dot per diagonal stripe); a few too many (two dots on one yarn, or a dot on the black frame at the edge of #803).
All three open in blind mode with a purple badge: there are no starting dots.

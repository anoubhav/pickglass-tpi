"""provenance_scan.py -- read-only provenance audit of the 150 original photos (WVAQ 2027).

Reads  $PICKGLASS_NOTEBOOKS/final_image_set/{0..149}.jpeg  (never writes there)
and    annotation_tool/data/manifest.json (window polygons)
Writes camera_aug/originals_provenance.json

Checks per file: real container format (some '.jpeg' files are PNG), EXIF presence, JPEG quantisation
tables (compared with libjpeg/Pillow quality-75 tables), chroma subsampling, baseline/progressive,
8x8 blockiness of PNGs (evidence of an earlier JPEG encode), pick-glass window size in pixels.
Then groups near-duplicate photos: high-pass (sigma 3 px) normalised cross-correlation of 256x192
grey thumbnails (both 0 and 180 deg, portrait images rotated to landscape); pairs with r >= 0.99 are
joined (union-find).  A finer check compares each 4608x3456 image, area-downsampled x4, against its
1152x864 partner at full resolution (sigma 1 px high-pass).

Run:  <venv2>/bin/python camera_aug/provenance_scan.py
"""
import hashlib, io, itertools, json, math, os, collections
import numpy as np, cv2
from PIL import Image

ORIG = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "photos")), "final_image_set", "")
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "annotation_tool", "data", "manifest.json")
OUT = os.path.join(HERE, "originals_provenance.json")

def q75_tables():
    b = io.BytesIO(); Image.new("RGB", (16, 16)).save(b, "JPEG", quality=75)
    q = Image.open(io.BytesIO(b.getvalue())).quantization
    return [list(q[0]), list(q[1])]

def blockiness(gray):
    g = np.abs(np.diff(gray.astype(float), axis=1)).mean(0)
    i = np.arange(len(g))
    return float(g[i % 8 == 7].mean() / g[i % 8 != 7].mean())

def hp_thumb(path):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a.shape[0] > a.shape[1]: a = np.rot90(a)
    a = cv2.resize(a, (256, 192), interpolation=cv2.INTER_AREA).astype(np.float32)
    h = a - cv2.GaussianBlur(a, (0, 0), 3)
    return (h - h.mean()) / (h.std() + 1e-6)

def main():
    man = json.load(open(MANIFEST))
    poly = {im["id"]: im["polygon"] for im in man["images"]}
    iaa = man.get("iaa_subset", [])
    ref = q75_tables()
    files = {}
    for i in range(150):
        p = f"{ORIG}{i}.jpeg"
        im = Image.open(p)
        raw = open(p, "rb").read()
        r = dict(id=i, container=im.format, size=list(im.size), bytes=len(raw),
                 md5=hashlib.md5(raw).hexdigest(), exif_tags=len(im.getexif()))
        if im.format == "JPEG":
            q = im.quantization
            r["q_tables_equal_libjpeg_q75"] = [list(q[0]) == ref[0], list(q[1]) == ref[1]]
            r["sampling"] = [(l[1], l[2]) for l in im.layer]
            r["progressive"] = bool(im.info.get("progressive") or im.info.get("progression"))
            r["jfif"] = "jfif" in im.info
        else:
            r["blockiness_8x8"] = round(blockiness(np.asarray(im.convert("L"))), 3)
        P = poly[i]; side = sum(math.dist(P[k], P[(k + 1) % 4]) for k in range(4)) / 4
        r["window_side_px"] = round(side, 1)
        r["px_per_inch"] = round(side, 1)            # the window is 1 inch
        r["window_frac_long_side"] = round(side / max(im.size), 4)
        files[i] = r
    T = {i: hp_thumb(f"{ORIG}{i}.jpeg") for i in range(150)}
    pairs = []
    for i, j in itertools.combinations(range(150), 2):
        c = max(float((T[i] * T[j]).mean()), float((T[i] * np.rot90(T[j], 2)).mean()))
        if c >= 0.99: pairs.append((i, j, round(c, 4)))
    par = list(range(150))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j, _ in pairs: par[f(i)] = f(j)
    groups = collections.defaultdict(list)
    for i in range(150): groups[f(i)].append(i)
    groups = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    # fine check 4608x3456 vs 1152x864 partners
    def hp1(a): b = a - cv2.GaussianBlur(a, (0, 0), 1.0); return (b - b.mean()) / (b.std() + 1e-6)
    fine = []
    for i, j, _ in pairs:
        si, sj = sorted(files[i]["size"]), sorted(files[j]["size"])
        if {tuple(si), tuple(sj)} == {(3456, 4608), (864, 1152)}:
            big, small = (i, j) if max(files[i]["size"]) == 4608 else (j, i)
            B = cv2.imread(f"{ORIG}{big}.jpeg", 0).astype(np.float32)
            S = cv2.imread(f"{ORIG}{small}.jpeg", 0).astype(np.float32)
            B4 = cv2.resize(B, (B.shape[1] // 4, B.shape[0] // 4), interpolation=cv2.INTER_AREA)
            best = max((float((hp1(np.rot90(B4, k))[8:-8, 8:-8] * hp1(S)[8:-8, 8:-8]).mean()), k)
                       for k in range(4) if np.rot90(B4, k).shape == S.shape)
            fine.append(dict(full_res=big, quarter_res=small, r_fine=round(best[0], 3), rot90_k=best[1]))
    md5groups = collections.defaultdict(list)
    for i, r in files.items(): md5groups[r["md5"]].append(i)
    exact = [g for g in md5groups.values() if len(g) > 1]
    gid = {i: k for k, g in enumerate(groups) for i in g}
    iaa_dup = [(a, b) for a, b in itertools.combinations(iaa, 2) if gid[a] == gid[b]]
    summary = dict(
        n_files=150, n_jpeg=sum(r["container"] == "JPEG" for r in files.values()),
        n_png_named_jpeg=sum(r["container"] == "PNG" for r in files.values()),
        n_with_exif=sum(r["exif_tags"] > 0 for r in files.values()),
        n_jpeg_libjpeg_q75=sum(r.get("q_tables_equal_libjpeg_q75") == [True, True] for r in files.values()),
        size_hist={f"{w}x{h}": n for (w, h), n in collections.Counter(tuple(r["size"]) for r in files.values()).items()},
        n_unique_scenes=len(groups), group_size_hist=dict(collections.Counter(len(g) for g in groups)),
        n_byte_identical_groups=len(exact), byte_identical_groups=exact,
        n_fullres_quarterres_pairs=len(fine),
        fine_corr_median=float(np.median([x["r_fine"] for x in fine])) if fine else None,
        iaa_subset=iaa, iaa_pairs_same_scene=iaa_dup,
        window_frac_long_side_median=float(np.median([r["window_frac_long_side"] for r in files.values()])),
    )
    json.dump(dict(summary=summary, groups=groups, coarse_pairs=pairs, fine_pairs=fine,
                   files=[files[i] for i in range(150)]), open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()

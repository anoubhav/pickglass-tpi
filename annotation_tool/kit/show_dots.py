#!/usr/bin/env python3
"""show_dots.py -- draw an annotator's exact dots on the rectified crops and zoom in on every suspicious spacing.

    python3 annotation_tool/kit/show_dots.py export.json annotation_tool/demo/data/crops review.jpg [--ids 801,802]

For each record: the crop with both guide lines and every dot (cyan = H line, pink = V line); then one zoom per gap
(a probably missed yarn: spacing > 1.6 x the median) and per double click (spacing < 0.5 x the median), boxed in red.
Use it to give a screened or newly hired annotator concrete feedback. Level guide lines only (a turned line is
drawn level). Needs Pillow.
"""
import argparse
import json
import os
import statistics as st

from PIL import Image, ImageDraw, ImageFont

CY, PK, YL, RD = (34, 211, 238), (244, 114, 182), (245, 197, 66), (239, 68, 68)


def problems(xs):
    xs = sorted(xs)
    g = [b - a for a, b in zip(xs, xs[1:])]
    if not g:
        return 0.0, [], []
    m = st.median(g)
    return m, [(a, b) for a, b, s in zip(xs, xs[1:], g) if s > 1.6 * m], \
        [(a, b) for a, b, s in zip(xs, xs[1:], g) if s < 0.5 * m]


def font(n):
    for f in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_dots(im, r, scale, x0=0, y0=0, rad=4):
    d = ImageDraw.Draw(im)
    W, H = im.size
    hy, vx = (r["h_line_y"] - y0) * scale, (r["v_line_x"] - x0) * scale
    d.line([(0, hy), (W, hy)], fill=YL, width=max(1, int(rad / 3)))
    d.line([(vx, 0), (vx, H)], fill=YL, width=max(1, int(rad / 3)))
    for x in r["h_dots"]:
        X = (x - x0) * scale
        d.ellipse([X - rad, hy - rad, X + rad, hy + rad], fill=CY, outline=(0, 0, 0))
    for y in r["v_dots"]:
        Y = (y - y0) * scale
        d.ellipse([vx - rad, Y - rad, vx + rad, Y + rad], fill=PK, outline=(0, 0, 0))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("crops")
    ap.add_argument("out")
    ap.add_argument("--ids", default=None, help="comma list of record ids (default: every record with dots)")
    a = ap.parse_args()
    d = json.load(open(a.export, encoding="utf-8"))
    want = {int(x) for x in a.ids.split(",")} if a.ids else None
    tiles, zooms, T = [], [], 700
    for r in d["records"]:
        if (want and r["id"] not in want) or not (r.get("h_dots") or r.get("v_dots")):
            continue
        img = Image.open(os.path.join(a.crops, f"{r['id']}.jpg")).convert("RGB")
        S = r["rect_size"]
        full = img.resize((T, T), Image.LANCZOS)
        draw_dots(full, r, T / S, rad=3)
        tiles.append((full, f"#{r['id']}: H line {len(r['h_dots'])} dots, V line {len(r['v_dots'])} dots"))
        for ax in ("h", "v"):
            m, gaps, dbl = problems(r[f"{ax}_dots"])
            for (p0, p1), what in [(g, "missed yarn?") for g in gaps] + [(g, "double click?") for g in dbl]:
                c = (p0 + p1) / 2
                cx, cy = (c, r["h_line_y"]) if ax == "h" else (r["v_line_x"], c)
                half = 3.2 * m
                x0, y0 = int(cx - half), int(cy - half)
                z = img.crop((x0, y0, int(cx + half), int(cy + half))).resize((T // 2, T // 2), Image.LANCZOS)
                sc = (T // 2) / (2 * half)
                dd = draw_dots(z, r, sc, x0, y0, rad=8)
                bx, by = (cx - x0) * sc, (cy - y0) * sc
                dd.rectangle([bx - 0.8 * m * sc, by - 0.8 * m * sc, bx + 0.8 * m * sc, by + 0.8 * m * sc],
                             outline=RD, width=4)
                zooms.append((z, f"#{r['id']}, {ax.upper()} line: {what}"))
    if not tiles:
        raise SystemExit("no record with dots")
    pad, head, f1, f2 = 16, 34, font(22), font(18)
    W = max(len(tiles) * (T + pad), len(zooms) * (T // 2 + pad)) + pad
    Hh = 50 + head + T + pad + (head + T // 2 + pad if zooms else 0)
    out_im = Image.new("RGB", (W, Hh), (255, 255, 255))
    dr = ImageDraw.Draw(out_im)
    dr.text((pad, 12), f"{d.get('annotator', '')}: every dot as placed in the tool (cyan = H line, pink = V line)",
            fill=(0, 0, 0), font=f1)
    y = 50
    for i, (im, t) in enumerate(tiles):
        x = pad + i * (T + pad)
        dr.text((x, y + 6), t, fill=(0, 0, 0), font=f2)
        out_im.paste(im, (x, y + head))
    y += head + T + pad
    for i, (im, t) in enumerate(zooms):
        x = pad + i * (T // 2 + pad)
        dr.text((x, y + 6), t, fill=RD, font=f2)
        out_im.paste(im, (x, y + head))
    out_im.save(a.out, quality=90)
    print("wrote", a.out, out_im.size, "| zooms:", [t for _, t in zooms])


if __name__ == "__main__":
    main()

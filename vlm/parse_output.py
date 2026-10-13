"""Robust parsing of raw VLM answers.

parse_counts(text) -> dict(vertical, horizontal, weave, method, ok)
    1. drop <think>...</think> blocks and markdown fences
    2. try every {...} object (last one first) with json.loads, then with a
       lenient repair (quote bare keys / bare enum values, single->double quotes,
       remove trailing commas); accept the last object that has a count key
    3. fallback regex on free text ("vertical_yarns: 52", "vertical yarns = 52",
       "52 vertical yarns" ...)
    Values given as ranges ("50-60") become the midpoint; method records the path.

parse_points(text) -> dict(n_points, points, method)
    Molmo2 '<points coords="1 1 234 567 2 345 678 ...">', Molmo-1
    '<point x=".." y="..">' / '<points x1=".." y1=".." ...>', Qwen
    '[{"point_2d": [x, y]}, ...]'.  Exact duplicate points are removed.
"""
from __future__ import annotations

import json
import re

KEYS_V = ("vertical_yarns", "vertical", "vertical_yarn", "warp", "ends")
KEYS_H = ("horizontal_yarns", "horizontal", "horizontal_yarn", "weft", "picks")
WEAVES = ("plain", "twill", "satin", "other")


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"^.*?</think>", " ", text, flags=re.S) if "</think>" in text else text
    return text.replace("```json", " ").replace("```", " ")


def _num(v):
    """int/float/'52'/'~52'/'50-60' -> float or None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _repair(s: str) -> str:
    s = s.replace("'", '"')
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_ ]*?)\s*:", r'\1"\2":', s)      # bare keys
    s = re.sub(r':\s*(plain|twill|satin|other)([^"\w]|$)', r': "\1"\2', s)    # bare enum
    s = re.sub(r":\s*(plain|twill|satin|other)\|[a-z|]+", r': "\1"', s)       # copied template
    s = re.sub(r",\s*([}\]])", r"\1", s)                                       # trailing commas
    return s


def _from_obj(d: dict):
    low = {str(k).strip().lower().replace(" ", "_"): v for k, v in d.items()}
    v = next((_num(low[k]) for k in KEYS_V if k in low), None)
    h = next((_num(low[k]) for k in KEYS_H if k in low), None)
    w = low.get("weave")
    w = str(w).strip().lower() if w is not None else None
    if w is not None:
        w = None if "|" in w else next((x for x in WEAVES if x in w), w)   # '|' = template echoed
    return v, h, w


def parse_counts(text: str) -> dict:
    out = dict(vertical=None, horizontal=None, weave=None, method="none", ok=False)
    if not text:
        return out
    t = _clean(text)
    objs = re.findall(r"\{[^{}]*\}", t)
    for raw in reversed(objs):
        for method, s in (("json", raw), ("json_repaired", _repair(raw))):
            try:
                d = json.loads(s)
            except Exception:
                continue
            if isinstance(d, dict):
                v, h, w = _from_obj(d)
                if v is not None or h is not None:
                    out.update(vertical=v, horizontal=h, weave=w, method=method,
                               ok=v is not None and h is not None)
                    return out
    # free-text fallback
    num = r"(\d+(?:\.\d+)?(?:\s*(?:-|to|–)\s*\d+(?:\.\d+)?)?)"
    pats_v = [rf"vertical[_ ]yarns?\"?\s*(?:[:=]|is|are|count(?:ed)?|≈|~)?\s*(?:approximately|about|around|~)?\s*{num}",
              rf"{num}\s+vertical\s+yarns"]
    pats_h = [rf"horizontal[_ ]yarns?\"?\s*(?:[:=]|is|are|count(?:ed)?|≈|~)?\s*(?:approximately|about|around|~)?\s*{num}",
              rf"{num}\s+horizontal\s+yarns"]

    def last(pats):
        hits = []
        for p in pats:
            hits += [(m.start(), m.group(1)) for m in re.finditer(p, t, flags=re.I)]
        return _num(sorted(hits)[-1][1]) if hits else None

    v, h = last(pats_v), last(pats_h)
    wm = re.findall(r"\b(plain|twill|satin)\b", t, flags=re.I)
    w = wm[-1].lower() if wm else None
    if v is not None or h is not None:
        out.update(vertical=v, horizontal=h, weave=w, method="regex", ok=v is not None and h is not None)
    else:
        out.update(weave=w)
    return out


# ----------------------------------------------------------------------------
# points
# ----------------------------------------------------------------------------
# closing quote optional: a list cut off by the token cap still yields the points emitted so far
_MOLMO2_COORDS = re.compile(r"<(?:points|tracks)[^>]*?coords=\"([0-9\t:;, .]+)(\"?)")
_MOLMO2_TRIPLE = re.compile(r"([0-9]+) ([0-9]{1,4}) ([0-9]{1,4})")
_MOLMO1_POINT = re.compile(r"<point\s+x=\"([\d.]+)\"\s+y=\"([\d.]+)\"")
_MOLMO1_POINTS = re.compile(r"x(\d+)\s*=\s*\"([\d.]+)\"\s*y\1\s*=\s*\"([\d.]+)\"")
_QWEN_POINT = re.compile(r"\"?point_2d\"?\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def parse_points(text: str) -> dict:
    pts, method = [], "none"
    if not text:
        return dict(n_points=0, points=[], method=method, truncated=False)
    t = _clean(text)
    truncated = False
    for m in _MOLMO2_COORDS.finditer(t):
        s = m.group(1)
        truncated = truncated or m.group(2) != '"'
        # strip a leading image/frame index ("1 1 234 567 2 ..." -> triples after the first number)
        for grp in re.split(r"[\t;]", s):
            grp = grp.strip()
            first = re.match(r"([0-9.]+)\s+(.*)$", grp)
            body = first.group(2) if first else grp
            pts += [(float(x), float(y)) for _, x, y in _MOLMO2_TRIPLE.findall(body)]
        method = "molmo2_coords"
    if not pts:
        pts = [(float(x), float(y)) for x, y in _MOLMO1_POINT.findall(t)]
        pts += [(float(x), float(y)) for _, x, y in _MOLMO1_POINTS.findall(t)]
        method = "molmo1" if pts else method
    if not pts:
        pts = [(float(x), float(y)) for x, y in _QWEN_POINT.findall(t)]
        method = "qwen_point_2d" if pts else method
    uniq = list(dict.fromkeys(pts))
    return dict(n_points=len(uniq), points=uniq, method=method, truncated=truncated)


if __name__ == "__main__":  # tiny self-test
    tests = [
        '{"vertical_yarns": 52, "horizontal_yarns": 48, "weave": "plain"}',
        "```json\n{vertical_yarns: 60, horizontal_yarns: 55, weave: twill}\n```",
        "Step 1 ... so about 70 vertical yarns. Horizontal yarns: approximately 64. Weave: plain",
        '{"vertical_yarns": "50-60", "horizontal_yarns": 40, "weave": "plain|twill|satin|other"}',
        "<think>hmm</think>{'vertical_yarns': 12, 'horizontal_yarns': 13, 'weave': 'satin',}",
    ]
    for s in tests:
        print(parse_counts(s))
    print(parse_points('<points coords="1 1 120 500 2 180 501 3 240 499">yarn</points>'))
    print(parse_points('[{"point_2d": [12, 500]}, {"point_2d": [30, 501]}]'))
    print(parse_points('<points x1="10.0" y1="50.0" x2="20.0" y2="50.0" alt="y">y</points>'))

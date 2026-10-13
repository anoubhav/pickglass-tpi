"""Qualitative examples for the VLM baselines: input crop + ground truth + raw model answers.

Writes figs/qual_{k}_{image}.png (panel) and figs/qual_{k}_{image}.txt (full raw answers)
for a few hand-picked legacy images.  Selection rationale is in EXAMPLES below.

    python make_figs.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_vlm as D                    # noqa: E402
from evaluate_vlm import LABEL, load_records   # noqa: E402
from gt_legacy import legacy_gt         # noqa: E402

FIGS = os.path.join(HERE, "figs")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

# (image, caption, [(model, set, view, prompt_id)], molmo point overlay view or None)
EXAMPLES = [
    ("r7", "Plain weave, GT 42 ends x 65 picks: the best frontier model is within 2 TPI; others under-count",
     [("gpt-6.1-sol", "legacy", "crop", "P1"), ("gemini-3.8-flash", "legacy", "crop", "P1"),
      ("claude-fable-5.1", "legacy", "crop", "P1"), ("gemma-4-26b-a4b", "legacy", "crop", "P1"),
      ("molmo2_8b", "legacy", "crop", "P1")], None),
    ("a1", "2/1 twill; GT warp '26 + 25*2 = 76' is formula-derived, weft 64. GPT matches the weft exactly; its warp 51 = 26 + 25",
     [("gpt-6.1-sol", "legacy", "crop", "P1"), ("gpt-6.1-sol", "legacy", "qc", "P3"),
      ("gemini-3.1-pro", "legacy", "crop", "P1"), ("qwen3.8-max-prime", "legacy", "crop", "P1")], None),
    ("a4", "Dense plain weave, GT 115 (range 110-120) x 73: Molmo2's points (yellow ticks) are an evenly spaced template, not the yarns",
     [("molmo2_8b", "legacy", "guide_h", "P4h"), ("molmo2_8b", "legacy", "qc_guide_h", "P4h"),
      ("molmo2_8b", "legacy", "crop", "P1"), ("gpt-6.1-sol", "legacy", "crop", "P1"),
      ("gemini-3.1-pro", "legacy", "crop", "P1")], "guide_h"),
    ("a5", "Plain weave, GT 47 x 49, step-by-step prompt (P2): Qwen3.8-27B enumerates until the token cap; GPT-6.1-sol is within 1 TPI",
     [("qwen3.8-27b", "legacy", "crop", "P2"), ("gemini-3.8-flash", "legacy", "crop", "P2"),
      ("gpt-6.1-sol", "legacy", "crop", "P2")], None),
]


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def find(recs, model, st, img, view, pid):
    for r in recs:
        if (r["short"] == model and r["set"] == st and str(r["image_id"]) == img and r["view"] == view
                and r["prompt_id"] == pid and r.get("repeat", 0) == 0):
            return r
    return None


def summarize(r):
    p = r["parsed"]
    if r["prompt_id"].startswith("P4"):
        k = 4 if r["view"].startswith("qc") else 1
        tr = " (cut off at the token cap)" if p.get("truncated") else ""
        return f"{p.get('n_points', 0)} points{tr}" + (f" x4 = {4 * p.get('n_points', 0)}" if k == 4 else "")
    v, h = p.get("vertical"), p.get("horizontal")
    if v is None or h is None:
        return f"PARSE FAIL (finish={r.get('finish_reason')})"
    if r["view"] == "qc":
        return f"{v:g} x {h:g} per 1/4 in -> {4 * v:g} x {4 * h:g} per in, weave={p.get('weave')}"
    return f"{v:g} x {h:g}, weave={p.get('weave')}"


def main():
    os.makedirs(FIGS, exist_ok=True)
    recs = load_records()
    gt = legacy_gt()
    for k, (img, cap, picks, overlay) in enumerate(EXAMPLES, 1):
        crop = cv2.imread(os.path.join(D.LEGACY_CROP_DIR, f"{img}.png"))
        S = 640
        im = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
        if overlay:
            r = find(recs, "molmo2_8b", "legacy", img, overlay, "P4h" if overlay.endswith("_h") else "P4v")
            if r:
                im = D.guide_line_view(im, "h" if overlay.endswith("_h") else "v")
                for x, y in r["parsed"].get("points", []):      # thin ticks: dense lists stay readable
                    px, py = int(round(x / 1000 * S)), int(round(y / 1000 * S))
                    cv2.line(im, (px, py - 14), (px, py + 14), (0, 255, 255), 1)
        pil_img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        W, H = 1500, 720
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(pil_img, (20, 60))
        d = ImageDraw.Draw(canvas)
        g = gt.get(img)
        d.text((20, 14), f"{img}  -  {cap}", fill="black", font=font(20, True))
        x0, y = 690, 70
        if g:
            d.text((x0, y), f"Ground truth: warp (ends) {g['warp']:g}, weft (picks) {g['weft']:g}, "
                            f"weave {g['weave_name']}", fill=(0, 90, 0), font=font(19, True))
            y += 36
        txt_out = [f"{img}: {cap}", f"GT: {g}" if g else "GT: none", ""]
        for model, st, view, pid in picks:
            r = find(recs, model, st, img, view, pid)
            if r is None:
                continue
            head = f"{LABEL[model]}  [{pid if not pid.startswith('P4') else ('P4q' if view.startswith('qc') else 'P4')}, {view}]"
            d.text((x0, y), head, fill="black", font=font(17, True))
            y += 24
            d.text((x0, y), "parsed: " + summarize(r), fill=(150, 0, 0), font=font(16))
            y += 22
            raw = " ".join((r["raw"] or "").split())
            snippet = raw if len(raw) <= 230 else raw[:110] + " ... " + raw[-115:]
            for line in textwrap.wrap(snippet, 92)[:4]:
                d.text((x0, y), line, fill=(70, 70, 70), font=font(14))
                y += 18
            y += 12
            txt_out += [head, "parsed: " + summarize(r), f"finish={r.get('finish_reason')} seconds={r.get('seconds')} "
                        f"usd={r.get('cost_usd')}", "RAW:", r["raw"] or "", "-" * 60]
        out = os.path.join(FIGS, f"qual_{k}_{img}")
        canvas.save(out + ".png")
        open(out + ".txt", "w").write("\n".join(txt_out))
        print("wrote", out + ".png")


if __name__ == "__main__":
    main()

"""Versioned prompts for the VLM thread-counting baseline.

Every prompt has an id and a version; both are written into each raw-output record
so that results can always be traced back to the exact wording.  Never edit a
prompt in place: add a new version and bump PROMPT_SET_VERSION.

Prompt families
---------------
P1  direct      : whole rectified 1-inch window, one call, JSON answer.
P2  step-by-step: same input, asks the model to reason first, JSON on the last line.
P3  decomposed  : a 1/4-inch x 1/4-inch patch of the window (upsampled to the
                  model's input size); count in the patch, then
                    P3c  = 4 x count(central patch)            (spec: "x4")
                    P3s  = sum of counts over the 4 diagonal patches of a 4x4
                           grid (each covers a different quarter of x and y,
                           so the sum is a full-inch count without extrapolation)
P4  pointing    : a red guide line is drawn through the centre of the window;
                  the model points at every yarn crossing the line and we count
                  the points (Molmo2 native <points> format; Qwen point_2d JSON).
"""

PROMPT_SET_VERSION = "v1-2026-10-06"

JSON_SPEC = '{vertical_yarns:int, horizontal_yarns:int, weave: plain|twill|satin|other}'

PROMPTS = {
    # --- P1: the direct-count prompt, fixed before any run -----------------------
    "P1": dict(
        version="v1",
        max_tokens=200,
        text=("This is a magnified photo of woven fabric seen through a 1-inch pick glass. "
              "Count the number of vertical yarns crossing a horizontal line across the full window, "
              "and the number of horizontal yarns crossing a vertical line. "
              f"Answer JSON {JSON_SPEC}"),
    ),
    # --- P2: chain-of-thought / step-by-step -----------------------------------
    "P2": dict(
        version="v1",
        max_tokens=1200,
        text=("This is a magnified photo of woven fabric seen through a 1-inch pick glass; "
              "the image shows exactly a 1 inch x 1 inch square of the fabric.\n"
              "Think step by step:\n"
              "1. Identify the weave structure (plain, twill, satin or other) and the two yarn directions.\n"
              "2. Pick a small region (for example one tenth of the image width), count the vertical yarns "
              "in it carefully, and scale up to the full width. Do the same for the horizontal yarns "
              "along the full height.\n"
              "3. Sanity-check the two numbers against the apparent yarn spacing.\n"
              "Finally, on the last line output only the JSON object "
              '{"vertical_yarns": <int>, "horizontal_yarns": <int>, "weave": "plain|twill|satin|other"}, '
              "where vertical_yarns is the number of vertical yarns crossing a horizontal line across the "
              "full window and horizontal_yarns is the number of horizontal yarns crossing a vertical line."),
    ),
    # --- P3: quarter-inch patch -------------------------------------------------
    "P3": dict(
        version="v1",
        max_tokens=200,
        text=("This is a strongly magnified photo of woven fabric. The image shows a square patch of "
              "fabric that is exactly 1/4 inch wide and 1/4 inch tall. "
              "Count the number of vertical yarns crossing a horizontal line across the full patch, "
              "and the number of horizontal yarns crossing a vertical line across the full patch. "
              f"Answer JSON {JSON_SPEC}"),
    ),
    # --- P4: pointing on a guide line (two calls: horizontal line, vertical line) ---
    # v1.1: same wording, max_tokens 2500 -> 1024 because degenerate point lists
    # ran to the cap and took >12 min per image on the shared GPU.
    # v1.2: max_tokens 1536.  Molmo2 spends ~11.3 tokens per point, so 1024 only
    # allowed ~90 points (< GT up to 122); 1536 allows ~135.  v1/v1.1 records (a1, a2) all
    # stopped on their own except a1 qc_guide_v (quarter patch, degenerate list at the cap).
    "P4h": dict(
        version="v1.2",
        max_tokens=1536,
        text="Point to every vertical yarn where it crosses the red horizontal line.",
    ),
    "P4v": dict(
        version="v1.2",
        max_tokens=1536,
        text="Point to every horizontal yarn where it crosses the red vertical line.",
    ),
    # Qwen-style grounding variant of P4 (Qwen3-VL / Qwen3.5 emit point_2d JSON)
    "P4h_json": dict(
        version="v1",
        max_tokens=3000,
        text=("Locate every vertical yarn where it crosses the red horizontal line. "
              'Output a JSON list of points in the form [{"point_2d": [x, y]}, ...], '
              "one entry per crossing, with no other text."),
    ),
    "P4v_json": dict(
        version="v1",
        max_tokens=3000,
        text=("Locate every horizontal yarn where it crosses the red vertical line. "
              'Output a JSON list of points in the form [{"point_2d": [x, y]}, ...], '
              "one entry per crossing, with no other text."),
    ),
}


def get(pid: str) -> dict:
    p = dict(PROMPTS[pid])
    p["id"] = pid
    p["set_version"] = PROMPT_SET_VERSION
    return p


if __name__ == "__main__":
    for k, v in PROMPTS.items():
        print(f"[{k} {v['version']}] (max_tokens={v['max_tokens']})\n{v['text']}\n")

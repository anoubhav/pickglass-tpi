"""Shared bits for the Molmo2 synthetic-pointing fine-tune: prompts (verbatim project prompt set),
answer formatting in Molmo2's native point syntax, model loading."""
import os
import sys

FW = os.path.expanduser(("~/" + os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass")))
sys.path.insert(0, os.path.join(FW, "code", "mac_vlm"))
import prompts as PR                                    # noqa: E402  verbatim Mac prompt set
from parse_output import parse_counts, parse_points     # noqa: E402  verbatim Mac parsers

REPO_4B = "allenai/Molmo2-4B"
REV_4B = "042abfa7a38879a376cec03d949eff0aefaa0600"
LINE_COORD = 501            # round(1000 * (322 + 0.5) / 644): the guide line in Molmo2 0-999 units
LABEL = {"h": "vertical yarns", "v": "horizontal yarns"}
PROMPT_ID = {"h": "P4h", "v": "P4v"}


def coord(q):
    return min(999, max(0, int(round(1000 * q))))


def points_answer(positions, axis):
    """positions: crossing positions along the guide line (view-normalised 0..1), sorted.
    axis 'h' = horizontal guide line (x varies), 'v' = vertical guide line (y varies)."""
    parts = ["1"]
    for k, q in enumerate(positions, 1):
        x, y = (coord(q), LINE_COORD) if axis == "h" else (LINE_COORD, coord(q))
        parts.append(f"{k} {x:03d} {y:03d}")
    return f'<points coords="{" ".join(parts)}">{LABEL[axis]}</points>'


def count_answer(nv, nh, weave):
    return '{"vertical_yarns": %d, "horizontal_yarns": %d, "weave": "%s"}' % (nv, nh, weave)


def model_path(repo=REPO_4B, rev=REV_4B):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, token=False)

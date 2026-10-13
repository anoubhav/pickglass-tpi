#!/usr/bin/env python
"""Tests of the v2.2.2 example-figure selection (make_tool_examples.py): never an IAA-subset scene, nor another
photo of the same patch of fabric as one (SameSpot, needs cv2).

    <venv2>/bin/python tests/test_examples.py

Reads data/manifest.json, data/crops/ and ../methods/predictions_ours_150.csv; writes nothing (no figure is drawn).
"""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path.insert(0, TOOL)
import make_tool_examples as MX  # noqa: E402

MAN = MX.load_manifest()
PRED_PATH = os.environ.get("WVAQ_PRED", MX.PRED)
PREDS = MX.load_preds(PRED_PATH)
IAA = MX.iaa_ids(MAN)
IMAGES = sorted(MAN["images"], key=lambda e: int(e["id"]))
BY_ID = {int(e["id"]): e for e in IMAGES}
SS = MX.SameSpot(IMAGES, IAA)   # shared: features and results are cached
n = 0


def t(name):
    def deco(fn):
        global n
        fn()
        n += 1
        print("ok", n, name)
        return fn
    return deco


def scene_ids(ids):
    return {MX.scene_of(BY_ID[i]) for i in ids}


@t("the IAA subset is read from the manifest (20 scenes, the v2 list)")
def _():
    assert IAA == set(MAN["iaa_subset"]) and len(IAA) == 20
    assert IAA == {3, 7, 14, 18, 33, 34, 35, 38, 39, 49, 50, 55, 67, 76, 77, 89, 97, 99, 107, 124}


@t("excluded(): copies, IAA reps and copies of IAA scenes are excluded; non-IAA reps are not")
def _():
    for e in IMAGES:
        exp = (e.get("is_scene_rep") is False) or int(e["id"]) in IAA or MX.scene_of(e) in IAA
        assert MX.excluded(e, IAA) == exp, e["id"]
    copies_of_iaa = [e for e in IMAGES if not e["is_scene_rep"] and e["scene_id"] in IAA]
    assert copies_of_iaa and all(MX.excluded(e, IAA) for e in copies_of_iaa)


@t("pick_examples on the real data: ~4 non-IAA reps covering plain / twill / other at full and one quarter resolution")
def _():
    ex = MX.pick_examples(IMAGES, PREDS, IAA)
    ids = [i for i, _ in ex]
    assert 4 <= len(ids) <= 5, ex
    assert not (set(ids) & IAA) and not (scene_ids(ids) & IAA), ex
    assert all(BY_ID[i]["is_scene_rep"] for i in ids)
    res = {MX.resolution(BY_ID[i]) for i in ids}
    assert res == {"full", "quarter"}, res
    wc = {MX.weave_class(PREDS[i]["weave"]) for i in ids}
    assert {"plain", "twill", "other"} <= wc, wc
    assert ids == [i for i, _ in MX.pick_examples(IMAGES, PREDS, IAA)]   # deterministic


@t("pick_examples never returns an IAA scene even when IAA reps would rank best (doctored predictions / attention)")
def _():
    preds = copy.deepcopy(PREDS)
    images = copy.deepcopy(IMAGES)
    for e in images:   # make every IAA rep the most confident, 'low' attention candidate of its slot
        if int(e["id"]) in IAA:
            e["attention"] = "low"
            preds[int(e["id"])]["confidence"] = "1.0"
    ids = [i for i, _ in MX.pick_examples(images, preds, IAA)]
    assert ids and not (set(ids) & IAA) and not (scene_ids(ids) & IAA), ids
    # with an EMPTY subset the same doctored data would pick IAA reps -> the exclusion is what keeps them out
    ids_open = [i for i, _ in MX.pick_examples(images, preds, set())]
    assert set(ids_open) & IAA, ids_open
    # only IAA candidates for a slot -> the slot is skipped, never filled with an IAA scene
    only = [e for e in images if int(e["id"]) in IAA]
    assert MX.pick_examples(only, preds, IAA) == []


@t("pick_problem_photos: non-IAA reps only (the old bad_photos.png used IAA scenes 39, 124 and a copy of 77)")
def _():
    pr = MX.pick_problem_photos(IMAGES, PREDS, IAA)
    ids = [i for i, _, _ in pr]
    assert len(ids) == len(set(ids)) >= 4, pr
    assert not (set(ids) & IAA) and not (scene_ids(ids) & IAA) and all(BY_ID[i]["is_scene_rep"] for i in ids), pr
    for bad in (39, 124, 21):
        assert bad not in ids
    preds = copy.deepcopy(PREDS)
    for i in IAA:   # IAA reps with extreme measures are still never picked
        preds[i]["glare_frac"] = "1.0"; preds[i]["dark_frac"] = "1.0"; preds[i]["sharpness"] = "0.0"
    ids2 = [i for i, _, _ in MX.pick_problem_photos(IMAGES, preds, IAA)]
    assert not (set(ids2) & IAA), ids2


@t("build_index: refuses a figure that would show an IAA scene; the howto figure (8, 20) is kept only if non-IAA")
def _():
    ex = MX.pick_examples(IMAGES, PREDS, IAA)
    pr = MX.pick_problem_photos(IMAGES, PREDS, IAA)
    idx = MX.build_index(MAN, ex, pr)
    for f in idx["figures"]:
        assert f["scenes"] and not (set(f["scenes"]) & IAA) and not (scene_ids(f["scenes"]) & IAA), f
        assert f["src"].startswith("guide/") and f["src"] != "guide/annotation_guide.png"
    assert any(f["src"] == "guide/howto_count.png" for f in idx["figures"])
    try:
        MX.build_index(MAN, [(77, "x")], [])
    except ValueError:
        pass
    else:
        raise AssertionError("an IAA example must be refused")
    howto_iaa = dict(MX.HOWTO, scenes=[8, 77])
    idx2 = MX.build_index(MAN, ex, pr, howto=howto_iaa)
    assert not any(f["src"] == "guide/howto_count.png" for f in idx2["figures"])
    assert idx["iaa_subset_excluded"] == sorted(IAA)


@t("same patch: #8 (howto_count.png) and #98 are other photos of IAA scenes #99 / #124; never shown, howto left out")
def _():
    assert [j for _, j in SS.matches(BY_ID[8])] == [99], SS.matches(BY_ID[8])
    assert [j for _, j in SS.matches(BY_ID[98])] == [124], SS.matches(BY_ID[98])
    # the detector is not specific to IAA: a known non-IAA same-patch pair (27 / 103), and unrelated fabrics
    assert MX.SameSpot(IMAGES, {103})(BY_ID[27]) and not MX.SameSpot(IMAGES, {65})(BY_ID[110])
    ex = MX.pick_examples(IMAGES, PREDS, IAA, same_spot=SS)
    pr = MX.pick_problem_photos(IMAGES, PREDS, IAA, same_spot=SS)
    ids = [i for i, _ in ex] + [i for i, _, _ in pr]
    assert not any(SS(BY_ID[i]) for i in ids), ids
    assert 98 in [i for i, _, _ in MX.pick_problem_photos(IMAGES, PREDS, IAA)]   # without the check #98 is the blur pick
    assert 98 not in [i for i, _, _ in pr]
    idx = MX.build_index(MAN, ex, pr, same_spot=SS)
    assert not any(f["src"] == MX.HOWTO["src"] for f in idx["figures"]), "howto_count.png shows #8 = same patch as #99"
    assert all(not SS(BY_ID[int(s_)]) for f in idx["figures"] for s_ in f["scenes"])
    try:
        MX.build_index(MAN, [(8, "x")], [], same_spot=SS)
    except ValueError:
        pass
    else:
        raise AssertionError("a same-patch example must be refused")
    # doctored ranking: #8 the best plain full-resolution candidate -> picked without the check, never with it
    preds, images = copy.deepcopy(PREDS), copy.deepcopy(IMAGES)
    for e in images:
        if int(e["id"]) == 8:
            e["attention"] = "low"
    preds[8]["confidence"] = "9.9"
    preds[8]["weave"] = "plain"
    assert 8 in [i for i, _ in MX.pick_examples(images, preds, IAA)]
    assert 8 not in [i for i, _ in MX.pick_examples(images, preds, IAA, same_spot=SS)]


@t("SameSpot reads only crops (never data/saved) and flags no rep that matches no IAA rep (gap: <= 5 vs >= 12 inliers)")
def _():
    import cv2
    real, seen = cv2.imread, []

    def spy(path, *a, **k):
        seen.append(str(path))
        return real(path, *a, **k)

    cv2.imread = spy
    try:
        ss = MX.SameSpot(IMAGES, IAA)
        flagged = {int(e["id"]) for e in IMAGES if not MX.excluded(e, IAA) and ss(e)}
    finally:
        cv2.imread = real
    assert seen and all(os.sep + "crops" + os.sep in p_ for p_ in seen) and not any("saved" in p_ for p_ in seen), seen[:3]
    assert {8, 98} <= flagged and not flagged & {27, 65, 110, 93, 43, 48, 20, 9, 126, 103}, sorted(flagged)
    for e in IMAGES:
        if not MX.excluded(e, IAA) and int(e["id"]) not in flagged:
            assert max(ss.inliers(int(e["id"]), j) for j in ss.iaa_reps) <= 5, e["id"]


@t("guide/examples/examples.json (if generated) equals build_index() and lists no IAA scene; captions as required")
def _():
    p = os.path.join(TOOL, "guide", "examples", "examples.json")
    if not os.path.exists(p):
        print("   (skipped: no examples.json yet)")
        return
    with open(p) as f:
        disk = json.load(f)
    want = MX.build_index(MAN, MX.pick_examples(IMAGES, PREDS, IAA, same_spot=SS),
                          MX.pick_problem_photos(IMAGES, PREDS, IAA, same_spot=SS), same_spot=SS)
    no_px = [{k: v for k, v in f.items() if k != "px"} for f in disk["figures"]]
    assert dict(disk, figures=no_px) == json.loads(json.dumps(want)), "examples.json is stale: re-run make_tool_examples.py"
    for f in disk["figures"]:
        assert isinstance(f.get("px"), list) and len(f["px"]) == 2 and min(f["px"]) > 0, f
    for f in disk["figures"]:
        assert not (scene_ids(f["scenes"]) & IAA)
        assert not any(SS(BY_ID[int(s_)]) for s_ in f["scenes"]), f
        assert os.path.exists(os.path.join(TOOL, f["src"])), f["src"]
        if f["src"].startswith("guide/examples/example_"):
            assert f["caption"] == "machine proposals — shows the layout, not verified counts"
            # no weave guess is printed (the tool never shows the counter's weave guess)
            low = (f["title"] + " " + f["caption"]).lower()
            assert not any(w in low for w in ("plain", "twill", "satin")), f["title"]
    assert not any(f["src"] == "guide/howto_count.png" for f in disk["figures"])


@t("no annotation is read: the module never opens data/saved")
def _():
    import builtins
    real = builtins.open
    seen = []

    def spy(path, *a, **k):
        seen.append(str(path))
        return real(path, *a, **k)

    builtins.open = spy
    try:
        m = MX.load_manifest()
        p = MX.load_preds(PRED_PATH)
        MX.build_index(m, MX.pick_examples(m["images"], p, MX.iaa_ids(m)), MX.pick_problem_photos(m["images"], p, MX.iaa_ids(m)))
    finally:
        builtins.open = real
    assert seen and not any("saved" in s for s in seen), seen


print("\nall %d example-figure selection tests passed" % n)

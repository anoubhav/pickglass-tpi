#!/usr/bin/env python
"""Tests of the v2.2 'check carefully' attention levels (attention.py, prepare.py --attention-only).

    <venv2>/bin/python tests/test_attention.py

No labels are involved: attention is computed from predictions only. Writes nothing except temporary
directories under the system temp dir (set TMPDIR to keep them elsewhere).
"""
from __future__ import annotations

import builtins
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path.insert(0, TOOL)
import attention as AT  # noqa: E402
import prepare as PR  # noqa: E402

MAN = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
n = 0


def t(fn):
    global n
    fn()
    n += 1
    print("ok", n, fn.__doc__.strip().splitlines()[0])


def fresh_entries():
    return [dict(e) for e in MAN["images"]]


@t
def test_disagreement_rule():
    """disagreement(): one outlier ignored, 2:2 split at 2x = 41%, ours alone off, too few estimators"""
    d, k, c = AT.disagreement({"ours": 40, "log_gabor": 40, "gpt_p1": 80, "cnn_v5b0": 41})
    assert d < 2.6 and k == 4, d                                      # a single estimator at 2x does not count
    d, _, c = AT.disagreement({"ours": 40, "log_gabor": 40, "gpt_p1": 80, "cnn_v5b0": 80})
    assert abs(d - 100 * (math.sqrt(2) - 1)) < 1e-9 and abs(c - 40 * math.sqrt(2)) < 1e-9
    d, _, _ = AT.disagreement({"ours": 40, "log_gabor": 80, "gpt_p1": 80, "cnn_v5b0": None})
    assert abs(d - 100) < 1e-9                                         # the proposals alone are off by 2x
    d, _, _ = AT.disagreement({"ours": 40, "log_gabor": 20, "gpt_p1": 80})
    assert abs(d - 100) < 1e-9                                         # ours in the middle, but the others split
    assert AT.disagreement({"ours": 40, "log_gabor": None, "gpt_p1": float("nan"), "cnn_v5b0": 0})[0] is None
    assert AT.disagreement({"ours": None, "log_gabor": 30, "gpt_p1": 31})[0] is None
    d, _, _ = AT.disagreement({"ours": 50, "log_gabor": 50})
    assert d == 0


def synth_inputs(**kw):
    """A one-image input set (id 1) for assess()."""
    inp = {"est": {e: {} for e in AT.ESTIMATORS}, "conf": {1: 0.9}, "eta": {1: 0.5},
           "th": {"t_c": 0.469, "r_min": 0.002, "eta_p05": 0.009}, "strata": {1: {"S": 1900.0, "sat240": 0.0}},
           "missing": [], "md5": {}}
    for e, hv in kw.pop("est", {"ours": (50, 60), "log_gabor": (50, 60), "gpt_p1": (51, 61)}).items():
        inp["est"][e][1] = hv
    for k, v in kw.items():
        if k in ("conf", "eta"):
            inp[k][1] = v
        else:
            inp["strata"][1][k] = v
    return inp


@t
def test_points_and_levels():
    """assess(): each signal's points, the level rule and the reason order"""
    a = AT.assess(1, synth_inputs())
    assert (a["attention"], a["attention_score"], a["attention_reasons"]) == ("low", 0, [])
    a = AT.assess(1, synth_inputs(conf=0.3))
    assert a["attention"] == "medium" and a["attention_codes"] == ["low_confidence"]
    a = AT.assess(1, synth_inputs(sat240=0.02))
    assert a["attention"] == "medium" and a["attention_codes"] == ["glare"]
    a = AT.assess(1, synth_inputs(eta=0.005))
    assert a["attention"] == "medium" and a["attention_codes"] == ["low_resolvability"]
    a = AT.assess(1, synth_inputs(eta=0.001))
    assert a["attention"] == "high" and a["attention_score"] == 2
    a = AT.assess(1, synth_inputs(S=280.0))          # finer axis ~ 60.5/in -> 4.6 px per yarn
    assert a["attention"] == "medium" and a["attention_signals"]["ppy_bin"] == "4-5"
    a = AT.assess(1, synth_inputs(S=200.0))          # 3.3 px per yarn
    assert a["attention"] == "high" and a["attention_codes"] == ["fine_yarns"]
    est = {"ours": (50, 60), "log_gabor": (50, 75), "gpt_p1": (52, 75)}  # V: ours 20% below the others
    a = AT.assess(1, synth_inputs(est=est))
    assert a["attention"] == "medium" and a["attention_reasons"] == ["estimators disagree on V by 25%"], a
    est = {"ours": (50, 60), "log_gabor": (100, 75), "gpt_p1": (100, 75), "cnn_v5b0": (98, None)}
    a = AT.assess(1, synth_inputs(est=est, conf=0.2))
    assert a["attention"] == "high" and a["attention_score"] == 3, a
    assert a["attention_codes"] == ["disagree_h", "disagree_v", "low_confidence"], a
    assert a["attention_reasons"][0] == "estimators disagree on H by 98%", a
    assert a["attention_signals"]["n_h"] == 4 and a["attention_signals"]["n_v"] == 3
    # missing thresholds / signals: those checks are simply skipped
    inp = synth_inputs(conf=0.1, eta=0.0001)
    inp["th"] = {"t_c": None, "r_min": None, "eta_p05": None}
    assert AT.assess(1, inp)["attention"] == "low"


@t
def test_manifest_fields_and_no_counts():
    """compute(): fields on the 76 reps only, generic reasons, count-free signals, block counts"""
    entries = fresh_entries()
    meta = AT.compute(entries, MAN["iaa_subset"])
    reps = [e for e in entries if e.get("is_scene_rep")]
    assert len(reps) == 76 and meta["n_reps"] == 76 and sum(meta["levels"].values()) == 76
    for e in entries:
        assert ("attention" in e) == bool(e.get("is_scene_rep")), e["id"]
    fixed = set(AT.REASON_TEXT.values())
    for e in reps:
        for r in e["attention_reasons"]:
            assert r in fixed or re.fullmatch(r"estimators disagree on [HV] by \d+%", r), r
            assert not re.search(r"\d", re.sub(r"\d+%", "", r)), r
        s = e["attention_signals"]
        assert set(s) == {"estimators", "d_h", "d_v", "n_h", "n_v", "ppy_bin", "eta", "sat240", "conf"}
        # no count can hide in the signals: names, small integers, shares, a ratio and percentages only
        assert set(s["estimators"]) <= set(AT.ESTIMATORS) and s["estimators"][0] == "ours"
        assert 2 <= s["n_h"] <= 4 and 2 <= s["n_v"] <= 4
        assert 0 <= s["conf"] <= 1 and 0 <= s["sat240"] <= 1 and s["eta"] >= 0
        assert s["ppy_bin"] in ("<4", "4-5", ">=5") and s["d_h"] >= 0 and s["d_v"] >= 0
    assert meta["levels"] == {lv: sum(1 for e in reps if e["attention"] == lv) for lv in AT.LEVELS}
    assert meta["missing_inputs"] == [] and set(meta["inputs_md5"]) == set(AT.INPUTS)


@t
def test_deterministic_and_matches_manifest():
    """compute() twice -> identical; equals what the deployed manifest holds"""
    a, b = fresh_entries(), fresh_entries()
    ma, mb = AT.compute(a, MAN["iaa_subset"]), AT.compute(b, MAN["iaa_subset"])
    assert a == b and ma == mb
    if MAN.get("attention"):
        assert {k: v for k, v in MAN["attention"].items() if k != "updated"} == ma
        for e, m in zip(a, MAN["images"]):
            for k in AT.ATTN_FIELDS:
                assert e.get(k) == m.get(k), (e["id"], k)


@t
def test_reads_no_annotation():
    """compute() opens only the six prediction / descriptor inputs (never data/saved/, exports or labels)"""
    opened = []
    real_open = builtins.open

    def spy(path, *a, **k):
        opened.append(os.path.abspath(str(path)))
        return real_open(path, *a, **k)

    builtins.open = spy
    try:
        AT.compute(fresh_entries(), MAN["iaa_subset"])
    finally:
        builtins.open = real_open
    allowed = {os.path.abspath(os.path.join(AT.ROOT, p)) for p in AT.INPUTS}
    assert opened and set(opened) <= allowed, set(opened) - allowed
    src = real_open(os.path.join(TOOL, "attention.py")).read().split('"""', 2)[2]   # code, without the docstring
    for bad in ("data/saved", "SAVED", "propagate", "annotations_", "labels_per"):
        assert bad not in src, bad


@t
def test_prepare_attention_only():
    """prepare.py --attention-only: only additions, idempotent byte for byte, never reads data/saved/"""
    tmp = tempfile.mkdtemp()
    man_path = os.path.join(tmp, "manifest.json")
    base = dict(MAN)
    base.pop("attention", None)
    base["images"] = [{k: v for k, v in e.items() if k not in AT.ATTN_FIELDS} for e in MAN["images"]]
    json.dump(base, open(man_path, "w"), separators=(",", ":"))
    old_manifest, old_saved = PR.MANIFEST, PR.SAVED
    PR.MANIFEST, PR.SAVED = man_path, os.path.join(tmp, "no_such_saved_dir")
    try:
        PR.attention_only(None)
        b1 = open(man_path, "rb").read()
        PR.attention_only(None)
        b2 = open(man_path, "rb").read()
    finally:
        PR.MANIFEST, PR.SAVED = old_manifest, old_saved
    assert b1 == b2, "a second run must write an identical manifest"
    new = json.loads(b1)
    assert set(new) - set(base) == {"attention"}
    for k in base:
        if k != "images":
            assert new[k] == base[k], k
    for x, y in zip(base["images"], new["images"]):
        assert x == {k: v for k, v in y.items() if k not in AT.ATTN_FIELDS}
    assert not os.path.exists(os.path.join(tmp, "no_such_saved_dir"))
    # a stale field on a copy is removed, a rep's field refreshed
    new["images"][0]["attention"] = "high"
    rep = next(e for e in new["images"] if e.get("is_scene_rep"))
    level = rep["attention"]
    rep["attention"] = "bogus"
    json.dump(new, open(man_path, "w"))
    PR.MANIFEST = man_path
    try:
        PR.attention_only(None)
    finally:
        PR.MANIFEST = old_manifest
    again = json.load(open(man_path))
    assert ("attention" in again["images"][0]) == bool(again["images"][0].get("is_scene_rep"))
    assert next(e for e in again["images"] if e["id"] == rep["id"])["attention"] == level


@t
def test_missing_optional_inputs():
    """only the counter's CSV present: still computed (ours + log-Gabor), missing inputs listed"""
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "methods"))
    shutil.copy(os.path.join(AT.ROOT, AT.P_OURS), os.path.join(root, AT.P_OURS))
    entries = fresh_entries()
    meta = AT.compute(entries, MAN["iaa_subset"], root=root)
    assert set(meta["missing_inputs"]) == set(AT.INPUTS) - {AT.P_OURS}
    assert meta["cnn_valid_reps"] == 0 and sum(meta["levels"].values()) == 76
    for e in entries:
        if e.get("is_scene_rep"):
            assert e["attention_signals"]["estimators"] == ["ours", "log_gabor"]
            assert set(e["attention_codes"]) <= {"disagree_h", "disagree_v"}
    empty = tempfile.mkdtemp()
    try:
        AT.compute(fresh_entries(), [], root=empty)
    except SystemExit:
        pass
    else:
        raise AssertionError("the counter's predictions are required")


@t
def test_cnn_heldout_only():
    """the CNN enters only where the held-out file has a value (its fine-tuning scenes are blank)"""
    inp = AT.load_inputs()
    rows = {int(r["image_id"]): r for r in csv.DictReader(open(os.path.join(AT.ROOT, AT.P_CNN)))}
    entries = fresh_entries()
    AT.compute(entries, MAN["iaa_subset"])
    for e in entries:
        if not e.get("is_scene_rep"):
            continue
        valid = rows[e["id"]][AT.CNN_KEY + "__v"].strip() != ""
        assert ("cnn_v5b0" in e["attention_signals"]["estimators"]) == valid, e["id"]
        assert ("gpt_p1" in e["attention_signals"]["estimators"]) == (e["id"] in inp["est"]["gpt_p1"])


print(f"\nall {n} python attention tests passed")

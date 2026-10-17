"""fabric_groups_v2.py -- label-free re-shoot audit of the 76 scene representatives.

Why: tables/fabric_groups_candidate.csv (64 groups) merges only the 13 cross-scene pairs of cnn/selfcheck_aug.py
check E1 (band-pass NCC > 0.9). The annotator kit's same-patch matcher (freelancer/same_patch_pairs.json) compared
only non-IAA x IAA representatives and already found 11 more pairs that the candidate groups do not merge. Here the
SAME matcher is run over ALL 76 x 75 / 2 = 2,850 representative pairs, in both directions.

Matcher (unchanged, imported read-only): annotation_tool/make_tool_examples.py SameSpot -- SIFT (4,000 features) on
the central 70% of each 1-inch crop resized to 500 px with CLAHE, ratio test 0.75, RANSAC similarity
(cv2.estimateAffinePartial2D, 3 px), inliers counted only when the fitted scale is 0.8-1.25 (any rotation).
Pair score = max(inliers(a->b), inliers(b->a)). A pair is "the same fabric patch" when score >= SAME_SPOT_MIN (12),
the matcher's own pre-declared threshold. The all-pairs distribution decides whether that is defensible: it is
written to the provenance JSON (histogram, the largest score below the threshold, the smallest above it). On the
v2 data every score lies in {0, 2..5} or >= 16 (no pair in 6..15), so any threshold in 6..16 gives the same groups;
a stricter >= 100 alternative is reported as a sensitivity row.
Diagnostics (not used for the decision): aligned high-pass NCC of the crop pair under the RANSAC similarity,
rotation closure on every triangle of accepted pairs, and the frozen counter's own counts
(methods/predictions_ours_150.csv; machine output, no human count) on both scenes of each pair.

Groups: union-find over the accepted pairs and the 13 E1 pairs (all 13 are also accepted by the matcher); a group is
labelled by its smallest scene id (the candidate file's convention), so score150.py --groups reads it unchanged
(it uses only the columns scene_id and fabric_group; extra columns are ignored).

CNN split (cnn/runs_aug/pseudo_labels.csv, one split per scene): 'train' = the 28 pseudo-label training scenes;
'val' (10, selected the variant) and 'lowconf' (38) = the 48 held-out scenes Tab. 1's pseudo-label row is scored on.
Lists every accepted pair that crosses train / held-out and every held-out scene whose fabric group contains a
training scene.

No human count is read: data/saved/ and every annotation export are never opened.

Inputs (read-only): annotation_tool/data/manifest.json, annotation_tool/data/crops/*.jpg,
  annotation_tool/make_tool_examples.py, cnn/runs_aug/report/selfcheck.json (E1), cnn/runs_aug/pseudo_labels.csv,
  methods/predictions_ours_150.csv, paper/tables/fabric_groups_candidate.csv, freelancer/same_patch_pairs.json
  (reproduction check only).
Outputs: paper/tables/fabric_groups_v2.csv, paper/tables/fabric_groups_v2.json (provenance + analysis),
  paper/tables/src/same_patch_allpairs.csv (all 2,850 pairs).

Usage (any venv with numpy, pandas, opencv-python(-headless) >= 4.4):
  <venv>/bin/python tools/fabric_groups_v2.py            # from paper/; ~3 min on a laptop
Deterministic (OpenCV's RANSAC uses a fixed RNG state per call).

Provenance: the committed paper/tables/fabric_groups_v2.json records script_md5 632c2e4f873c71d2dfa3c88571097795,
the MD5 of the script that wrote it. This released copy differs from that script only in docstring and comment
wording (the code is unchanged), so its MD5, and the script_md5 a re-run writes, differ.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TOOL = os.path.join(ROOT, "annotation_tool")
P_MAN = os.path.join(TOOL, "data", "manifest.json")
P_MATCHER = os.path.join(TOOL, "make_tool_examples.py")
P_E1 = os.path.join(ROOT, "cnn", "runs_aug", "report", "selfcheck.json")
P_SPLIT = os.path.join(ROOT, "cnn", "runs_aug", "pseudo_labels.csv")
P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_CAND = os.path.join(PAPER, "tables", "fabric_groups_candidate.csv")
P_KIT = os.path.join(ROOT, "freelancer", "same_patch_pairs.json")
OUT_CSV = os.path.join(PAPER, "tables", "fabric_groups_v2.csv")
OUT_JSON = os.path.join(PAPER, "tables", "fabric_groups_v2.json")
OUT_PAIRS = os.path.join(PAPER, "tables", "src", "same_patch_allpairs.csv")
ALT_THRESHOLDS = (100,)       # sensitivity rows (stricter alternative threshold)
AGREE = 0.03                  # counter-consistency diagnostic: both axes within 3% (h/v swap allowed)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def union_groups(ids, pairs):
    par = {s: s for s in ids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in pairs:
        par[find(a)] = find(b)
    comp = {}
    for s in ids:
        comp.setdefault(find(s), []).append(s)
    lab = {}
    for members in comp.values():
        g = min(members)
        for s in members:
            lab[s] = g
    return lab


def main():
    import cv2
    if TOOL not in sys.path:
        sys.path.insert(0, TOOL)
    import make_tool_examples as M

    man = json.load(open(P_MAN, encoding="utf-8"))
    reps = sorted(int(e["id"]) for e in man["images"] if e.get("is_scene_rep"))
    assert len(reps) == 76, len(reps)
    assert all(int(e["scene_id"]) == int(e["id"]) for e in man["images"] if e.get("is_scene_rep"))
    ss = M.SameSpot(man["images"], set(man.get("iaa_subset") or []))
    T = int(M.SAME_SPOT_MIN)
    n = M.SAME_SPOT_PX

    def detail(a, b):
        """SameSpot.inliers(a, b) plus the fitted similarity (same code path; checked against it below)."""
        ka, da = ss._features(a)
        kb, db = ss._features(b)
        d = dict(good=0, ransac=0, scale=np.nan, angle=np.nan, inl=0, A=None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return d
        good = [m[0] for m in cv2.BFMatcher().knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        d["good"] = len(good)
        if len(good) < 8:
            return d
        pa = np.float32([ka[g.queryIdx].pt for g in good])
        pb = np.float32([kb[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is None or inl is None:
            return d
        sc = float(np.hypot(A[0, 0], A[1, 0]))
        d.update(ransac=int(inl.sum()), scale=sc, angle=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                 inl=int(inl.sum()) if 0.8 <= sc <= 1.25 else 0, A=A)
        return d

    HP = {}

    def hp(i):
        if i not in HP:
            im = cv2.imread(os.path.join(TOOL, ss.by_id[i]["crop"]), cv2.IMREAD_GRAYSCALE)
            im = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA))
            im = im.astype(np.float32)
            HP[i] = im - cv2.GaussianBlur(im, (0, 0), 8)
        return HP[i]

    centre = np.zeros((n, n), bool)
    centre[int(n * .15):int(n * .85), int(n * .15):int(n * .85)] = True

    def aligned_ncc(a, b, A):
        ones = cv2.warpAffine(np.ones((n, n), np.float32), A, (n, n), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        wb = cv2.warpAffine(hp(b), A, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        m = centre & (ones > 0.5)
        if m.sum() < 2000:
            return np.nan
        x, y = hp(a)[m], wb[m]
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))

    rows = []
    for a, b in itertools.combinations(reps, 2):
        d1, d2 = detail(a, b), detail(b, a)
        assert d1["inl"] == ss.inliers(a, b) and d2["inl"] == ss.inliers(b, a), (a, b)
        best = d1 if d1["inl"] >= d2["inl"] else d2
        rows.append(dict(scene_a=a, scene_b=b, inliers_ab=d1["inl"], inliers_ba=d2["inl"],
                         score=max(d1["inl"], d2["inl"]), ratio_matches_ab=d1["good"], ratio_matches_ba=d2["good"],
                         ransac_ab=d1["ransac"], ransac_ba=d2["ransac"],
                         scale_ab=round(d1["scale"], 4), angle_ab_deg=round(d1["angle"], 2),
                         scale_ba=round(d2["scale"], 4), angle_ba_deg=round(d2["angle"], 2),
                         aligned_hp_ncc=(round(aligned_ncc(a, b, d1["A"]), 3) if best is d1 and d1["inl"] else
                                         round(aligned_ncc(b, a, d2["A"]), 3) if d2["inl"] else np.nan)))
    P = pd.DataFrame(rows)

    # ---- reproduction of the kit's file (non-IAA query -> IAA rep, same direction)
    kit = json.load(open(P_KIT))
    repro = []
    for q, lst in kit["pairs"].items():
        for e in lst:
            repro.append(dict(query=int(q), iaa_rep=int(e["iaa_rep"]), kit=int(e["inliers"]),
                              now=int(ss.inliers(int(q), int(e["iaa_rep"])))))
    repro_ok = all(r["kit"] == r["now"] for r in repro)

    # ---- E1 pairs (candidate groups) and the candidate file
    e1 = sorted((min(int(q["scene_a"]), int(q["scene_b"])), max(int(q["scene_a"]), int(q["scene_b"])))
                for q in json.load(open(P_E1))["E1_fabric_near_duplicates_not_straddling"]["pairs"])
    cand = pd.read_csv(P_CAND).set_index("scene_id")["fabric_group"].to_dict()
    assert union_groups(reps, e1) == {int(k): int(v) for k, v in cand.items()}, "candidate file != union of E1 pairs"
    sc = P.set_index(["scene_a", "scene_b"])["score"]

    # ---- distribution and threshold
    hist = P.score.value_counts().sort_index()
    below = P.score[P.score < T]
    above = P.score[P.score >= T]
    gap = [int(below.max()) + 1, int(above.min()) - 1]
    acc = P[P.score >= T]
    acc_pairs = [(int(a), int(b)) for a, b in zip(acc.scene_a, acc.scene_b)]
    lab = union_groups(reps, sorted(set(acc_pairs) | set(e1)))

    # ---- diagnostics
    ours = pd.read_csv(P_OURS).set_index("id")

    def counter_reldiff(a, b):
        ha, va, hb, vb = (float(ours.at[i, c]) for i, c in ((a, "h_count"), (a, "v_count"), (b, "h_count"), (b, "v_count")))
        return min(max(abs(ha / hb - 1), abs(va / vb - 1)), max(abs(ha / vb - 1), abs(va / hb - 1)))
    acc = acc.assign(counter_max_reldiff=[round(counter_reldiff(a, b), 3) for a, b in acc_pairs])
    ang = {}
    for r in acc.itertuples():
        th = r.angle_ab_deg if r.inliers_ab >= r.inliers_ba else -r.angle_ba_deg
        ang[(r.scene_a, r.scene_b)], ang[(r.scene_b, r.scene_a)] = th, -th
    closure = []
    for a, b, c in itertools.combinations(reps, 3):
        if (a, b) in ang and (b, c) in ang and (a, c) in ang:
            e = (ang[(a, b)] + ang[(b, c)] - ang[(a, c)] + 180) % 360 - 180
            closure.append(dict(triangle=[a, b, c], closure_deg=round(float(e), 2)))
    sub = P[(P.score < T) & (P.score > 0)].sort_values("score", ascending=False)

    # ---- CNN pseudo-label split
    pl = pd.read_csv(P_SPLIT)
    split = {int(s): str(g.split.iloc[0]) for s, g in pl.groupby("scene")}
    assert pl.groupby("scene").split.nunique().max() == 1 and set(split) == set(reps)
    held = lambda s: split[s] != "train"   # noqa: E731

    def split_report(pairs, lab_):
        cross = [dict(pair=[a, b], score=int(sc[(a, b)]) if (a, b) in sc.index else None,
                      splits=[split[a], split[b]], held_out=[s for s in (a, b) if held(s)][0])
                 for a, b in pairs if held(a) != held(b)]
        members = {}
        for s, g in lab_.items():
            members.setdefault(g, []).append(s)
        aff = sorted(s for s in reps if held(s) and any(not held(t) for t in members[lab_[s]]))
        return dict(cross_pairs=cross,
                    affected_heldout=[dict(scene=s, split=split[s], fabric_group=lab_[s],
                                           train_scenes_in_group=sorted(t for t in members[lab_[s]] if not held(t)))
                                      for s in aff],
                    n_heldout=sum(held(s) for s in reps), n_heldout_clean=sum(held(s) for s in reps) - len(aff),
                    val_sharing_train=sorted(s for s in aff if split[s] == "val"))

    groups = {}
    for s in reps:
        groups.setdefault(lab[s], []).append(s)
    multi = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    out = dict(
        made=dt.datetime.now().isoformat(timespec="seconds"),
        script="paper/tools/fabric_groups_v2.py", script_md5=md5(os.path.abspath(__file__)),
        inputs={os.path.relpath(p, ROOT): md5(p) for p in (P_MAN, P_MATCHER, P_E1, P_SPLIT, P_OURS, P_CAND, P_KIT)},
        opencv=cv2.__version__, numpy=np.__version__,
        matcher=("annotation_tool/make_tool_examples.py SameSpot (imported, unchanged): SIFT 4000 on the central 70% "
                 "of each crop at 500 px with CLAHE, ratio 0.75, RANSAC similarity 3 px, scale 0.8-1.25; pair score = "
                 "max of both directions"),
        human_counts_read=False,
        n_reps=len(reps), n_pairs=len(P),
        kit_reproduction=dict(ok=repro_ok, n=len(repro), rows=repro),
        threshold=T, threshold_source="make_tool_examples.SAME_SPOT_MIN (pre-declared before this all-pairs run)",
        score_histogram={int(k): int(v) for k, v in hist.items()},
        largest_below=int(below.max()), smallest_at_or_above=int(above.min()), empty_gap=gap,
        n_accepted_pairs=len(acc_pairs), accepted_pairs=acc[["scene_a", "scene_b", "inliers_ab", "inliers_ba",
                                                             "angle_ab_deg", "aligned_hp_ncc",
                                                             "counter_max_reldiff"]].to_dict("records"),
        e1_pairs=e1, e1_all_accepted=all(p in set(acc_pairs) for p in e1),
        new_pairs_vs_candidate=[p for p in acc_pairs if cand[p[0]] != cand[p[1]]],
        n_groups=len(groups), n_groups_candidate=len(set(cand.values())), multi_scene_groups=multi,
        n_scenes_in_multi=sum(len(g) for g in multi),
        candidate_nested_in_v2=all(lab[a] == lab[b] for a in reps for b in reps if cand[a] == cand[b]),
        diagnostics=dict(
            scale_range_accepted=[float(acc[["scale_ab", "scale_ba"]].min().min()),
                                  float(acc[["scale_ab", "scale_ba"]].max().max())],
            aligned_hp_ncc_range_accepted=[float(acc.aligned_hp_ncc.min()), float(acc.aligned_hp_ncc.max())],
            triangle_rotation_closure=closure,
            counter_both_axes_within_3pct=int((acc.counter_max_reldiff <= AGREE).sum()),
            counter_pairs_off_by_more_than_10pct=int((acc.counter_max_reldiff > 0.10).sum()),
            sub_threshold_nonzero=sub[["scene_a", "scene_b", "inliers_ab", "inliers_ba", "scale_ab", "angle_ab_deg",
                                       "aligned_hp_ncc"]].to_dict("records")),
        cnn_split=dict(source="cnn/runs_aug/pseudo_labels.csv (split per scene; train = pseudo-label training scenes; "
                              "val + lowconf = the 48 held-out scenes of real150_aug_wide_heldout.csv)",
                       n={k: int(v) for k, v in pd.Series(split).value_counts().items()},
                       **split_report(sorted(set(acc_pairs) | set(e1)), lab)),
        sensitivity={},
    )
    for t in ALT_THRESHOLDS:
        pp = [(int(a), int(b)) for a, b in zip(P.scene_a[P.score >= t], P.scene_b[P.score >= t])]
        lt = union_groups(reps, sorted(set(pp) | set(e1)))
        out["sensitivity"][f"score>={t}"] = dict(n_pairs=len(pp), n_groups=len(set(lt.values())),
                                                 dropped_pairs=[p for p in acc_pairs if p not in set(pp)],
                                                 cnn=split_report(sorted(set(pp) | set(e1)), lt))

    # ---- write
    links = {s: [] for s in reps}
    for r in acc.itertuples():
        links[r.scene_a].append(f"{r.scene_b}:{r.inliers_ab}/{r.inliers_ba}")
        links[r.scene_b].append(f"{r.scene_a}:{r.inliers_ba}/{r.inliers_ab}")
    fg = pd.DataFrame(dict(scene_id=reps, fabric_group=[lab[s] for s in reps],
                           candidate_group=[int(cand[s]) for s in reps], cnn_split=[split[s] for s in reps],
                           group_size=[len(groups[lab[s]]) for s in reps],
                           same_patch_links=[";".join(links[s]) for s in reps]))
    fg.to_csv(OUT_CSV, index=False)
    P.to_csv(OUT_PAIRS, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"{len(P)} pairs; histogram {out['score_histogram']}")
    print(f"threshold {T}: {len(acc_pairs)} pairs, gap {gap}; {len(groups)} groups (candidate {len(set(cand.values()))})")
    print(f"kit reproduction ok: {repro_ok} ({len(repro)} rows); E1 all accepted: {out['e1_all_accepted']}")
    cs = out["cnn_split"]
    print("cross train/held-out pairs:", [c["pair"] for c in cs["cross_pairs"]])
    print("held-out scenes sharing a fabric group with a training scene:",
          [(a["scene"], a["split"], a["train_scenes_in_group"]) for a in cs["affected_heldout"]],
          f"-> {cs['n_heldout']} held out, {cs['n_heldout_clean']} clean")
    for k, v in out["sensitivity"].items():
        print(f"{k}: {v['n_pairs']} pairs, {v['n_groups']} groups, clean held-out {v['cnn']['n_heldout_clean']}, "
              f"affected {[a['scene'] for a in v['cnn']['affected_heldout']]}")
    print("wrote", os.path.relpath(OUT_CSV, ROOT), os.path.relpath(OUT_JSON, ROOT), os.path.relpath(OUT_PAIRS, ROOT))


if __name__ == "__main__":
    main()

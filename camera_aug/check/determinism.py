"""Skeptical check: re-render 2 profiles x 3 images, compare md5 with manifest and disk; read-only."""
import os, sys, json, hashlib, time
HERE = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "camera_aug")
sys.path.insert(0, HERE)
import build_stress as B
import camera_sim as CS
import pandas as pd

mode = sys.argv[1] if len(sys.argv) > 1 else "seq"
jobs_spec = [("legacy13", sid, cond, 0) for sid in ("a4", "r4", "r8") for cond in ("flagship", "budget_fullres_50mp")]
jobs_spec += [("real150", "0", "flagship", 0)]  # one real150 render as extra
out = []
for set_name, sid, cond, rep in jobs_spec:
    man = pd.read_csv(os.path.join(HERE, "stress", set_name, "manifest.csv"), dtype={"source_id": str})
    src = dict(B.sources(set_name))
    job = (set_name, sid, src[sid], cond, rep)
    t0 = time.time()
    row = B.render_one(job, write=False)
    m = man[(man.source_id == sid) & (man.condition == cond) & (man.rep == rep)].iloc[0]
    disk = hashlib.md5(open(os.path.join(HERE, m.path), "rb").read()).hexdigest()
    pj_same = json.loads(row["params_json"]) == json.loads(m.params_json)
    out.append(dict(set=set_name, sid=sid, cond=cond, rep=rep, md5_new=row["md5"], md5_manifest=m.md5, md5_disk=disk,
                    match_manifest=row["md5"] == m.md5, match_disk=row["md5"] == disk, params_same=pj_same,
                    seed=row["seed"], seed_manifest=m.seed, src_md5_same=row["source_md5"] == m.source_md5,
                    lf_h=row["label_factor_h"], lf_h_man=m.label_factor_h, secs=round(time.time() - t0, 1)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"determinism_{mode}.csv"), index=False)
print("ALL MATCH" if all(r["match_manifest"] and r["match_disk"] and r["params_same"] for r in out) else "MISMATCH")

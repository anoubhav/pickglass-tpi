"""Build RESULTS.md from RESULTS.template.md + results/summary.json + the raw Molmo2 logs.

Run after evaluate_vlm.py (e.g. once the local Molmo2 run has been resumed and finished):
    python evaluate_vlm.py && python build_results.py
Tables and all Molmo2 numbers are regenerated; the hand-written text lives in the template."""
import contextlib
import io
import json
import os
import sys

VLM = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(VLM, "RESULTS.template.md")
sys.path.insert(0, VLM)
import evaluate_vlm as E  # noqa: E402

s = json.load(open(os.path.join(VLM, "results", "summary.json")))
f = E.f
LBL = dict(E.LABEL)

# ---------------- Stage A table (same as the evaluator's printout) ----------------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    E.print_tables(s)
txt = buf.getvalue()
stageA = txt.split("## Stage A")[1].split("\n\n", 1)[1].split("\n\n")[0].strip()

# ---------------- resolution ablation ----------------
rows, seen = [], set()
for r in s["ablation_resolution"]:
    m = r["model"]
    if m not in seen:
        seen.add(m)
        rows.append(f"| {LBL.get(m, m)} | 1 | ≈ 1800 | {f(r['mae_best_full'])} | {f(r['mape_best_full'])} | "
                    f"{r['within10_full']}/{2 * r['n_gt_paired']} | {f(r['bias_full'])} | – |")
    extra = f" (parse fail {r['parse_fail_down']}/{r['n_run_down']})" if r["parse_fail_down"] else ""
    rows.append(f"| {LBL.get(m, m)} | {r['factor']} | ≈ {1800 // r['factor']} | {f(r['mae_best_down'])} | "
                f"{f(r['mape_best_down'])} | {r['within10_down']}/{2 * r['n_gt_paired']} | {f(r['bias_down'])} | "
                f"{f(r['mean_abs_change_tpi'])} (n = {r['n_pairs_all20']}){extra} |")
abl = "\n".join(rows)

# ---------------- Stage B ----------------
b = ["| Model | parsed | median vertical / horizontal | 5–95 % range (TPI) | mean TPI small crops (n) | "
     "mean TPI large crops (n) | s/img | $/img | $ total |",
     "|---|---|---|---|---|---|---|---|---|"]
for r in s["stageB"]:
    b.append(f"| {LBL.get(r['model'], r['model'])} | {r['n_parsed']}/{r['n_run']} | {f(r['median_v'], 0)} / "
             f"{f(r['median_h'], 0)} | {f(r['p05'], 0)}–{f(r['p95'], 0)} | {f(r['mean_tpi_small_crops'])} ({r['n_small']}) | "
             f"{f(r['mean_tpi_large_crops'])} ({r['n_large']}) | {f(r['sec_per_img'])} | {f(r['usd_per_img'], 4)} | "
             f"{f(r['usd_total'], 2)} |")
stageB = "\n".join(b)

# ---------------- Molmo2 text ----------------
A = {(r["model"], r["method"]): r for r in s["stageA"]}


def mrow(meth):
    return A.get(("molmo2_8b", meth))


def desc(meth):
    r = mrow(meth)
    if not r or not r["n_ok"]:
        return "not run yet"
    nrun = r["n_gt"] - (r.get("not_run_gt") or 0)
    cov = f"{r['n_ok']}/{r['n_gt']} GT images" if nrun == r["n_gt"] else f"run on {nrun} of {r['n_gt']} GT images so far, {r['n_ok']} parsed"
    return (f"{cov}: MAE {f(r['mae_best'])} TPI ({f(r['mape_best'])} %), "
            f"{r['within10_best']}/{r['n_values']} within 10 %, bias {f(r['bias_best'])}")


def jl(name):
    p = os.path.join(VLM, "raw", name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


mp = s.get("molmo2_pointing", {})
from collections import Counter
L, Rb, Rx = jl("molmo2_8b__legacy.jsonl"), jl("molmo2_8b__rect150.jsonl"), jl("molmo2_8b__legacy_x4.jsonl")
ok = lambda xs: [x for x in xs if not x.get("error")]          # noqa: E731
p1 = [x for x in ok(L) if x["prompt_id"] == "P1"]
is10 = lambda x: x["parsed"].get("vertical") == 10 and x["parsed"].get("horizontal") == 10   # noqa: E731
p1_10 = sum(map(is10, p1))
rb_ok = ok(Rb)
pair_counts = {}
for pid in ("P2", "P3"):
    xs = [x for x in ok(L) if x["prompt_id"] == pid and x["parsed"].get("ok")]
    pair_counts[pid] = Counter((x["parsed"]["vertical"], x["parsed"]["horizontal"]) for x in xs)
rb_10 = sum(map(is10, rb_ok))
_log = open(os.path.join(VLM, "logs", "molmo2_local.log")).read()
n_err = _log.count("exit code 3") + _log.count("Metal error on")  # warm-up failures exit without a record
gt_imgs = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
keys_done = {(x["image_id"], x["view"]) for x in ok(L)}
need = ["crop", "qc", "guide_h", "guide_v", "qc_guide_h", "qc_guide_v"]
imgs_full = [i for i in gt_imgs if all((i, v) in keys_done for v in need)]
complete = len(imgs_full) == 13 and len(rb_ok) >= 150 and len(ok(Rx)) >= 20
secs = sorted(x["seconds"] for x in ok(L) + rb_ok if isinstance(x.get("seconds"), (int, float)))
med_s = secs[len(secs) // 2] if secs else float("nan")
grid_pct = round(100 * (mp.get("frac_uniform_grid_cv_lt_0p05") or 0))

if complete:
    status = "**Status of the local Molmo2 run:** complete (all prompts on the 13 GT images, P1 on 20 legacy images, 150 crops and the x4 set)."
else:
    status = ("**Status of the local Molmo2 run: PARTIAL.** The shared GPU was saturated by other jobs on this machine "
              f"(load average up to ≈ 690; {n_err} Metal command-buffer timeouts). "
              f"Done so far: P1 on {len(p1)}/20 legacy images; all of P2, P3, P4 and P4q on {len(imgs_full)}/13 GT images "
              f"({', '.join(imgs_full) or 'none'}); rect150 P1 on {len(rb_ok)}/150 crops; ×4 ablation on {len(ok(Rx))}/20. "
              "The runner resumes from its logs: `python driver.py --models molmo2_8b --sets legacy --prompts P2,P3,P4,P4q "
              "--ids a1,a2,a4,a5,a6,a8,a10,r2,r3,r4,r6,r7,r8`, then `--sets rect150` and `--sets legacy_x4`, "
              "then `python evaluate_vlm.py`.")
tldr = (f"**Molmo2-8B** (local pointing specialist) fails on this task. Its direct-count answer is the constant 10 × 10 "
        f"on {p1_10}/{len(p1)} legacy images" + (f" and {rb_10}/{len(rb_ok)} of the 150 crops run so far" if rb_ok else "")
        + f" (P1 MAE {f(mrow('P1')['mae_best'])} TPI). Its pointing outputs are evenly spaced templates rather than "
        f"yarn positions: {grid_pct} % of {mp.get('n_outputs', 0)} point lists have a spacing CV below 0.05.")
section = "\n".join([
    "| Prompt | Result (best assignment) |", "|---|---|",
    f"| P1 direct JSON | {desc('P1')} |",
    f"| P2 step-by-step | {desc('P2')} |",
    f"| P3 quarter × 4 | {desc('P3')} |",
    f"| P4 pointing, whole window | {desc('P4')} |",
    f"| P4q pointing, quarter patch × 4 | {desc('P4q')} |",
    "",
    "**Direct count (P1).**",
    f"- Molmo2 answers exactly `10 x 10` on {p1_10}/{len(p1)} legacy images"
    + (f" and on {rb_10}/{len(rb_ok)} of the rectified crops of the 150 set" if rb_ok else "") + ", whatever the fabric.",
    "- The other count prompts give round template numbers too: " + "; ".join(
        f"{pid} answers " + ", ".join(f"{a:g} × {b:g} ({n})" for (a, b), n in sorted(cnt.items(), key=lambda kv: -kv[1]))
        for pid, cnt in pair_counts.items()) + " (raw answers per patch; P3 is multiplied by 4).",
    "",
    "**Pointing (P4/P4q).**",
    f"- Diagnostics are in `results/molmo2_pointing_diagnostics.csv`: {mp.get('n_outputs', 0)} point lists, median "
    f"{f(mp.get('median_n_points_full'), 0)} points per whole-window line and {f(mp.get('median_n_points_quarter'), 0)} per "
    "quarter-patch line.",
    f"- {grid_pct} % of the lists have inter-point spacing CV < 0.05 (median CV {f(mp.get('median_spacing_cv'), 3)}). "
    "These are arithmetic progressions in Molmo's 0–1000 coordinate frame (e.g. a1: 011, 050, 090, 130, …; a2: 005, 030, 055, 080, …).",
    "- Real yarn positions in a hand-held photo jitter by several percent. The model emits a template grid whose spacing, not the fabric, sets the count.",
    "- In the smoke test on the striped r2 fabric, whole-window pointing returned only the 2 white stripes.",
    "- Whole-window lines for dense fabrics can hit the decoding cap (≈ 11.3 tokens per point). Those lists are counted as emitted and flagged `truncated`.",
    "",
    "**Input resolution.** Molmo2 sees at most 644 px for a square image (2 × 2 crops of 378 px), i.e. 5–11 px per yarn "
    "at 60–120 TPI on the whole window. That is why the quarter-patch variant (P4q, ≈ 25 px per yarn) was added. It does "
    "not rescue the model: its quarter-patch lists are also uniform grids.",
])
speed = (f"median {f(med_s)} s per generation on the shared, saturated GPU (8–25 s when the GPU was less loaded); "
         f"{n_err} Metal command-buffer timeouts, each handled by the driver's automatic restart")
paper_m = (f"The pointing specialist Molmo2-8B does not engage with the texture: it answers 10 × 10 to the direct prompt on "
           f"{p1_10}/{len(p1)} images, and {grid_pct} % of its point lists are evenly spaced templates (spacing CV < 0.05) "
           "rather than yarn positions.")

md = open(TEMPLATE).read()
for key, val in (("__STAGEA_TABLE__", stageA), ("__ABL_TABLE__", abl), ("__STAGEB_TABLE__", stageB),
                 ("__MOLMO_STATUS__", status), ("__MOLMO_TLDR__", tldr), ("__MOLMO_SECTION__", section),
                 ("__MOLMO_SPEED__", speed), ("__MOLMO_PAPER__", paper_m)):
    md = md.replace(key, val)
left = [k for k in ("__STAGEA", "__ABL", "__STAGEB", "__MOLMO", "__PAPER") if k in md]
assert not left, left
open(os.path.join(VLM, "RESULTS.md"), "w").write(md)
print("RESULTS.md built; molmo complete =", complete, "| imgs_full", imgs_full, "| rect150", len(rb_ok), "| x4", len(ok(Rx)))

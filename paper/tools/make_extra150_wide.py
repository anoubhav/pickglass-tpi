"""Convert the label-free 150-photo outputs of Jobs 1 and 3 into the wide format that
tools/score150.py --extra-wide reads (image_id, <key>__v, <key>__h; <key>__v = VERTICAL yarns = our h).

  pg-lattice@ruler     PG-Lattice on the crop rectified from the glass's own ruler (Job 3,
                       methods/results/ruler/count150.csv: h_ruler / v_ruler; NaN for the 1 uncalibrated photo)
  cascade-gate         quality-aware cascade, frozen before labels (Job 1, tables/cascade_150_decisions.csv:
                       PG-Lattice unless confidence < t_c or resolvability < rho_min, then GPT-6.1-sol P1)

Writes tables/src/extra150_wide.csv. Run: <venv2>/bin/python tools/make_extra150_wide.py
Label day: tools/score150.py ... --extra-wide tables/src/extra150_wide.csv::pg-lattice@ruler,cascade-gate
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)

r = pd.read_csv(os.path.join(ROOT, "methods", "results", "ruler", "count150.csv")).set_index("id")
c = pd.read_csv(os.path.join(PAPER, "tables", "cascade_150_decisions.csv")).set_index("image")
# sanity: the cascade's "ours" columns and the ruler job's annotated-corner counts are the same frozen counter
j = r.join(c, how="inner")
assert len(j) == 150, len(j)
assert (abs(j.h_gt - j.ours_v_yarns) < 1e-3).all() and (abs(j.v_gt - j.ours_h_yarns) < 1e-3).all()
w = pd.DataFrame({"pg-lattice@ruler__v": r.h_ruler, "pg-lattice@ruler__h": r.v_ruler,
                  "cascade-gate__v": c.cascade_v_yarns, "cascade-gate__h": c.cascade_h_yarns,
                  "cascade-gate__escalated": c.escalate_gate})
w.index.name = "image_id"
out = os.path.join(PAPER, "tables", "src", "extra150_wide.csv")
w.sort_index().to_csv(out)
print(f"wrote {out}: {len(w)} rows; ruler NaN {int(w['pg-lattice@ruler__v'].isna().sum())}; "
      f"escalated {int(c.escalate_gate.sum())}/150")

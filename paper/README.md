# paper/

`tools/` holds the scripts that generate the paper's tables, figures and numbers, except those listed below; `tables/`
holds their outputs (CSV, JSON and the LaTeX fragments the paper inputs). The LaTeX sources and the compiled PDFs are
not part of this repository (the paper is under review; the PDF will be linked here), and `figs/` is not included
because several figures show the mill's photos. Scripts write figures into `paper/figs/` (git-ignored).

Not released: the script that writes `tables/final_numbers.tex` and `tables/final_numbers.json` (one macro per
76-scene number; the outputs are here, their generator is not), the scripts that draw the error-versus-discard curves
of supplementary Fig. S11 and the octave overlays of Fig. S24, and the spectral analysis of the halved and doubled
axes (main Sec. 5.2, supplementary Sec. S20).

Run every script from this folder (`cd paper`). The map from each table and figure to its command is in the
top-level README. Never run `tools/score150.py` or `tools/quality150.py` on partial or test labels with the default
`--out tables`: pass `--out ../scratch/...` (the default overwrites the paper's tables).

`tools/score150.py` is frozen (MD5 `5879ef44698de7b2a090e8bbee973625`, see `../CHECKSUMS.md`);
`tables/cascade_150_decisions.csv` holds the frozen cascade decisions (MD5 `57d49a4d4304bc395ed608894009f683`).
`tools/label_day.py` is the author's one-command label day for the two-annotator protocol (`--protocol dual`:
export checks, propagation to copies, the agreement of the two annotators, scoring against their mean, every table and
the paper build). It was frozen before any human count was looked at (SHA-256 in `../CHECKSUMS.md`; the released copy
differs only by path placeholders). It also needs the LaTeX sources and a TeX engine, so outside the author's setup
use the individual commands. `tools/iqa150.py` and `tools/iqa_generic.py` (the 76-scene generic-IQA test) and their
label-free input `tables/iqa150_scores.csv` were frozen the same way.

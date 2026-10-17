# paper/

`tools/` holds the scripts that generate the paper's tables, figures and numbers; `tables/` holds their outputs (CSV,
JSON and the LaTeX fragments the paper inputs). The LaTeX sources and the compiled PDFs are not part of this
repository (the paper is under review), and the paper's own figures are not included
(`figs/` holds only a README) because several show the mill's photos. Scripts write figures into `paper/figs/` (git-ignored).

`tools/reference/` holds the scripts that score methods against the reference annotator (A2), the textile-trained
annotator whose counts the main paper uses, and the other analyses that need both annotators' counts, for example the
main Tab. 1 (`make_tab1_ref.py`), the paired tests (`paired_ref.py`), the re-shoot table (`reshoot_consistency.py`,
`molmo_reshoot.py`) and the generic-IQA test against A2 (`expert_iqa.py`); its README lists them all. They read the
human counts, which are released after review. `tables/final_numbers.tex` was written before the reference
annotator's (A2's) counts were scored, so every macro holds a placeholder (`\TBD`) and none carries a value; it is kept
as a record; for the release only its comments were edited (new header, per-macro status notes removed, every
macro unchanged), and its generator is not part of this repository. The numbers scored
against A2 come from `tools/reference/` (`make_branchB_macros.py` writes `branchB_numbers.tex`), those against the
mean from `tools/label_day.py` and `tools/score150.py`.

Run every script from this folder (`cd paper`). The map from each table, figure and section of the paper to its
command is in the top-level README. Never run `tools/score150.py` or `tools/quality150.py` on partial or test labels
with the default `--out tables`: pass `--out ../scratch/...` (the default overwrites the paper's tables).

`tools/score150.py` is frozen (MD5 `5879ef44698de7b2a090e8bbee973625`, see `../CHECKSUMS.md`);
`tables/cascade_150_decisions.csv` holds the frozen cascade decisions (MD5 `57d49a4d4304bc395ed608894009f683`).
`tools/score150.py` scores every method against the mean of the annotation exports it is given: one export (A2's)
gives the scores against the reference annotator, both exports the scores against their mean.

`tools/label_day.py` runs the analysis of the human counts in one command (`--protocol dual`: export checks,
propagation to copies, the agreement of the two annotators, every method scored against the mean of both annotators,
the reference fixed in the analysis plan, and against each annotator alone, every table and the paper build). It was
frozen before any human count was looked at (SHA-256 in `../CHECKSUMS.md`; the released copy differs from the frozen
original only by redactions such as path placeholders, and behaves the same). The paper scores its main results
against A2 instead and reports A1, the annotator without textile training, as a non-expert human baseline; that choice
of reference is a declared deviation from the plan, decided after both annotators' counts were seen, and the results
against the mean of both are in the supplement (Tabs. S13, S14 and S18). `tools/label_day.py` also needs the LaTeX
sources and a TeX engine, so outside the author's setup use the individual commands. `tools/iqa150.py` and
`tools/iqa_generic.py` (the 76-scene generic-IQA test) and their label-free input `tables/iqa150_scores.csv` were
frozen the same way.

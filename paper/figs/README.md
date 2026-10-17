# paper/figs/

Output folder of the figure scripts in `paper/tools/`, `methods/make_figs.py`,
`vlm/make_scaffold_figs.py`, `camera_aug/make_camera_fig.py`, `genai_sr/make_figure.py`, `gpu_results/make_paper_figs.py`
and others (git-ignored except this file). The paper's own figures are not in the repository because several of them
show the mill's photos. The charts that need neither a photo nor a human count can be regenerated from the released
CSVs; the others need the photos or the human counts, which are released after review (see the top-level README for
what each figure needs). The scripts in `paper/tools/reference/` write to their own `--out` folder instead.

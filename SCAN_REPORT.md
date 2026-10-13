# Scan report

Scan of the staged repository before publication, 2026-10-12T23:30 (`scan_release.py`). 1116 files, 67.8 MB.

**Verdict: PASS (no blocker)**

| Check | Result |
|---|---|
| Key-shaped secrets (OpenAI/OpenRouter, HF, AWS, Google, GitHub, Bearer, password=, token=, private keys) | 0 |
| `.env` / key files | 0 |
| Absolute paths, scratch dirs, GPU host alias, personal host names | 0 |
| 'anoubhav' outside the repo URL / author name | 0 |
| Images / PDFs | 16, all on the synthetic allow-list: yes |
| Embedded base64 images in text files | 0 |
| Annotator names (applicant, author test name, screening stem) | 0 |
| Annotation exports / propagated labels / screening files | 0 |
| CDN, bucket, upload scripts | 0 (generic words: 0, listed below) |
| E-mail addresses | 0 |
| Frozen files byte-identical (MD5) | yes |
| Junk (.DS_Store, __pycache__, .pyc) | 0 |
| Files over 50 MB (GitHub hard warning) | 0 |
| Checklist grep: paths, GPU host, annotator names (blocking part) | 0 |
| Checklist grep: all hits (classified in section 0) | 19, unclassified: 0 |

## 0. Release checklist grep

The checklist command (run from the repository root) greps every file for absolute home and temp paths, the private GPU host alias and its SSH alias, the annotators' and applicants' names, and key prefixes. Its patterns are kept outside this repository.

Hits outside this report: 19. Absolute paths, scratch dirs, the GPU host alias and annotator names: 0. The key prefixes match only harmless text (none is a credential; the key-shaped patterns of section 1 find 0); they sit in code or logs that are kept as they are, some of them frozen (`paper/tools/label_day.py`, `paper/tools/iqa150.py`, `paper/tools/iqa_generic.py`), so they are classified, not rewritten:

| Kind | Hits | Files |
|---|---|---|
| English word ending in ...sk&#45; (task&#45;, risk&#45;) | 7 | `paper/tools/cascade_tradeoff.py:47`, `paper/tools/cascade_tradeoff.py:608`, `paper/tools/iqa150.py:42`, `paper/tools/iqa_generic.py:5`, `paper/tools/iqa_generic.py:261`, `paper/tools/quality150.py:6`, `paper/tools/quality150.py:9` |
| Hugging Face cache / module name (hf&#95;home, hf&#95;model, hf&#95;api) | 11 | `ft_results/logs/download_molmo2_4b.log:2`, `gpu_results/code/check_w4_layout.py:10`, `gpu_results/code/check_w4_layout.py:11`, `gpu_results/code/remote_top/env.sh:5`, `gpu_results/code/remote_top/env.sh:6`, `gpu_results/countgd/code/run_countgd.py:136`, `gpu_results/countgd/code/run_gpu_queue.sh:14`, `gpu_results/logs/download_mlx4bit.log:2`, `gpu_results/logs/download_molmo2.log:2`, `paper/tools/label_day.py:2008`, `vlm/logs/download.log:54` |
| the key scrubber's regex (vlm/openrouter_client.py scrub()) | 1 | `vlm/openrouter_client.py:149` |

## 1. Secrets

Every hit of the literal patterns, with why it is not a secret:

| Pattern | Hits | Not a secret because |
|---|---|---|
| sk&#45; | 8 | ordinary word (…sk&#45;) (7); redaction regex in scrub() (1) |
| sk&#45;or-v1 | 0 | - |
| hf&#95; | 11 | Hugging Face cache / module name (hf&#95;home, hf&#95;api, ...) (11) |
| AKIA | 0 | - |
| AIza | 0 | - |
| ghp_ | 0 | - |
| gho_ | 0 | - |
| Bearer | 3 | Authorization header built from the environment variable (3) |
| password | 0 | - |
| token= | 3 | token=False: anonymous Hugging Face download (3) |

`vlm/openrouter_client.py` reads the key only from `$OPENROUTER_API_KEY` (the former fallback to another project's `.env` file was removed).

## 2. Absolute paths and host names

Hard hits: 0. Soft hits (folder names inside placeholders or prose, allowed): fabric_notebooks (local folder) 6, fabric analysis (local folder) 4.

Absolute paths were replaced at export by environment variables with repo-relative defaults (`common/paths.py`) in code, and by `$PICKGLASS_NOTEBOOKS`, `$PICKGLASS_LEGACY_ROOT`, `$PICKGLASS_SCRATCH`, `gpu-server`, `~/pickglass`, `local-mac` placeholders in logs and data files.

## 3. Images

| File | Type | Provenance |
|---|---|---|
| `annotation_tool/demo/data/crops/801.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops/801.jpg |
| `annotation_tool/demo/data/crops/802.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops/802.jpg |
| `annotation_tool/demo/data/crops/803.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops/803.jpg |
| `annotation_tool/demo/data/crops_enh/801.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops_enh/801.jpg |
| `annotation_tool/demo/data/crops_enh/802.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops_enh/802.jpg |
| `annotation_tool/demo/data/crops_enh/803.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/crops_enh/803.jpg |
| `annotation_tool/demo/data/thumbs/801.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/thumbs/801.jpg |
| `annotation_tool/demo/data/thumbs/802.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/thumbs/802.jpg |
| `annotation_tool/demo/data/thumbs/803.jpg` | jpeg | synthetic, byte-identical to freelancer/web/practice/data/thumbs/803.jpg |
| `docs/img/guide/mistakes.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/mistakes.jpg |
| `docs/img/guide/photo_problems.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/photo_problems.jpg |
| `docs/img/guide/post_example.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/post_example.jpg |
| `docs/img/guide/practice_tool.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/practice_tool.jpg |
| `docs/img/guide/warp_weft.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/warp_weft.jpg |
| `docs/img/guide/weave_types.jpg` | jpeg | synthetic, byte-identical to freelancer/web/public_guide/img/weave_types.jpg |
| `docs/img/overview_synthetic.png` | png | synthetic, generated by tools/make_readme_figure.py |

Other binary files (not images): `methods/results/ruler/calib150.pkl`.
Every image was also inspected visually: the demo crops are the three synthetic practice fabrics, the guide pictures are synthetic renders (`docs/img/guide/SOURCES.json`) and a screenshot of the tool on synthetic practice fabric #801, the overview figure is a synthetic scene. No real photo, crop, render, contact sheet, dot map or figure of the paper is included.

## 4. Annotators, exports, CDN, e-mail

- Annotator-name hits: 0
- Values of `"annotator"` fields in data files: {'A': 1, 'B': 1, 'T': 5, 'U': 2} (test fixtures and synthetic examples only; no human count of PickGlass is included)
- Annotation exports / propagated labels / screening files: none
- CDN / bucket hits: 0
- E-mail addresses: none

## 5. Frozen files

| File | MD5 now | MD5 disclosed |
|---|---|---|
| `methods/results/counting_frozen_final.py` | `3d0520d6bec8a016543ce7e5b353d784` | `3d0520d6bec8a016543ce7e5b353d784` |
| `paper/tools/score150.py` | `5879ef44698de7b2a090e8bbee973625` | `5879ef44698de7b2a090e8bbee973625` |
| `paper/tables/cascade_150_decisions.csv` | `57d49a4d4304bc395ed608894009f683` | `57d49a4d4304bc395ed608894009f683` |

## 6. Size budget

Total 67.8 MB in 1116 files (GitHub: no file over 50 MB, repository well under 1 GB; no Git LFS needed). Largest folders and files:

| Folder | MB |
|---|---|
| `methods` | 25.6 |
| `camera_aug` | 8.5 |
| `vlm` | 7.8 |
| `genai_sr` | 4.7 |
| `gpu_results` | 4.6 |
| `annotation_tool` | 3.5 |
| `cnn` | 2.8 |
| `docs` | 2.6 |
| `paper` | 2.5 |
| `data` | 1.8 |
| `aug1016` | 1.7 |
| `ft_results` | 1.2 |
| `(root)` | 0.5 |
| `tools` | 0.0 |
| `common` | 0.0 |
| `handover` | 0.0 |
| `freelancer` | 0.0 |

| File | MB |
|---|---|
| `camera_aug/stress/real150/manifest.csv` | 4.52 |
| `methods/results/all150_degrade.csv` | 3.85 |
| `methods/results/combined_v1_synth_heldout.csv` | 2.59 |
| `methods/results/cascade_synth_cal_legacygrid.csv` | 2.36 |
| `methods/results/cascade_synth_cal_default.csv` | 2.31 |
| `methods/guard/results/synth_counts_kappa.csv` | 2.06 |
| `gpu_results/raw/molmo2_8b_bf16__rect150.jsonl` | 1.51 |
| `genai_sr/results/per_count_long.csv` | 1.06 |
| `docs/img/overview_synthetic.png` | 0.99 |
| `methods/guard/results/synth_counts_guarded.csv` | 0.92 |
| `methods/guard/results/real_counts.csv` | 0.92 |
| `camera_aug/stress/legacy13/manifest.csv` | 0.90 |

Left out on purpose (sizes in the private project): photo-derived images and PDFs, `camera_aug/stress/` (about 2.8 GB of renders), `cnn/data/` (about 2.1 GB of synthetic and rectified training data), raw CountGD dumps (about 80 MB), the LoRA adapter and every checkpoint (the 43 MB synthetic-only CNN checkpoint can be attached to a GitHub release), the 2024 notebooks.

## 7. Functional checks

Reproduction test of Oct 12 2026: a fresh copy of this repository, a new Python 3.11 virtual environment with only requirements.txt plus the documented optional scikit-learn and openpyxl (numpy 1.26.4, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2, opencv-python-headless 4.11.0.86, Pillow 12.3.0, scikit-learn 1.9.1, openpyxl 3.1.5) and Node.js 22; no photos, no human labels, no GPU, no API key, no PICKGLASS_* variable set. Each output was compared with the released file (SHA256SUMS).

| Check | Command | Result |
|---|---|---|
| Checksums of the frozen files and of every file | `python3 tools/verify_checksums.py` | all match (before and after the runs below, except the run-stamped outputs listed in the last rows) |
| README figure (frozen counter on a synthetic fabric) | `python3 tools/make_readme_figure.py` | byte-identical to docs/img/overview_synthetic.png |
| Ruler summary (Tab. S10 numbers) | `python3 methods/ruler_summary.py` | byte-identical (needs scikit-learn) |
| PickGlass-Stress on legacy-13 (Tab. S16) | `python3 camera_aug/score_stress.py; cd paper && python3 tools/make_camera_stress_supp.py` | byte-identical |
| px per yarn audit (Tab. S15) | `python3 camera_aug/label_audit.py summarise` | 4 outputs byte-identical; label_audit_long.csv.gz identical after decompression (gzip header time differs) |
| Native-resolution guard (Tab. S18) | `python3 methods/guard/analyze_real.py; python3 methods/guard/posthoc_supplement.py --tables` | 12 result files byte-identical |
| Quality-aware cascade (Tab. S13), incl. the frozen decisions | `cd paper && python3 tools/cascade_tradeoff.py --eval` | byte-identical; cascade_150_decisions.csv rewritten with MD5 57d49a4d... |
| Label-free per-scene numbers, crop sensitivity, legacy degradation, tolerance tables | `cd paper && python3 tools/scenes76_labelfree.py; python3 tools/make_supp_tables.py; python3 tools/degradation_table.py; python3 tools/tolerance_legacy13.py` | byte-identical |
| Generic no-reference IQA (Tab. S6) | `cd paper && python3 tools/iqa_generic.py` | iqa_legacy13.csv, iqa_genai.csv, iqa_summary.json, supp_iqa.tex byte-identical |
| LoRA and generative-enhancement tables | `python3 ft_results/code/make_ft_tables.py; python3 genai_sr/make_paper_table.py` | byte-identical |
| combined-v1 predictions (frozen rule on the frozen counter's outputs) | `python3 methods/combined_v1.py --predict` | combined_v1_150_wide.csv (the Tab. 1 input) byte-identical; 4 other outputs identical except their run-timestamp column |
| combined-v1 held-out synthetic test | `python3 methods/combined_v1_eval.py --synth` | same numbers and the same decision (better than both components); the JSON differs only in its creation time and in the hash of the released, path-redacted PREREG copy |
| Annotation tool tests | `node annotation_tool/tests/test_core.js, test_scenes.js, test_attention.js; python3 annotation_tool/tests/test_scenes.py, test_attention.py` | 38 of 39 (the 39th checks that every crop file exists: photo release), 11/11, 9/9, 7/7, 8/8 |
| Export checker, two annotators | `python3 annotation_tool/kit/check_export.py <A1 export> --second <A2 export>` | on synthetic exports of the demo set: per-annotator checks, overlap, shared-name and completeness-rule (8-dot line) detection work |
| Needs the photos (not reproducible from the repository alone) | `python3 vlm/make_scaffold_figs.py` | stops at Fig. S14 (drawn from legacy crop a2) before its table and Fig. S15 |

57 released result files were regenerated byte-identically: `camera_aug/results/label_audit_pairs.csv`, `camera_aug/results/label_audit_summary.csv`, `camera_aug/results/label_audit_sweep.csv`, `camera_aug/results/label_audit_thresholds.json`, `camera_aug/results/stress_legacy13_metrics.csv`, `camera_aug/results/stress_legacy13_per_count.csv`, `ft_results/tables/ft_supp_table.tex`, `ft_results/tables/legacy13_ft_rows.tex`, `genai_sr/results/genai_hallucination_summary.json`, `genai_sr/results/genai_hallucination_table.csv`, `methods/guard/results/ph_cascade_policies.csv`, `methods/guard/results/ph_clean150_scenes.csv`, `methods/guard/results/ph_defocus_detail.csv`, `methods/guard/results/ph_flag_vs_failure.csv`, `methods/guard/results/ph_rhr_scene_cluster.csv`, `methods/guard/results/real_posthoc_ungated_summary_repro.csv`, `methods/guard/results/real_repro_checks.csv`, `methods/guard/results/real_table_cascade.csv`, `methods/guard/results/real_table_cascade_vs_vlm.csv`, `methods/guard/results/real_table_clean.csv`, `methods/guard/results/real_table_genai.csv`, `methods/guard/results/real_tables.md`, `methods/results/combined_v1_150_wide.csv`, `methods/results/combined_v1_eval_synth_by_condition.csv`, `methods/results/ruler/summary.json`, `paper/tables/camera_stress_legacy13.csv`, `paper/tables/camera_stress_legacy13.tex`, `paper/tables/camera_stress_legacy13_supp.tex`, `paper/tables/cascade_150_decisions.csv`, `paper/tables/cascade_150_escalation.csv`, `paper/tables/cascade_failure_catch.csv`, `paper/tables/cascade_legacy13.csv`, `paper/tables/cascade_legacy13_by_condition.csv`, `paper/tables/cascade_legacy13_inputs.csv`, `paper/tables/cascade_main.tex`, `paper/tables/cascade_signal_quality.csv`, `paper/tables/cascade_sweep.csv`, `paper/tables/degradation_legacy13.csv`, `paper/tables/degradation_legacy13_conf.csv`, `paper/tables/degradation_legacy13_per_image.csv`, `paper/tables/fabric_groups_candidate.csv`, `paper/tables/genai_hallucination.csv`, `paper/tables/genai_hallucination.tex`, `paper/tables/iqa_genai.csv`, `paper/tables/iqa_legacy13.csv`, `paper/tables/iqa_summary.json`, `paper/tables/quality76.csv`, `paper/tables/ruler_calib.csv`, `paper/tables/ruler_calib.tex`, `paper/tables/scenes76_labelfree.json`, `paper/tables/supp_crop_sensitivity.tex`, `paper/tables/supp_degradation_legacy.tex`, `paper/tables/supp_iqa.tex`, `paper/tables/supp_tolerance.tex`, `paper/tables/tolerance_image_paired.csv`, `paper/tables/tolerance_legacy13.csv`, `paper/tables/tolerance_paired.csv`.


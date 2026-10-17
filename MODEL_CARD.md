# Model card: PG-Lattice and Molmo2-4B + synthetic LoRA (and the quality-aware cascade)

Following *Model Cards for Model Reporting* (Mitchell et al., FAT* 2019). This card covers the paper's two methods: **PG-Lattice**, a training-free spectral counter, and **Molmo2-4B + synthetic LoRA**, a small open vision-language model that we tuned only on synthetic crops. PickGlass scores are against **A2**, the textile-trained annotator, as in the paper's Tab. 1. The photos, the human counts and the per-scene tables computed from them (for example `paper/tables/main150.csv` and `paper/tables/strata150_mape.csv`) are released after review.

## Model details

**PG-Lattice**

- **What.** A training-free estimator of thread density on a rectified pick-glass window: a joint reciprocal-lattice hypothesis search over the 2-D power spectrum, fused with a projection-profile (YIN) and a tiled FFT estimator; outputs `h`, `v` (yarns crossing a horizontal / vertical line across the window), a weave label and a confidence. Code: `methods/results/counting_frozen_final.py` (frozen; MD5 `3d0520d6bec8a016543ce7e5b353d784`), confidence model `methods/conf_model.json` (logistic, fitted on synthetic data only). `methods/counting.py` is the same estimator with a non-square-crop extension of `guide_positions()`.
- **Window.** Corners from a human polygon or from the glass's engraved scales (`methods/ruler_calib.py`).
- **Combined counter (pre-registered).** combined-v1 takes, per axis, the geometric mean of PG-Lattice and the log-Gabor bank when their log ratio is at most 0.18, and PG-Lattice otherwise (`methods/results/combined_frozen_v1.py`, `methods/results/PREREG_combined_v1.json`, frozen before any human label; [CHECKSUMS.md](CHECKSUMS.md)). On fresh synthetic crops it raises the share of counts within 5 % from 70.3 to 72.4 %; on PickGlass it barely changes the error (34.4 % against 34.8 %, paper Tab. 1).
- **Cascade (optional).** A no-reference gate (`paper/tools/cascade_tradeoff.py`, decisions for PickGlass frozen in `paper/tables/cascade_150_decisions.csv`, MD5 `57d49a4d...`) sends a photo to a VLM (GPT-6.1-sol, prompt P1) when the counter's confidence or resolvability is low. On PickGlass it barely changes PG-Lattice's error (-0.3 points, 95 % CI -3.1 to +2.3; paper Sec. 5.4).
- **Development.** Every design choice was made on 300 rendered synthetic images with exact labels (`methods/eval_synth.py`, `dev_lattice_ablation.py`; log in the counter's docstring); no parameter was fitted on the 13 hand-counted legacy crops, and the paper lists where the legacy set informed later choices, such as the cascade's resolvability check (paper Sec. 3; supplement Sec. S1). The counter's MD5, `3d0520d6bec8a016543ce7e5b353d784`, identifies the version.

**Molmo2-4B + synthetic LoRA**

- **What.** Molmo2-4B (`allenai/Molmo2-4B`), an open point-then-count VLM, LoRA-tuned (rank 32, 1,000 steps) on 12,000 crops from our generator, which knows exactly where each yarn crosses a guide line. 20 % of the training examples ask the direct prompt P1 (`h`, `v` and the weave as JSON), 30 % ask the model to point at every yarn crossing a guide line across the whole inch (P4) and 50 % to do so on magnified windows. No real image or human count is used; one run, one seed, no hyper-parameter search (supplement Sec. S8). Code: `ft_results/code/` (`train_lora.py`, `synth_pts.py`, `infer_ft.py`); predictions: `data/predictions/pickglass150/molmo2_4b_lora_wide.csv` and `data/predictions/legacy13/molmo2_4b_lora.csv`.
- **Readout.** The checkpoint and the readout, P1, were fixed on synthetic validation (7.2 % MAPE) before any legacy crop or human count was scored. The P4s readout (four magnified quarter inches, summed) is post hoc.
- **Compute.** Tuning took 5.9 GPU-hours on two consumer GPUs; inference takes about 0.19 s per crop on an RTX 4090 (supplement Tab. S3). The weights are open, so a mill could host the model on one GPU. The LoRA adapter is not in git ([THIRD_PARTY.md](THIRD_PARTY.md)); the base model's own licence applies to it.

**Both.** Licence of our code: MIT. **Contact:** GitHub issues.

## Intended use

- **Primary.** PG-Lattice: a transparent, offline **baseline and diagnostic** for thread density from a phone photo through a pick glass; its spectrum can be inspected, and it shows why counts come out at half or double. Molmo2-4B + synthetic LoRA: the lowest-error method on PickGlass against A2, with open weights a mill could host. Neither meets the ±5 % screening tolerance on PickGlass (no tested method does; paper Sec. 5.1), so neither is yet a screening tool. A counter that met the tolerance could help an inspector check more places per piece and flag pieces for a full manual count.
- **Users:** textile quality inspectors, small mills, researchers.
- **Out of scope:** certification or acceptance testing (the reference remains the manual count of IS 1963 / ISO 7211-2 / ASTM D3775 at five or more places per piece); the sole basis for rejecting a piece or cutting a weaver's pay; monitoring individual weavers (an excluded misuse); knitted, nonwoven, pile, leno and lappet fabrics; up-sampled or "AI-enhanced" input.

## Factors

Resolution (px per yarn), defocus and motion blur, glare and shadow, corner error, weave (plain, twill, satin, colour-woven checks), fabric density, the glass model (its opening and scales), the phone pipeline (sensor, compression, messaging re-encoding), and, for the human reference, which annotator counted (their readings often differ by a factor of two).

## Metrics

Share of counts within 5 % and 10 % of the reference, share of images with both counts within 5 % (`both5`), MAE, MAPE with 95 % bootstrap CI over images or scenes, the symmetric error abs(log2(count / reference)), median signed error (bias), weave accuracy, repeatability across re-shoots of the same fabric patch, self-consistency under degradation, risk–coverage of the confidence ([problem definition](docs/problem_definition.md)).

## Evaluation data

- **Legacy-13** (held out; no counter was fitted on it except two post-hoc variants; it informed some design choices, supplement Sec. S1): 13 hand-counted crops from an earlier project, 26 counts, one crop rule for every method.
- **PickGlass** (test): 76 scenes, each counted from scratch by two annotators. A2, the textile-trained annotator, is the reference; A1, without textile training, is a non-expert human baseline (first row of the paper's Tab. 1). The plan fixed before labeling scored against the mean of both annotators; scoring against A2 was decided after both annotators' counts were seen and is a declared deviation (mean of both: supplement Tabs. S18 and S13; each annotator alone: Tab. S15). The noise floor is the re-shoot spread: even A2 repeats a count across two photos of the same fabric patch only to a median 8.9 % (paper Tab. 2 and Sec. 5.3), which is why ±5 % cannot be verified against one photo count.
- **Degradations and PickGlass-Stress:** label-free self-consistency on the 76 scenes; accuracy on legacy-13.
- **Training data:** none for PG-Lattice (the confidence model and the gate thresholds are fitted on synthetic data from `cnn/synth.py`); 12,000 synthetic crops, and no real image, for Molmo2-4B + synthetic LoRA.

## Quantitative analyses (from the paper)

| Data | PG-Lattice | Molmo2-4B + synthetic LoRA |
|---|---|---|
| PickGlass against A2 (paper Tab. 1, 76 scenes) | MAPE 34.8 % [28, 41]; 28 % of counts within 5 %, 39 % within 10 %; both counts within 5 % on 6 scenes. Does not beat guessing each scene as the mean of the other scenes' counts (33.2 %; +1.6 points, CI -6.1 to +9.4), mostly because it reads half or double the yarns: on 36 of the 152 lines (24 half, 12 double; paper Sec. 5.2) | MAPE 25.2 % [21, 31], the lowest of every method; 14 % of counts within 5 %, 29 % within 10 %; both counts within 5 % on 1 scene. Beats the mean-density guess by 8.0 points (CI 2.9 to 13.3) and the non-expert annotator (37.3 %) by 12.1 points (CI 4.6 to 19.7); level on MAPE with GPT-6.1-sol P1 (27.1 %; -1.9 points, CI -6.5 to +2.6) and closer to A2 on the symmetric error (-0.13, CI -0.22 to -0.05; paper Sec. 5.1) |
| PickGlass, other references | mean of both annotators 35.1 % (supplement Tab. S18; 35.3 % without the four guide-example scenes #27, #65, #93, #110, Tab. S4); A1 alone 41.7 % (Tab. S15) | mean of both annotators 28.2 % [24.2, 32.4], against 28.7 % for the mean-density guess; A1 alone 47.6 %, where its lead over PG-Lattice disappears (supplement Sec. S8) |
| Re-shoots, 41 pairs of photos of one patch (paper Tab. 2) | median worse-axis difference 1.8 %; 24 pairs within 5 % on both axes (the most of any method or annotator), but 8 pairs about twofold apart, and on 8 of the 24 repeating pairs both photos read about half or double A2 on an axis | median 14.3 %; 11 pairs within 5 %; 1 pair about twofold apart |
| Legacy-13 (supplement Tab. S3) | MAPE 11.6 % [3.1, 23.1]; 19 of 26 counts within 5 %; four counts miss by more than 25 % (a blurred twill on both axes, a glare-dominated weft, a twill warp). Cannot be separated from the best of seven VLMs, GPT-6.1-sol P1 at 8.8 % (difference +2.8 points, 95 % CI -5.1 to +13.2) | MAPE 18.4 % [10.6, 29.1], against 82.9 % for the untuned model; fewer counts within 10 % than PG-Lattice (9 vs 21 of 26); behind GPT-6.1-sol P1 (+9.6 points, CI +2.7 to +18.5) |
| Legacy-13, degraded | defocus of 2-4 u makes the counter over-count by up to nearly 7x (MAPE 186-246 %, supplement Tab. S5); the cascade sends 24 % of the 117 legacy photos (13 crops x 9 capture conditions, clean included) to the VLM at about a quarter of the VLM-only cost, with error not distinguishable from the VLM alone (13.6 % vs 13.5 %; paired 95 % CI -4.6 to +6.3 points; supplement Tab. S8) | not part of the degradation study |
| Per fabric type and quality bin | by resolution class, capture-quality flag, px per yarn, near-saturated pixels and window side ratio, against the mean of both annotators: supplement Sec. S9. Error per fabric type and quality bin against A2 (as stated in the paper's impact section) is released with the human counts after review | released with the human counts after review |
| Cost | offline, deterministic, free; about 3.5 s per crop on a laptop CPU | open weights, about 0.19 s per crop on an RTX 4090; the API VLM of the cascade costs about US$0.011 per photo and needs connectivity |

## Ethical considerations

- Sending factory images to a VLM API shares them with a third party. PG-Lattice runs offline and Molmo2-4B + synthetic LoRA can be hosted locally; the cascade calls the API VLM only when needed.
- A model that is wrong in a systematic direction (bias) can push pieces out of tolerance; report bias, not only MAE, and keep the manual count as the reference. A halved count makes a compliant piece look 50 % below specification, so a count near half or double the specification must go to a manual recount before it can affect a piece's acceptance or a weaver's pay (paper Sec. 6).
- PG-Lattice shows the yarns it counted (dots on the two guide lines), so a person can check it. Molmo2-4B's tuned point lists are near-perfectly regular (median gap coefficient of variation 0.000, against 0.04 for the exact crossings): it reads the pitch without locating each yarn, so its points cannot be checked yarn by yarn (supplement Sec. S8).
- Woven designs and factory photos may be a client's or the mill's property; see [DATASHEET.md](DATASHEET.md).

## Caveats and recommendations

- **Calibrate the glass**: counts are threads per window side, which is per inch only if the opening is one inch. On the glass used for PickGlass the opening spans 26.0 fine divisions of the engraved scales: +2.4 % if the divisions are millimetres, +4 % if they are 1/25 inch, and the loupe's pincushion distortion adds about +2.7 % on centre lines. Only a physical measurement of the glass can settle the scale (paper Sec. 3; supplement Secs. S10 and S13).
- **Ask for at least 4 px per yarn** (about 7 for flip-free counts; counts collapse below about 3; paper Sec. 5.5, supplement Sec. S13), never up-sample or enhance, and gate blur before trusting a PG-Lattice count: its confidence misses the defocus failure.
- **The ranking depends on the reference.** Against A2, Molmo2-4B + synthetic LoRA errs least; against the mean of both annotators its error (28.2 %) is level with the mean-density guess (28.7 %), and against A1 it is higher than PG-Lattice's (47.6 % vs 41.7 %). Read every number next to the human noise floor.
- **Weave labels are unreliable** for every method; use them as a hint. On PickGlass the two annotators' weave labels agree no better than chance (kappa -0.01, supplement Tab. S4).
- **Colour-woven checks** add spectral peaks; satins other than 5-end ones, basket weaves, jacquards and knits fall outside the lattice model.
- Evaluated at one site with one glass and probably one phone; legacy-13 has three formula-derived twill labels; VLM comparisons are a snapshot in time (model ids and access times are in the run logs, `vlm/raw/` and `vlm/logs/`).

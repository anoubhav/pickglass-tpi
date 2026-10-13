# Model card: PG-Lattice thread counter (and the quality-aware cascade)

Following *Model Cards for Model Reporting* (Mitchell et al., FAT* 2019). Numbers marked **TBD** need the PickGlass human counts (two annotators, label day Oct 12 2026) and will be filled from `paper/tables/main150.csv` and `strat150.csv`.

## Model details

- **What.** PG-Lattice, a training-free estimator of thread density on a rectified pick-glass window: a joint reciprocal-lattice hypothesis search over the 2-D power spectrum, fused with a projection-profile (YIN) and a tiled FFT estimator; outputs `h`, `v` (yarns crossing a horizontal / vertical line across the window), a weave label and a confidence. Code: `methods/results/counting_frozen_final.py` (frozen; MD5 `3d0520d6bec8a016543ce7e5b353d784`), confidence model `methods/conf_model.json` (logistic, fitted on synthetic data only). `methods/counting.py` is the same estimator with a non-square-crop extension of `guide_positions()`.
- **Window.** Corners from a human polygon or from the glass's engraved scales (`methods/ruler_calib.py`).
- **Combined counter (pre-registered).** combined-v1 takes, per axis, the geometric mean of PG-Lattice and the log-Gabor bank when their log ratio is at most 0.18, and PG-Lattice otherwise (`methods/results/combined_frozen_v1.py`, `methods/results/PREREG_combined_v1.json`, frozen before any human label; [CHECKSUMS.md](CHECKSUMS.md)). On fresh synthetic crops it raises the share of counts within 5 % from 70.3 to 72.4 %.
- **Cascade (optional).** A no-reference gate (`paper/tools/cascade_tradeoff.py`, decisions for PickGlass frozen in `paper/tables/cascade_150_decisions.csv`, MD5 `57d49a4d...`) sends a photo to a VLM (GPT-6.1-sol, prompt P1) when the counter's confidence or resolvability is low.
- **Development.** Every design choice was made on 300 rendered synthetic images with exact labels (`methods/eval_synth.py`, `dev_lattice_ablation.py`; log in the counter's docstring); the 13 hand-counted legacy images were not looked at until the configuration was frozen. Version 2026-10-07.
- **Licence:** MIT. **Contact:** GitHub issues.

## Intended use

- **Primary:** a transparent, offline **baseline and diagnostic** for thread density from a phone photo through a pick glass: its spectrum can be inspected, and it shows why counts come out at half or double. It does **not** meet the ±5 % screening tolerance on PickGlass (no tested method does; paper Sec. 5.1), so it is not yet a screening tool. A counter that met the tolerance could help an inspector check more places per piece and flag pieces for a full manual count.
- **Users:** textile quality inspectors, small mills, researchers.
- **Out of scope:** certification or acceptance testing (the reference remains the manual count of IS 1963 / ISO 7211-2 / ASTM D3775 at five or more places per piece); the sole basis for rejecting a piece or cutting a weaver's pay; monitoring individual weavers (an excluded misuse); knitted, nonwoven, pile, leno and lappet fabrics; up-sampled or "AI-enhanced" input.

## Factors

Resolution (px per yarn), defocus and motion blur, glare and shadow, corner error, weave (plain, twill, satin, colour-woven checks), fabric density, the glass model (its opening and scales), the phone pipeline (sensor, compression, messaging re-encoding).

## Metrics

Share of counts within 5 % and 10 % of the reference, share of images with both counts within 5 % (`both5`), MAE, MAPE with 95 % bootstrap CI over images or scenes, median signed error (bias), weave accuracy, self-consistency under degradation, risk–coverage of the confidence ([problem definition](docs/problem_definition.md)).

## Evaluation data

- **Legacy-13** (held out): 13 hand-counted crops from 2019, 26 counts, one crop rule for every method.
- **PickGlass** (test): 76 scenes; the reference is the mean of two annotators' counts, each made from scratch; the agreement between the two annotators is **TBD**. The noise floor of the paper's main table is the re-shoot spread: the first annotator's count of another photo of the same fabric patch, used as a prediction (Tab. 1, "Annotator, another photo"; **TBD**). This spread is why ±5 % cannot be verified against one photo count (paper Sec. 3).
- **Degradations and PickGlass-Stress:** label-free self-consistency on the 76 scenes; accuracy on legacy-13.
- **Training data:** none for the counter (the confidence model and the gate thresholds are fitted on synthetic data from `cnn/synth.py`).

## Quantitative analyses (from the paper)

| Data | Result |
|---|---|
| Legacy-13 | MAPE 11.6 % [3.1, 23.1]; 19 of 26 counts within 5 %; four counts miss by more than 25 % (a blurred twill on both axes, a glare-dominated weft, a twill warp). Cannot be separated from the best of seven VLMs, GPT-6.1-sol P1 at 8.8 % (difference +2.8 points, 95 % CI -5.1 to +13.2) |
| Legacy-13, degraded | defocus of 2-4 u makes the counter over-count up to 7x (MAPE 186-246 %); the cascade sends 24 % of degraded legacy photos to the VLM at a quarter of its API cost, with error not distinguishable from the VLM alone (13.6 % vs 13.5 %; paired 95 % CI -4.6 to +6.3 points) |
| PickGlass | From the paper (Sec. 5.1-5.2): no tested method comes near the ±5 % tolerance; PG-Lattice trails GPT-6.1-sol at both resolutions and, unlike on legacy-13, does no better than a predictor of the mean density; much of its error is a factor of two (it reads half or double the yarns). Numbers **TBD**: against the mean of the two annotators: MAPE, within 5 % / 10 %, both5, bias, per resolution class (50 full / 26 quarter-only scenes), on the 13 scenes whose weave the counter calls "unknown", and without the four guide-example scenes (#27, #65, #93, #110); next to the agreement between the two annotators |
| Per fabric type and quality bin | **TBD** (promised in the paper's impact section): error by weave, by px per yarn, and by each capture-quality flag |
| Cost | offline, deterministic, free; about 3.5 s per crop on a laptop CPU; the VLM costs about US$0.011 per photo and needs connectivity |

## Ethical considerations

- Sending factory images to a VLM API shares them with a third party; the cascade calls the VLM only when needed, and the counter itself runs offline.
- A model that is wrong in a systematic direction (bias) can push pieces out of tolerance; report bias, not only MAE, and keep the manual count as the reference.
- The system shows the yarns it counted (dots on the two guide lines), so a person can check it.
- Woven designs and factory photos may be a client's or the mill's property; see [DATASHEET.md](DATASHEET.md).

## Caveats and recommendations

- **Calibrate the glass**: counts are threads per window side, which is per inch only if the opening is one inch. On the glass used for PickGlass the opening spans 26.0 fine divisions of the engraved scales: +2.4 % if the divisions are millimetres, +4 % if they are 1/25 inch, and the loupe's pincushion distortion adds about +2.7 % on centre lines. Only a physical measurement of the glass can settle the scale (paper Sec. 5.4 and supplement).
- **Ask for at least 4 px per yarn** (about 7 for flip-free counts; counts collapse below about 2.5 to 3.5; paper Sec. 5.4), never up-sample or enhance, and gate blur before trusting a count: the confidence misses the defocus failure.
- **Weave labels are unreliable** for every method; use them as a hint.
- **Colour-woven checks** add spectral peaks; satins, basket weaves, jacquards and knits fall outside the lattice model.
- Evaluated at one site with one glass and probably one phone; legacy-13 has three formula-derived twill labels; VLM comparisons are a dated snapshot (accessed 2026-10-07).

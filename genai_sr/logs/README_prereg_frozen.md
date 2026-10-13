# genai_sr: does generative enhancement hallucinate or erase threads?

Job A of the WVAQ 2027 paper (workshop topics: "hallucination effects in image super-resolution and
restoration using diffusion models" and "impact of image quality on CV tasks").

**Status of this file.** Sections 1 to 8 are the protocol. They were written and frozen on
2026-10-07 23:15 UTC, before any restoration model was downloaded or run and before any count was
computed on a restored image. The md5 of the frozen text is in `logs/prereg_md5.txt`, and a copy is
at `logs/README_prereg_frozen.md`. Deviations go in section 9 with a reason, and results go in
section 10. The protocol text above section 9 is never edited.

---

## 1. Question and hypotheses

Question: if a cheap, low-resolution pick-glass capture is "enhanced" by a modern generative
super-resolution or restoration model before counting, do the yarn counts get better, stay the same,
or get worse? Does the model invent periodic texture (hallucinate threads) or wipe it out (erase
threads)?

- **H1 (primary).** At ×8 down-scaling, no generative restoration lowers the frozen counter's ("ours")
  error compared with plain bicubic up-sampling. Some may raise it.
- **H2.** Generative restorations add spectral peaks that are not on the reference yarn lattice
  (hallucinated periodicity) more often than bicubic does.
- **H3 ("looks better, counts worse").** Generative restorations score better than bicubic on
  perceptual image-quality metrics (LPIPS against the reference, and no-reference MUSIQ / CLIP-IQA),
  yet their count error is not lower.
- **H4 (silent failure).** When a generative restoration makes ours wrong by more than 10 %, ours'
  confidence often stays above the deployment threshold.

The tests are two-sided. The direction of any effect is the finding.

## 2. Reference sets (fixed before any run)

| Set | Images | Reference image | Reference counts |
|---|---|---|---|
| **R-HR** | the 53 of the 150 crops with side S ≥ 1500 px (≈1890 px/inch) **and** ours confidence ≥ t_c = 0.5741 in `methods/predictions_ours_150.csv`. t_c is the cascade's confidence-only gate (`paper/tables/cascade_thresholds.json`, fits.vlm9_a10.conf.t_c), which was fitted on synthetic data only. Ids: 1 5 9 15 17 18 25 27 28 30 33 38 39 45 47 48 49 50 52 53 54 60 62 64 65 67 70 73 77 80 84 86 88 90 91 95 98 99 110 118 123 129 130 131 133 137 138 140 141 143 144 145 146 | `annotation_tool/data/crops/{i}.jpg` (used as is) | **Primary:** ours on the clean reference image, computed in this run in the same environment as every degraded and restored image (a pseudo ground truth). **Secondary:** each estimator against its own clean prediction (self-consistency). |
| **R-LEG** | legacy-13: a1 a2 a4 a5 a6 a8 a10 r2 r3 r4 r6 r7 r8 | the canonical crops `vlm/crops/legacy/{name}.png` (1736–1941 px) | Human hand counts, parsed by the 2019 harness exactly as in `paper/tools/legacy_unified.py` (`load_gt`). Axis assignment is "axis", as in the paper: h_count ↔ warp, v_count ↔ weft. |

## 3. Degradations (cheap capture emulation)

These use `methods/degrade.py`, unchanged (md5 fd049ff3…).

- **D4**: `downscale(img, 4)` (INTER_AREA). On R-HR this gives about 472 px/inch, which is the
  resolution of the 85 real low-res captures.
- **D8**: `downscale(img, 8)`, about 236 px/inch. The LR Nyquist limit is S/16 ≈ 118 threads/inch, so
  denser fabrics alias. Expected: 11 R-HR counts and 2 R-LEG warp counts lie above this limit.
- **D4J**: `jpeg(downscale(img, 4), 50)`.

Each LR image is saved once as a lossless PNG with an md5 manifest. Every restoration reads those
exact bytes.

## 4. Restorations (every output is brought back to the reference size S×S)

| Key | Family | Model / version | Notes |
|---|---|---|---|
| `none` | no restoration | n/a | The LR image itself; the counters are resolution independent (counts per crop side). |
| `bicubic` | interpolation (**control**) | `cv2.resize INTER_CUBIC` | |
| `swinir` | non-generative regression SR | SwinIR-M classical ×4 (DF2K, L1 loss) | Strong PSNR-oriented baseline. |
| `realesrgan` | GAN blind SR | Real-ESRGAN x4plus | Adversarial and deterministic. Lies between regression and diffusion. |
| `seedvr2` | one-step diffusion transformer | SeedVR2-3B (ByteDance-Seed, 2025, Apache-2.0), fp16 weights from `numz/SeedVR2_comfyUI` run by the `numz/ComfyUI-SeedVR2_VideoUpscaler` standalone CLI | The most-downloaded open diffusion restorer on the HF Hub (2025–26). Output resolution is set by the target side. |
| `invsr` | diffusion inversion SR (SD-Turbo) | InvSR (Yue et al., CVPR 2025), `noise_predictor_sd_turbo_v5`, 1 step | Academic real-ISR state of the art, 1–5 steps. |
| `flux2klein` | rectified-flow instruction editor | FLUX.2 [klein] 4B (Black Forest Labs, Jan 2026, Apache-2.0), 4 steps | Open "AI enhance". The prompt is identical to the commercial model's. |
| `nbpro` | commercial AI image editor | Nano Banana Pro = `google/gemini-3-pro-image` via OpenRouter | R-LEG only, 13 × 3 = 39 images (cap 40). Prompt: "Enhance and upscale this fabric photo, keep it realistic". |

Scale handling, fixed in advance:
- The ×4 models (SwinIR, Real-ESRGAN, InvSR) are applied **once**. For D8, their ×4 output (S/2) is
  bicubic-resized to S.
- SeedVR2 receives the LR image with target resolution S.
- FLUX.2-klein receives the LR image bicubic-upsampled to 1024², generates 1024², and the result is
  resized to S.
- Nano Banana Pro receives the LR PNG as is, with 2K output requested if the API accepts that.
- The final resize to S×S uses INTER_CUBIC when up-sampling and INTER_AREA when down-sampling.
- Seeds: 20261008 wherever a model takes one. Default sampler settings are used otherwise. No
  per-image tuning and no cherry-picking: every output is scored.

## 5. Estimators

- **ours**: the frozen counter (`methods/counting.py`, estimators identical to
  `results/counting_frozen_final.py`, confidence from `methods/conf_model.json`), with defaults
  (lcn, glare mask, max_work 1024).
- **log-Gabor**: same file, `method="gabor"`.
- **CNN**: `cnn/runs/synth_r18_v2mix_aniso/model.pt` (md5 dad61b74…), with `cap_side(·,1024)` and the
  3-scale × 3-flip TTA, exactly as `cnn/degrade_vlmcrops.py`.
- **VLM (R-LEG, D8 only)**: GPT-6.1-sol, prompt P1, on the SeedVR2 output and on the bicubic output.
  The input is a 1536² JPEG q92, as for every VLM row in the paper. If the US$6 cap allows, the same
  is run on the Nano Banana Pro output. The SeedVR2 choice is fixed now: it is the most-adopted open
  diffusion restorer.

## 6. Metrics

Per count, a crop gives two counts (h and v). APE = |pred − ref| / ref × 100.

- **Count accuracy** (primary: ours; also Gabor, CNN): MAPE with a 95 % percentile-bootstrap CI over
  images (10 000 resamples, both counts of an image kept together, seed 20261008); the share of counts
  within 5 % and within 10 %; median APE.
- **Paired comparison against bicubic** (same images and condition):
  - paired-bootstrap 95 % CI of ΔMAPE;
  - Wilcoxon signed-rank on the per-image mean APE;
  - exact McNemar on per-count within-10 %;
  - Holm correction over the 6 non-control restorations × 3 conditions, within each reference set
    and estimator.
- **Count drift against bicubic**: per count, |c_m − c_bicubic| / c_bicubic; the share above 5 % and
  above 10 %; and whether the change moves toward or away from the reference.
- **Silent failure**: among restored crops where ours has an APE above 10 % on either count, the share
  with ours confidence ≥ t_c = 0.5741.
- **Lattice fidelity** uses the counter's own whitened spectrum (`counting.Spectrum`) at the common
  working size N = 1024 (`none` is analysed at its own size). The reference lattice is
  B_ref = `est_lattice` on the clean reference (centroid basis a, b) plus its weave sub-lattice W_ref.
  - (a) **Dominant-peak shift**: per yarn family, the strongest near-axis (±10°) peak in the band
    [18, 0.42 N] cycles/inch. Report the relative frequency shift and the angle shift against the same
    peak in the reference, and the share of families shifted by more than 5 %.
  - (b) **Hallucinated-peak energy fraction (HPEF)**: significant local maxima of the restored Z map
    (z ≥ 6, 5×5 maxima, band [18, 0.42 N], half plane). A peak is *explained* if it lies within
    max(1.5 bins, 3 % of |f|) of any integer lattice point m·a + n·b (|m|, |n| ≤ 6), of a W_ref
    sub-lattice point, or of a significant reference peak (z ≥ 4). HPEF = Σ g⁺(unexplained) / Σ g⁺(all),
    where g⁺ = z minus the noise baseline. Also report the number of unexplained peaks.
  - (c) **Lattice erasure**: among the reference lattice points (a, b, a±b, 2a, 2b and the W_ref
    points) with reference g ≥ 3, the share with restored g < 3. This is split into points below the LR
    Nyquist limit (recoverable: erasure) and points above it (not recoverable: a restored g ≥ 3 there
    means re-synthesis).
- **Image quality**:
  - PSNR and SSIM on luminance against the reference at S×S (skimage, Gaussian window σ 1.5,
    data_range 255);
  - LPIPS (AlexNet, v0.1) on RGB after both images are INTER_AREA-resized to 1024²;
  - no-reference MUSIQ (KonIQ) and CLIP-IQA (pyiqa) on the 1024² image, to show whether a restoration
    "looks better".

## 7. Decision rules (fixed in advance)

- A restoration **worsens counting** at a condition if its ours MAPE is higher than bicubic's and the
  paired-bootstrap CI of ΔMAPE excludes 0. It **helps** if the MAPE is lower and the CI excludes 0.
  Otherwise there is no detectable difference.
- **Hallucinates periodicity**: its median HPEF is above bicubic's (paired Wilcoxon p < 0.05).
- **Erases threads**: its below-Nyquist erasure is above bicubic's (paired Wilcoxon p < 0.05).
- **"Looks better but counts worse"**: LPIPS lower than bicubic (or MUSIQ higher), and ours MAPE not
  lower (CI includes or is above 0).
- Everything else is exploratory and is labelled as such.

## 8. Budget and safety rules

- OpenRouter: at most **US$6** in total for this job, tracked in its own ledger
  `logs/openrouter_costs.csv`. The code stops at US$5.9. Nano Banana Pro runs a 1-image pilot (counted
  in the 39) and stops if the projected total exceeds US$5.3.
- GPU server gpu-server: work only in `~/pickglass/genai_sr/`, with all caches under
  `env.sh`. Only **idle** GPUs are used (the `pick_gpu.sh` rule: < 500 MiB used, < 5 % utilisation, no
  compute process). GPU 2 is never used. PIDs are tracked. Nothing is deleted.
- Full-resolution restored PNGs stay on the server (md5 manifest). The Mac copy under `genai_sr/`
  holds code, CSVs, logs, the figure, the full-resolution figure examples, and 768-px JPEG viewing
  copies of every output, because the Mac disk has little space left.

---

## 9. Deviations from the protocol

(Filled in after the runs.)

## 10. Results

(Filled in after the runs.)

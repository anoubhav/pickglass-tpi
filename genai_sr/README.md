# genai_sr: does generative enhancement hallucinate or erase threads?

The generative-enhancement study of the paper for the WACV 2027 workshop WVAQ (workshop topics:
"hallucination effects in image super-resolution and restoration using diffusion models" and "impact
of image quality on CV tasks").

**Status of this file.** Sections 1 to 8 are the protocol. They were written and frozen before any
restoration model was downloaded or run and before any count was computed on a restored image. The
md5 of the original frozen text is in `logs/prereg_md5.txt`, and a copy, redacted for the public
release, is at `logs/README_prereg_frozen.md`. Deviations go in section 9 with a reason, and results
go in section 10. The protocol text above section 9 was not changed after freezing except for the
redactions of the public release (paths, keys, names, internal dates and process notes); no analysis
choice was changed.

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
| **R-LEG** | legacy-13: a1 a2 a4 a5 a6 a8 a10 r2 r3 r4 r6 r7 r8 | the canonical crops `vlm/crops/legacy/{name}.png` (1736–1941 px) | Human hand counts, parsed by the legacy harness exactly as in `paper/tools/legacy_unified.py` (`load_gt`). Axis assignment is "axis", as in the paper: h_count ↔ warp, v_count ↔ weft. |

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
  is run on the Nano Banana Pro output. The SeedVR2 choice is fixed in advance: it is the most-adopted
  open diffusion restorer.

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

- OpenRouter: at most **US$6** in total for this study, tracked in its own ledger
  `logs/openrouter_costs.csv`. The code stops at US$5.9. Nano Banana Pro runs a 1-image pilot (counted
  in the 39) and stops if the projected total exceeds US$5.3.
- GPU server gpu-server: work only in `~/pickglass/genai_sr/`, with all caches under
  `env.sh`. Only **idle** GPUs are used (the `pick_gpu.sh` rule: < 500 MiB used, < 5 % utilisation, no
  compute process).
- Full-resolution restored PNGs stay on the server (md5 manifest). The Mac copy under `genai_sr/`
  holds code, CSVs, logs, the figure, the full-resolution figure examples, and 768-px JPEG viewing
  copies of every output.

---

## 9. Deviations from the protocol

**D1. Additions made after scoring only `clean`, `none` and `bicubic`.** At this
point no generative or SR output had been counted, and only SwinIR / Real-ESRGAN images existed on disk.

**What was found.** Bicubic ×8 up-sampling makes ours lock onto the interpolation *imaging replica*
f_s − f₀ of the yarn frequency (f_s = LR side in cycles/inch). For example, hr1 gives 195.7 instead of
45.4, at confidence 0.47. The R-HR MAPE goes from 15.8 % (`none`) to 239.7 % (`bicubic`). The cause is
that the counter's whitening (radial-median z-scores) amplifies weak periodic structure in the empty
high band of an up-sampled image, and its lattice prior ("the finest periodicity is the yarn") then
prefers the replica. log-Gabor is unaffected (27.9 % vs 27.7 %).

**Why this matters for the protocol.** Bicubic is therefore a poor control for ours. Any restoration
that merely suppresses the replicas would look like a large "improvement" over bicubic.

**Additions, made before any generative output was scored:**

1. A second paired comparison, every restoration **against `none`** (counting the cheap LR image as is,
   which is what a counting pipeline would naturally receive), with the same statistics and Holm
   family. Both comparisons are reported. For ours, the comparison against `none` is the practically
   relevant one.
2. HPEF is decomposed into
   - *sampling-replica* peaks: within the same tolerance of q + f_s·k for a reference lattice point or
     significant reference peak q and k ∈ {−2..2}² \ {0} (aliasing and imaging, predictable from the
     down-sampling);
   - *novel* peaks: neither on the lattice nor a replica. This is invented periodicity proper.

   The pre-registered HPEF (both classes together) is still reported.
3. A raw-power dominant peak (strongest near-axis maximum of log P, not of the whitened z) is reported
   next to the pre-registered whitened one.

**D2. Nano Banana Pro: content-filter refusals and budget.**
- **Refusals.** 4 of the 39 requests (leg_a10 D8, leg_a5 D4, leg_a10 D4J, leg_a2 D4J) came back with
  `finish_reason = content_filter`, no image and US$0 cost. One identical retry (same seed, same prompt)
  was refused again. They are reported as provider refusals (10 %) and are not re-tried with other
  seeds, which would amount to seed shopping. The paired Nano Banana Pro comparisons therefore use
  12 / 12 / 11 images for D8 / D4 / D4J.
- **Cost.** The pilot cost US$0.138 per image (2048² output). The 35 delivered images cost US$4.84.

**D3. Exploratory, not pre-registered: seed sensitivity.** FLUX.2-klein, InvSR and SeedVR2 are re-run
with seeds 1 and 2 (in addition to 20261008) on D4 and D8 for all 66 references. All other settings are
unchanged. This measures how much a count depends on the sampler seed alone.

**D4. Reporting choice, within the protocol.** On clean R-HR crops, log-Gabor and the CNN already
disagree with ours (MAPE 20.8 % and 55.8 % against the ours pseudo-GT). The compact tables therefore
show R-HR for every estimator against its *own* clean prediction, which is the pre-registered secondary
(self-consistency) reference; for ours, this is identical to the primary reference. Every
primary-reference row is in `results/table_counts.csv` (`ref = primary`).

## 10. Results

**Files.**
- `results/genai_hallucination_table.csv`: one row per set × condition × restoration; a copy is at
  `paper/tables/genai_hallucination.csv`, and `paper/tables/genai_hallucination.tex` has the compact D8
  table.
- `results/table_*.csv`: counts, paired tests, drift, silent failures, lattice, image quality, GPT, and
  seed sensitivity.
- `results/run_meta.json`: models, versions, licences, seeds, GPU-hours and spend.
- Figures: `paper/figs/genai_hallucination{,_all}.{pdf,png}`.

**Cost.**
- GPU: 2.26 h on one RTX 4090, using only idle GPUs 0 and 3. That is 0.86 h for the pre-registered
  restorations, 0.44 h for CNN and image-quality scoring, and 0.95 h for the exploratory seed runs.
- OpenRouter: US$5.28 of the US$6 cap. That is US$4.84 for Nano Banana Pro (35 images) and US$0.44 for
  38 GPT-6.1-sol calls.

### 1. Naive up-sampling is the worst "restoration" for a spectral counter

Bicubic ×8 raises ours' MAPE from 7.6 % to 301.7 % on R-LEG and from 15.8 % to 239.7 % on R-HR. Even
×4 doubles it (R-LEG 9.6 → 19.3 %). The counter locks onto the interpolation imaging replica, and does
so with confidence 0.46–0.81.

The CNN and log-Gabor are unaffected. This is a counter-specific failure that the paper's degradation
study could not reveal, because there the counter always saw the native LR pixels. It also applies to
digital zoom and any up-scaled upload. The simple rule "count at native resolution, never up-sample"
avoids it.

### 2. Generative enhancement does not reliably rescue cheap captures (H1)

The test compares each restoration against counting the degraded crop as is (`none`), over 3
estimators × 2 reference sets × 3 conditions:

| Group | Comparisons | Worse | No detectable difference | Better |
|---|---|---|---|---|
| The five generative restorers | 81 | 33 | 41 | 7 |
| SwinIR (non-generative) | 18 | 0 | 14 | 4 |

"Worse" and "better" mean the paired-bootstrap 95 % CI excludes 0. These tallies use each estimator's own clean
prediction as the R-HR reference (the secondary reference; D4 in section 9). With the primary reference, the
generative row reads 21 worse, 45 not different and 15 better. None of these tallies was pre-registered.

The CNN is hurt almost everywhere: 22 of 27 generative comparisons are worse (e.g. Real-ESRGAN R-HR D8
+12.0 points, InvSR R-LEG D8 +11.1). For ours, the effect depends on the model and the set:

| ours, ×8 | R-HR (pseudo-GT) | R-LEG (hand counts) |
|---|---|---|
| SeedVR2 | 3.8 % [1.3, 7.5] vs 15.8 %: helps | 24.1 % vs 7.6 %: n.s., ΔMAPE CI [−0.9, +38.8] |
| Nano Banana Pro | not run | 5.9 % vs 8.0 % on the same 12 crops: helps, ΔMAPE −2.1 [−4.9, −0.1] |
| Real-ESRGAN | 29.1 % vs 15.8 %: worse | n.s. |

Against the pre-registered bicubic control, almost every restoration is "better" for ours, but that
reflects the broken control (D1), not a real benefit.

### 3. Hallucinated periodicity is real but model-specific (H2)

Novel off-lattice spectral energy (neither lattice nor sampling replica) at ×8:

| Restorer | R-HR | R-LEG | Crops with ≥ 1 novel peak |
|---|---|---|---|
| Real-ESRGAN | 13.4 % | 13.3 % | 45–62 % |
| InvSR | 4.0 % | 8.9 % | |
| SeedVR2 | 1.3 % | 2.0 % | |
| FLUX.2-klein | 0.7 % | 2.3 % | |
| Nano Banana Pro | not run | 1.0 % (11.7 % at ×4) | |
| SwinIR | 0.5 % | 0 % | |
| Bicubic | 5.5 % | 9.9 % | |

Against bicubic, the pre-registered HPEF is lower for every generative model, because bicubic's own
replicas dominate. H2 is therefore **not supported against bicubic**. Real-ESRGAN is the only model
with more novel energy than bicubic (R-HR D8, p = 0.055).

Erasure of recoverable lattice peaks is highest for SwinIR (14–16 % at ×8; it over-smooths) and
Real-ESRGAN (19 % on R-HR).

Re-synthesis of lattice peaks above the LR Nyquist limit:
- R-HR: SeedVR2 88 %, SwinIR 65 %, InvSR 50 %, bicubic 19 %.
- R-LEG: 67–100 % for the SR and generative models.

These "correct hallucinations" are harmonics of fundamentals that the LR image still contains. This is
why SeedVR2 can help ours on R-HR.

### 4. "Looks better, counts worse" (H3) holds for the perceptual metric, not uniformly for counts

Every generative output scores higher on no-reference MUSIQ than bicubic (+10 to +28) and at or above
that of the **real high-resolution reference** itself (significantly in 25 of 27 set-degradation
cells; SeedVR2 on R-LEG at ×8 scores 52, the same as the reference). For example, FLUX.2-klein scores
61–63 against 47 (R-HR) and 52 (R-LEG) for the clean crop. At the same time:
- PSNR drops to 19–24 dB for InvSR, FLUX.2-klein, Nano Banana Pro and Real-ESRGAN (at ×8), against
  26–35 dB for bicubic, SwinIR and SeedVR2.
- LPIPS is worse than SwinIR for every generative model, except SeedVR2 on R-HR at ×8 and ×4+JPEG.
- The CNN gets worse in 22 / 27 comparisons, and ours in 3 / 27.

The figure shows the mechanism: Nano Banana Pro and FLUX.2 replace the yarn appearance (woolly,
re-textured yarns, in FLUX's case a re-framed scene) while often keeping the yarn spacing. When the
spacing is kept the count survives; when fine texture is invented (SeedVR2 on legacy a1:
229 instead of 76) the count fails.

### 5. Silent failures (H4)

Of the crops where ours is off by more than 10 % after a generative restoration, 109 / 211 (52 %) still
have confidence ≥ t_c. For the unrestored LR crops the figure is 23 / 49 (47 %). The confidence gate
does not single out hallucinated counts.

### 6. Seed sensitivity (D3, exploratory)

FLUX.2-klein is seed-dependent. With the same input and only the seed changed:
- ours changes by more than 10 % on 21–28 % of R-HR ×8 counts (v / h), and 15–28 % flip between right
  and wrong;
- the CNN changes by more than 10 % on 34–36 % of counts.

SeedVR2 changes by more than 10 % on 6–9 % of counts. InvSR is effectively deterministic (≤ 3 %).

### 7. VLM (GPT-6.1-sol, P1, R-LEG ×8, axis assignment)

| Input | MAPE (%) | 95 % CI |
|---|---|---|
| bicubic (this study) | 11.7 | [6.6, 17.1] |
| earlier stored run (not the one in the paper), LR resized straight to 1536 px | 12.2 | [6.8, 18.0] |
| the paper's run (supplement Tab. S5; `paper/tables/degradation_legacy13.csv`), LR resized straight to 1536 px | 12.6 | [7.2, 18.8] |
| SeedVR2 | 8.6 | [3.4, 14.8] |
| Nano Banana Pro (12 crops) | 7.9 | [4.3, 12.0] |
| clean crop | 8.8 | |

For the VLM, enhancement moves accuracy back towards the clean level. The CIs overlap, and n = 13.

### Limitations

- R-HR uses ours' own clean prediction as pseudo-GT, so it measures self-consistency on images selected
  as confident.
- R-LEG has only 13 crops, so its CIs are wide.
- The degradations are synthetic (down-sampling and JPEG), not real cheap captures.
- The models were run zero-shot with default settings and the protocol prompt.
- Each main run uses one seed.
- Nano Banana Pro was run on R-LEG only, with 4 / 39 refusals.

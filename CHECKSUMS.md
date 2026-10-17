# Checksums of the frozen files

The counter, the scorer and the cascade decisions were frozen before any human count of the 76 PickGlass scenes existed. The combined counter combined-v1 with its pre-registration, the 76-scene generic-IQA test and the analysis of the human counts, `paper/tools/label_day.py`, were frozen before any human count was looked at; `label_day.py` scores every method against the mean of the two annotators' counts, the reference fixed in the analysis plan, and against each annotator alone. The paper's main tables score against the textile-trained reference annotator (A2) instead, a declared deviation decided after both annotators' counts were seen. This file lists the hashes of these files and of the other pre-registration records, so that anyone can check that the files used to score the human counts are the files published here.

The methods octave-v1, ens-v1, octsel-v1 and ens-v1b, the analysis plan for the reference annotator's (A2's) counts and its addendum were fixed before those counts were opened; the method records give the SHA-256 of every file they depend on (section *Commitments* below). The plan and its addendum are not part of this repository; they are available on request, and their SHA-256 and MD5 are listed below.

Verify after cloning: `python3 tools/verify_checksums.py`. It checks the MD5 of the counter, the scorer and the cascade decisions, the SHA-256 of every file in the table below against `tools/freeze_checksums.json` (the same values in machine-readable form), and every released file against `SHA256SUMS`.

| File | SHA-256 (released file) | MD5 | Byte-identical to the frozen original |
|---|---|---|---|
| `methods/results/counting_frozen_final.py` | `fd97f9efe7f5e0ef91116d70c71df577bfa768d4a8fd74524a4d30eeafa6117a` | `3d0520d6bec8a016543ce7e5b353d784` | yes |
| `paper/tables/cascade_150_decisions.csv` | `0c65bc6addd018d228fde594b9d6928394211fe89f9451c6c16c8b4596aa1488` | `57d49a4d4304bc395ed608894009f683` | yes |
| `paper/tools/score150.py` | `007af02bf3b16f8e75c191adfe4eeb283ca2a050143691dc363ca5147126b0c5` | `5879ef44698de7b2a090e8bbee973625` | yes |
| `cnn/runs_aug/PREREG.json` | `dc3c21edacb080c0378a7d8723851e1e1e16bf3f8ba59318123354559c4ac060` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes; original SHA-256 below) |
| `vlm/scaffold/FROZEN.json` | `936508ecff73b74f2ebbfdaeecd2b9c7fcbf1c1502a6f5bb522d55a8fdc75790` | `-` | no: redacted for the release (paths, keys or names; original SHA-256 below) |
| `methods/guard/guard_params.json` | `b31fb9b0408d42f34ee6d101be303d5eac8ef4d61d689165afa09b5cb000df97` | `1182f9e926b09dbcbb63559ed4914d0a` | yes |
| `genai_sr/README.md` | `4a5954fcfa6565f2f1e14014791720b7fe5f6b23c841db43376fa391fac10c77` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes); in the results section (section 10), one sentence (MUSIQ against the real high-resolution reference) corrected to match the paper, one row of the VLM table relabelled as an earlier run and the paper's ×8 run added, and a sentence added on the reference behind the generative tallies; original SHA-256 below |
| `genai_sr/logs/prereg_md5.txt` | `9e21971bd1338bf61c2d2d133230d18fd300b17f559cfb51aacc0f46df0f114d` | `f54565404167e9edd65a9e411020945a` | yes |
| `genai_sr/logs/README_prereg_frozen.md` | `f70150ca16e1214e9c505d33c336193f8bc2492fecd5993a3c911793979614eb` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes; original SHA-256 below) |
| `methods/results/counting_frozen_20261007_1012.py` | `64dbf46e5409faa1ab6f20380dc5f125ac8ba58e50d11612780f0906215a8c63` | `17d2b6655d508285584105e73b605404` | yes |
| `methods/conf_model.json` | `082e01d380ea7619e95b862b111922db136a36aeade528d5c8f7ff678288dafb` | `3aea9b50e4b75e191cac08cb5161ca59` | yes |
| `methods/results/combined_frozen_v1.py` | `81f7d83df3777d767239200314a14f2a9a8c851d1b0985c83ee3d9371843cbe5` | `211e549cf850ddb8bdeedfbb8d9ef4e2` | yes |
| `methods/results/PREREG_combined_v1.json` | `91e17bff585b3fd3cfac50a0c198438cff7b5ee432de4276bdb14aa52ca772fe` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes; one stale supplement cross-reference corrected, supp. S14 to supplement Sec. S11; original SHA-256 below) |
| `paper/tools/iqa150.py` | `be4958ed93a7d9453e6af207f207ba951e321cbae96562c089d5e91cabfb8db6` | `d3782b5cd39784d69415961621070e8a` | yes |
| `paper/tools/iqa_generic.py` | `6db9ccf79a150a9656b0c1494984166d12aa54b5639fe086672dbb04c5f83719` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes in comments; original SHA-256 below) |
| `paper/tables/iqa150_scores.csv` | `4d17ea366346a07fb174f740d142c717da9b77a8b621a8c207eb2305f6afb348` | `61f6da2172ac61bf8b0a344de9d10eec` | yes |
| `paper/tools/label_day.py` | `9ba683de864c2c48044c57033b39c082d5ab4b1904918d591d872b9fcb5828bf` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes in comments and help text; original SHA-256 below) |
| `methods/octave/octave_v1.py` | `5d83de698f84e682128edb16b82a19123dc05f562e65798cf1f9ae5b8be8af3b` | `b8e1c8659dd61f166c69b12904301a0b` | yes |
| `methods/octave/eval_octave_v1.py` | `8d5b71c6e5fd1dcf88dda890c204ad8cd57e81c69a6d26d05758f343ac891d35` | `50c855dbf6c65dcb9700ba4493d6f356` | yes |
| `methods/octave/PREREG_octave_v1.json` | `56a55277d59e595fa867ceaafe0b6259281dba3a94f41d5feb5a8921b12d4a3b` | `5e0944bc14d8e63f77b19e6ca48789be` | yes |
| `methods/octave/predictions_octave_150.csv` | `afb7280345958857026f475a3f3b58ad5b428a05addfce27784cd64349e120f6` | `e137d0ce83385526c8ae010c336aa74a` | yes |
| `methods/octave/octave_v1_150_wide.csv` | `4e7635df9978eb1296dcc9215e84218322c7215c9107c2d9de45f166145d2f9d` | `0b12a55c3f880ca0868220e66b3e6d3d` | yes |
| `methods/ensemble/PREREG_ens_v1.json` | `9cc90deaccf815dfcc7013b6b97f10c2c04dd02c044bd094b1ac2dc334d0749b` | `db58e2cdcb63228dd32ac1ec6eeedfde` | yes |
| `methods/ensemble/make_ens_v1.py` | `b9779ad6727ce23b8dbc3834c9a1aa1ed228d2d9facc172eee7e0dbf9de9349f` | `ceabb83b5a487751f924089f6a2a42ed` | yes |
| `methods/ensemble/predictions_ens_v1_150.csv` | `1120f845a78f284dcca47200d74e3583ceb6308f76a623d8273fc14caf0a1898` | `b29914fdafbe4e9cf86f1ab1588adf3a` | yes |
| `methods/ensemble/cnn_fabric_oof_FLSA_150.csv` | `5f7169b95a479c2857909197e32f57f51d46da7dee5cd723eab569a557498ee3` | `45ffaf730af4190fb88b30e5e2bcb366` | yes |
| `methods/octave_learned/PREREG_octsel_v1.json` | `9479b188a2c9da776a55423622bd05efdbb412cb61b439770f4c670a8795cdf3` | `9eb9a068b3aae87fcee9b697a6e068b1` | yes |
| `methods/octave_learned/octsel_v1.py` | `437aa30bf80008b88fe31d2e5e711ef1320dcc66e6be41260f8207e0ead0e709` | `1a61177825e94d1827565191b25371ea` | yes |
| `methods/octave_learned/octsel_features.py` | `0c1339dd63be45253e1f61a3bfed74d63b4c76a3be24ba5139b67124916489ec` | `8aaa4565c1d9c2a3e98a5b31089a44cd` | yes |
| `methods/octave_learned/model_octsel_v1.json` | `bb0c083361d43d3094f999fe587e8a1530a56c07ef1d039a0dc07d604b16d762` | `258bde5dd6164f6d661e5692b0c6f7ec` | yes |
| `methods/octave_learned/predictions_octsel_oof_FLSA_150.csv` | `c84b06c0efc1c82e6760ed95288bcceeb5fa520840143d7e8e3f65537586b032` | `8b59e6261865f5135f8cdbb45009209d` | yes |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `95b3569dc42ce0ffa4dc857429930263e8e448dd6a5e17d566ffa9c0d82ac12c` | `-` | no: redacted for the release (paths, keys, names, internal dates and process notes; original SHA-256 below) |
| `methods/ensemble_aug/make_ens_v1b.py` | `dc546a2e5d230f73a41eca72c7e038d5837da80e39ca4f8a44b422fbce3a5248` | `9835f8d36722b10a2415bcbc821c1cbd` | yes |
| `methods/ensemble_aug/predictions_ens_v1b_150.csv` | `971ee92038be20a04cdd8d87629b2e32d18c54060609648bf54f2110a7949fae` | `06f680d316b3ff3af63927a8cb08bfbd` | yes |
| `methods/ensemble_aug/cnn_aug1016_fabric_oof_FLSA_150.csv` | `07dc84cbe9956cac688ad6377fb4a329379600268c204e26f45f24e92889c1fe` | `28003f72970a4bdcba48d172ccd33e01` | yes |

SHA-256 of the original (pre-redaction) files, as frozen on the author's disk:

- `cnn/runs_aug/PREREG.json`: `0b61f61ce8bb6a85e1d7eda6e711df83b3ef52275406dfa9f3b04c4f74a281be` (MD5 `e4b8a302cc9a5a3db0e55996c9eeee9c`)
- `vlm/scaffold/FROZEN.json`: `5db8ab3c0d978b3d3c02e7a722ac884fa3cc2c318740bf1642cbdd6f70806dc5` (MD5 `f7c8cc34964b53b0a07f78033fa31a4e`)
- `genai_sr/README.md`: `3bb92043c417694cf4780c58b4022e2c6217a4a154b26538098c2bd2e1f4a9ea` (MD5 `b1d1f0de7e7795f375d79bf31646b0a1`)
- `genai_sr/logs/README_prereg_frozen.md` (the frozen protocol, sections 1-8; its MD5 is the first line of `genai_sr/logs/prereg_md5.txt`): `72a16c8960a1d492cbe75537648c512731e333c8f1139a03793d0b9e39b9e382` (MD5 `0257e8e01f354e90675c4a1d233c9cf4`)
- `methods/results/PREREG_combined_v1.json` (pre-registration of combined-v1 (rule, thresholds, held-out synthetic test, plan for the human-count analysis); frozen with it): `1e4fd17727a0d47fb8a3b3db4accbeef0af143bb5dfdc5dfe0e0ed2deaf5197a` (MD5 `fda00448eebddd30b01cb409f662362f`)
- `paper/tools/iqa_generic.py` (the statistics iqa150.py imports; frozen with it): `4e5683c271341640d59e3b4f5c5dd2697eba39d55fd49182f6c930f87cb2b143` (MD5 `012d177cd3a1a70b4e7b55e30235e9d6`)
- `paper/tools/label_day.py` (the analysis of the human counts: 'dual' protocol, agreement, go/no-go, every method scored against the mean of both annotators and against each alone, every table; frozen before any human count was looked at): `a7151fb9c0f3963e470183f82b16a53632da1c43dc7e17e45f42bfd872a908b4` (MD5 `bc47acde3f7448f7f9cd87c8fbdc2261`)
- `methods/ensemble_aug/PREREG_ens_v1b.json` (ens-v1b (secondary), ens-v1 with the aug1016 CNN; frozen before the reference annotator's counts were opened): `08f5a65d75cdbcd447384a6dd8501ba9913ebc1857921fa412af90dde28c459b` (MD5 `beeff8f3cea2c33f6c54a0ec4720eeb4`)

Other frozen files, released byte-identical:

- `methods/results/combined_frozen_v1.py`: the combined counter combined-v1 (PG-Lattice + log-Gabor); frozen before any human count was looked at
- `paper/tools/iqa150.py`: the pre-specified 76-scene generic-IQA test; frozen before any human count was looked at
- `paper/tables/iqa150_scores.csv`: label-free generic IQA scores of the 150 crops read by iqa150.py; frozen with it
- `methods/octave/PREREG_octave_v1.json`: octave-v1, a label-free octave check; frozen before the reference annotator's counts were opened (its blind test)
- `methods/ensemble/PREREG_ens_v1.json`: ens-v1 = geometric mean of the fabric-grouped CNN and GPT-6.1-sol (the CNN–VLM ensemble of the paper); frozen before the reference annotator's counts were opened
- `methods/octave_learned/PREREG_octsel_v1.json`: octsel-v1, a learned octave selector; frozen before the reference annotator's counts were opened

Pre-registration records not in this repository (available on request), SHA-256 and MD5 of each:

- the analysis plan for the reference annotator's (A2's) counts, fixed before they were opened (cited in the supplement, Sec. S1): `9ae1d9caab117b97f6b9d4eaf4f9887f4e346a609663aefe081be377bfe8664c` (MD5 `0440f5ecc6d7f988fd0d19fbed9dccf5`)
- its addendum 1, fixed before those counts were opened: `586b606c38d6f0f8457c08f234b7d86ef0868d52b66b22ff704ceb5cd7a3752e` (MD5 `4ce2e885f4699a85833a23e9e9cf0660`)
- the recipe of a second ensemble variant (ens-v2), pre-registered before those counts were opened and not run on the final labels: `2d22687c5e15b506088c44e1d754c23c5456f5f58735215a21a8fd09d07d3d21` (MD5 `862191238f5657faa5d0714b21aa9681`)

`methods/octave_learned/PREREG_octsel_v1.json`, released byte-identical, still names the plan and its addendum by their original paths (`handover/results/PREREG_labelday_FLNG*.md`); they are the first two records of this list.

The MD5 values of the three frozen files are the ones disclosed in the paper's supplement (counter `3d0520d6...`, scorer `5879ef44...`, cascade decisions `57d49a4d...`).

## Commitments

Every file that a pre-registration record (`PREREG*.json`) in this repository commits to by SHA-256. "Released redacted" files had absolute paths, keys, names, internal dates or process notes replaced, so their released hash differs; the commitment is to the original. Files "not released yet" carry per-scene human counts, results computed from them, or the cross-validation folds used with them; they are released after review, together with the photos and the human counts, and anyone can then check them against the hash here. Their paths are the ones written in the frozen records, so they are kept as they are.

| Record | File | Committed SHA-256 | Status |
|---|---|---|---|
| `methods/ensemble/PREREG_ens_v1.json` | `methods/ensemble/cnn_fabric_oof_FLSA_150.csv` | `5f7169b95a479c28...` | released, byte-identical |
| `methods/ensemble/PREREG_ens_v1.json` | `methods/ensemble/cnn_fabric_oof_FLSA_raw_predictions.csv` | `86f37b7f541bd3c7...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/ensemble/PREREG_ens_v1.json` | `methods/ensemble/make_ens_v1.py` | `b9779ad6727ce23b...` | released, byte-identical |
| `methods/ensemble/PREREG_ens_v1.json` | `methods/ensemble/predictions_ens_v1_150.csv` | `1120f845a78f284d...` | released, byte-identical |
| `methods/ensemble/PREREG_ens_v1.json` | `methods/octave/eval_octave_v1.py` | `8d5b71c6e5fd1dcf...` | released, byte-identical |
| `methods/ensemble/PREREG_ens_v1.json` | `vlm/results/rect150_P1_predictions_wide.csv` | `7cb5a2c83df5b729...` | released, byte-identical |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `methods/ensemble_aug/cnn_aug1016_fabric_oof_FLSA_150.csv` | `07dc84cbe9956cac...` | released, byte-identical |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `methods/ensemble_aug/cnn_aug1016_fabric_oof_FLSA_raw_predictions.csv` | `aa4dc1e786dde80e...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `methods/ensemble_aug/make_ens_v1b.py` | `dc546a2e5d230f73...` | released, byte-identical |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `methods/ensemble_aug/predictions_ens_v1b_150.csv` | `971ee92038be20a0...` | released, byte-identical |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `methods/octave/eval_octave_v1.py` | `8d5b71c6e5fd1dcf...` | released, byte-identical |
| `methods/ensemble_aug/PREREG_ens_v1b.json` | `vlm/results/rect150_P1_predictions_wide.csv` | `7cb5a2c83df5b729...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `annotation_tool/data/manifest.json` | `a47bab3076359c63...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `cnn/synth.py` | `7d332ebdc757dbeb...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `common/rectify.py` | `a729283352a6ffaa...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/conf_model.json` | `082e01d380ea7619...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/legacy.py` | `f0a2b3ad889b5736...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/candidate_C2_C3_ridge_grid.csv` | `b3e1be4261494c65...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/chaneval.py` | `f0a66e71d125dfd8...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/chanrun.py` | `aa7115d3c19813b4...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/consist.py` | `cd5bbb5a7d37d166...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/devrun.py` | `c57d608b03352346...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/evalrules.py` | `552d0fdfd425e1e4...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/finalsumm.py` | `a1a505c630c89021...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/gen_poplin.py` | `816c8584864a9c1e...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/gen_synth.py` | `861b703b4e4be607...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/hyp2.py` | `d99ffd4b3d9f5c35...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/leanrun.py` | `6427fdda566b16c9...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/modeval.py` | `a1f2325325e740ed...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/modevalh.py` | `207f230f068ffa02...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octave_v1_candidates.json` | `ae583630ad6d9cb6...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octave_v1_dev_final_summary.json` | `34dd9a623cd1bff5...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octave_v1_dev_variants.json` | `4adaf167bd6969a8...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octave_v1h.py` | `b072075fd13cd623...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octfeat.py` | `988bad7997e7286a...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/octrule.py` | `c4d3ec342523d2fe...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/realcons.py` | `de1fa65f7eb98b2d...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/ridgedev.py` | `07190c1dc8a79818...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/ridgeeval.py` | `1f1c4a1c480868eb...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/single.py` | `217806754eefe8ef...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/synth_AB_labels.csv` | `2a28d15b1d0c84b9...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/synth_C_labels.csv` | `d55aca2e6691bab0...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/synthan.py` | `dbca5e6fbf165138...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/dev/variants.py` | `cb9a563bef5ad142...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/eval_octave_v1.py` | `8d5b71c6e5fd1dcf...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/octave_v1.py` | `5d83de698f84e682...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/octave_v1_150_wide.csv` | `4e7635df9978eb12...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/octave/predictions_octave_150.csv` | `afb7280345958857...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/predictions_ours_150.csv` | `ad48068f96be8e97...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/results/counting_frozen_final.py` | `fd97f9efe7f5e0ef...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `methods/results/legacy13_per_image.csv` | `d735a7ae33b988f6...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `paper/tables/fabric_groups_v2.json` | `16fe9af80ab9ee6e...` | released, byte-identical |
| `methods/octave/PREREG_octave_v1.json` | `vlm/results/rect150_P1_predictions_wide.csv` | `7cb5a2c83df5b729...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `common/rectify.py` | `a729283352a6ffaa...` | released redacted; the original on the author's disk matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `handover/results/labelday_diag_FLSA_20261012/cnn_cv_FLSA/fabric_grouped/folds.csv` | `29048b1a83d58e8f...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `handover/results/labelday_diag_FLSA_20261012/payment_audit/labels_per_scene_FL_SA.csv` | `708973fc6ea077ea...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/ensemble/predictions_ens_v1_150.csv` | `1120f845a78f284d...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave/eval_octave_v1.py` | `8d5b71c6e5fd1dcf...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/cv_results_FLSA.json` | `46be0645fc3ad1b3...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/cv_results_FLSA.txt` | `e025fa0446d23918...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/eval_FLSA_oof_with_eval_octave_v1.json` | `d56e6a26ebe8751f...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/eval_FLSA_oof_with_eval_octave_v1.txt` | `d9b88a62e3ff03a7...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/eval_FLSA_per_scene.csv` | `1399d8c3b9eba842...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/features_octsel_150.csv` | `ca0e4fcb228a67d3...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/model_octsel_v1.json` | `bb0c083361d43d30...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/octsel_features.py` | `0c1339dd63be4525...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/octsel_v1.py` | `437aa30bf80008b8...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_argmax_full_150.csv` | `4fc397e0ab08f9d3...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_argmax_oof_FLSA_150.csv` | `89a24a3f5153f665...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_full_150.csv` | `5704df2439e3288e...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_geo_ens_v1_oof_FLSA_150.csv` | `5466488840b1282d...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_geo_gpt_oof_FLSA_150.csv` | `d0d8342d8dbdb26c...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/predictions_octsel_oof_FLSA_150.csv` | `c84b06c0efc1c82e...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/octave_learned/training_table_FLSA.csv` | `b5465b901d9a0ec9...` | not released yet (released after review with the human counts); the original matches the commitment |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/predictions_ours_150.csv` | `ad48068f96be8e97...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `methods/results/counting_frozen_final.py` | `fd97f9efe7f5e0ef...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `paper/tables/fabric_groups_v2.csv` | `6b78d6038511622a...` | released, byte-identical |
| `methods/octave_learned/PREREG_octsel_v1.json` | `vlm/results/rect150_P1_predictions_wide.csv` | `7cb5a2c83df5b729...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `camera_aug/results/stress_legacy13_classical.csv` | `7b4090e458708dc8...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `cnn/synth.py` | `7d332ebdc757dbeb...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/combined_v1.py` | `fb1abf122a5c6d9f...` | released redacted; the original on the author's disk matches the commitment |
| `methods/results/PREREG_combined_v1.json` | `methods/conf_model.json` | `082e01d380ea7619...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/degrade.py` | `3c34d91f07d8a6e6...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/predictions_ours_150.csv` | `ad48068f96be8e97...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/cascade_synth_cal_default.csv` | `42fba01cece10099...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/cascade_synth_cal_legacygrid.csv` | `a96ab72b4ad93bf5...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/combined_frozen_v1.py` | `81f7d83df3777d76...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/combined_v1_150_wide.csv` | `c3849108f2bf03a3...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/combined_v1_fit.json` | `b35fc2301cdcd886...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/combined_v1_fit_grid.csv` | `58a66f37ee6ef477...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/counting_frozen_final.py` | `fd97f9efe7f5e0ef...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_combined_v1_recount.csv` | `264f78f43e755105...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_degrade.csv` | `1421be6b0c3e8928...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_degrade_combined_v1.csv` | `cfe6744837d4af4e...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_degrade_extra.csv` | `894c40c2648872a1...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_degrade_extra_combined_v1.csv` | `fdf941f74bd7ed79...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_estimators.csv` | `919d8b31b980bd31...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/legacy13_vlmcrops_estimators_combined_v1.csv` | `3cab1a1328c59d2a...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/predictions_combined_v1_150.csv` | `b1fe5356ce9de25c...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `methods/results/stress_legacy13_combined_v1.csv` | `c3f3995fb77778e7...` | released, byte-identical |
| `methods/results/PREREG_combined_v1.json` | `paper/tools/score150.py` | `007af02bf3b16f8e...` | released, byte-identical |

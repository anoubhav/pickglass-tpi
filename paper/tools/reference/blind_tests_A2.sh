#!/usr/bin/env bash
# blind_tests_A2.sh -- the pre-registered blind tests against A2, the reference annotator: the frozen
# methods/octave/eval_octave_v1.py (SHA-256 in methods/octave/PREREG_octave_v1.json; executed, never edited) on A2's
# per-scene counts, one JSON + TXT per test.
#   main Sec. 5.1: the CNN-VLM ensemble (ens-v1) minus GPT-6.1-sol P1 (pre-registered primary, PREREG_ens_v1.json) and
#                  the CNN trained on A1 (fabric-grouped) minus GPT;
#   main Sec. 5.2 / supplement Sec. S17: the octave check (octave-v1) minus PG-Lattice and the learned selector
#                  (octsel-v1, PREREG_octsel_v1.json) with its pre-registered secondary endpoints;
#   supplement Sec. S1: ens-v1b (PREREG_ens_v1b.json) against GPT-6.1-sol and against ens-v1.
# Every test: scene MAPE (mean of the h and v APE; a missing prediction counts 100%), scene bootstrap 10,000 resamples,
# numpy default_rng(20261012), Wilcoxon and exact McNemar tests (see the docstring of eval_octave_v1.py). "--pred X" is
# printed under the label 'octave-v1'; "--frozen Y" replaces the PG-Lattice slot by Y.
#
#   usage: bash paper/tools/reference/blind_tests_A2.sh [A2 per-scene CSV] [output dir]
#          (defaults: data/human_counts/labels_per_scene_A2.csv, paper/tools/reference/out; python: $PYTHON or python3)
# Outputs: <output dir>/eval_A2/<test>.json and .txt.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
LAB=${1:-$ROOT/data/human_counts/labels_per_scene_A2.csv}
OUT=${2:-$HERE/out}/eval_A2
case "$LAB" in /*) ;; *) LAB="$PWD/$LAB" ;; esac
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
[ -f "$LAB" ] || { echo "A2 per-scene counts not found: $LAB" >&2; exit 1; }
mkdir -p "$OUT"
PY=${PYTHON:-python3}
OL=methods/octave_learned
cd "$ROOT"
run() {  # name, then extra args
  n=$1; shift
  "$PY" -I -B methods/octave/eval_octave_v1.py --labels "$LAB" --names A2 --reference each "$@" \
      --out-json "$OUT/$n.json" --out-txt "$OUT/$n.txt" > /dev/null
  echo "blind eval $n: exit $?"
}
# the five blind tests run when the reference counts were opened
run octave_v1   --pred methods/octave/predictions_octave_150.csv
run ens_v1      --pred methods/ensemble/predictions_ens_v1_150.csv
run cnn_fabric  --pred methods/ensemble/cnn_fabric_oof_FLSA_150.csv
run octsel_v1   --pred $OL/predictions_octsel_oof_FLSA_150.csv
run ens_v1b     --pred methods/ensemble_aug/predictions_ens_v1b_150.csv
# pre-registered secondary endpoints (ens-v1b minus ens-v1; octsel-v1 against ens-v1 and its variants)
run ens_v1b_vs_ens_v1      --pred methods/ensemble_aug/predictions_ens_v1b_150.csv --frozen methods/ensemble/predictions_ens_v1_150.csv
run octsel_v1_vs_ens_v1    --pred $OL/predictions_octsel_oof_FLSA_150.csv --frozen methods/ensemble/predictions_ens_v1_150.csv
run octsel_argmax_oof      --pred $OL/predictions_octsel_argmax_oof_FLSA_150.csv
run octsel_full            --pred $OL/predictions_octsel_full_150.csv
run octsel_geo_gpt_oof     --pred $OL/predictions_octsel_geo_gpt_oof_FLSA_150.csv
run octsel_geo_ens_v1_oof  --pred $OL/predictions_octsel_geo_ens_v1_oof_FLSA_150.csv

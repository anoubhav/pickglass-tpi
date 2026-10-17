#!/usr/bin/env bash
# run_all.sh -- every script of paper/tools/reference in dependency order, on the two annotators' per-scene counts.
#   usage: bash paper/tools/reference/run_all.sh [A1 per-scene CSV] [A2 per-scene CSV] [output dir]
#          (defaults: data/human_counts/labels_per_scene_A1.csv, data/human_counts/labels_per_scene_A2.csv,
#           paper/tools/reference/out; python: $PYTHON or python3)
# Each script's console output goes to <output dir>/logs/<script>.txt; see README.md for what each one makes.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
A1=${1:-$ROOT/data/human_counts/labels_per_scene_A1.csv}
A2=${2:-$ROOT/data/human_counts/labels_per_scene_A2.csv}
OUT=${3:-$HERE/out}
case "$A1" in /*) ;; *) A1="$PWD/$A1" ;; esac
case "$A2" in /*) ;; *) A2="$PWD/$A2" ;; esac
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
for f in "$A1" "$A2"; do [ -f "$f" ] || { echo "per-scene counts not found: $f" >&2; exit 1; }; done
PY=${PYTHON:-python3}
mkdir -p "$OUT/logs"
step() {  # script name, then its arguments
  s=$1; shift
  "$PY" -B "$HERE/$s.py" "$@" > "$OUT/logs/$s.txt" 2>&1 || { echo "$s: FAILED (see $OUT/logs/$s.txt)" >&2; exit 1; }
  echo "$s: done"
}
L=(--a1 "$A1" --a2 "$A2")
step reshoot_consistency "${L[@]}" --out "$OUT"
step molmo_reshoot --out "$OUT"
step human_floor_paired "${L[@]}" --out "$OUT"
step logratio_endpoint "${L[@]}" --out "$OUT"
step branchB_numbers "${L[@]}" --out "$OUT"
step make_disputed_table "${L[@]}" --out "$OUT"
step rule1_histogram "${L[@]}" --out "$OUT"
step rule4_band "${L[@]}" --out "$OUT"
step make_tab1_ref "${L[@]}" --out "$OUT"
step paired_ref "${L[@]}" --out "$OUT"
step make_branchB_macros "${L[@]}" --out "$OUT"
step expert_iqa "${L[@]}" --out "$OUT"
step iqa_disagreement_mwu "${L[@]}"
step ruler_vs_ours_ref --a2 "$A2"
rm -f "$OUT/fig/fig_octave_hist_H2.pdf" "$OUT/fig/fig_octave_hist_H2.png"   # the figure script never overwrites
step make_fig_octave_hist "${L[@]}" --out "$OUT/fig"
PYTHON="$PY" bash "$HERE/blind_tests_A2.sh" "$A2" "$OUT" > "$OUT/logs/blind_tests_A2.txt" 2>&1 \
    || { echo "blind_tests_A2: FAILED (see $OUT/logs/blind_tests_A2.txt)" >&2; exit 1; }
echo "blind_tests_A2: done"
echo "outputs in $OUT"

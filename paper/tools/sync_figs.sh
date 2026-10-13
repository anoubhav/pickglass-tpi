#!/bin/sh
# Copy the paper figures produced by the methods/ and vlm/ jobs into paper/figs/
# so that paper/ is self-contained (e.g. for an Overleaf upload).
# Re-run after any make_figs.py run. Sources are only read.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$HERE")"
mkdir -p "$HERE/figs"
for f in fig_lattice fig_lattice_failures fig_profile fig_degradation_drift fig_conf_vs_drift fig_legacy_methods fig_degradation_examples; do
  cp "$ROOT/methods/figs/$f.pdf" "$HERE/figs/$f.pdf"
done
cp "$ROOT/vlm/figs/mae_by_model_prompt.pdf" "$HERE/figs/vlm_mae_by_model_prompt.pdf"
echo "figures synced into $HERE/figs"
ls -la "$HERE/figs"

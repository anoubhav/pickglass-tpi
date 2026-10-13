#!/bin/bash
# Molmo2-8B (bf16, transformers) queue on ONE idle GPU.  Resumable: re-run to continue, finished
# (image, view, prompt, repeat) records are skipped.   usage: run_main.sh main [w4mlx]
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}" || exit 1
source ./env.sh
echo $$ > run/molmo_queue.pid
G=$(./pick_gpu.sh) || { echo "$(date) no idle GPU among 0,1,3"; exit 1; }
echo "$(date) using GPU $G"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=$G
for plan in "$@"; do
  variant=bf16; [ "$plan" = w4mlx ] && variant=w4mlx
  echo "### $(date) plan $plan variant $variant"
  venv/bin/python code/run_molmo2_hf.py --plan "$plan" --variant "$variant" &
  echo $! > run/molmo_python.pid
  wait $!
  echo "### $(date) plan $plan exit $?"
done
echo "### $(date) queue finished"
rm -f run/molmo_python.pid

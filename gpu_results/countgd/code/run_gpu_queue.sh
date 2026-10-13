#!/bin/bash
# usage: run_gpu_queue.sh <gpu> "<args1>" "<args2>" ...
# Runs run_countgd.py jobs sequentially on one GPU (resumable: finished (image, method, part)
# records in outputs/raw are skipped).  All caches/temp stay inside $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}
# (env.sh); CountGD itself runs fully offline from local checkpoints.  The queue writes its
# process-group id to countgd/run/queue_gpu<G>.pid so it can be stopped with
#   kill -- -$(cat countgd/run/queue_gpu<G>.pid)        (never pkill -f)
GPU=$1; shift
source "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh"
CG=$FW/countgd
mkdir -p "$CG/run"
echo $$ > "$CG/run/queue_gpu$GPU.pid"
cd "$CG/code" || exit 1
export HF_HOME=$CG/hf_home HF_HUB_CACHE=$CG/hf_home/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
for job in "$@"; do
  echo "### GPU $GPU: run_countgd.py $job  $(date)"
  nvidia-smi -i "$GPU" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  "$FW/venv_countgd/bin/python" run_countgd.py $job 2>&1 | grep --line-buffered -v -i "warn\|load tokenizer\|text_encoder_type"
done
echo "### GPU $GPU queue finished $(date)"
rm -f "$CG/run/queue_gpu$GPU.pid"

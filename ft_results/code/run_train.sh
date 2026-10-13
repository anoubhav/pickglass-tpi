#!/bin/bash
# LoRA fine-tune on 2 idle GPUs (DDP).  Resumable: re-run the same command to continue.
# usage: run_train.sh GPUS(e.g. 0,3) OUTDIR [extra train_lora.py args]
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}" || exit 1
source ./env.sh
export SYNTH_PROFILE=v2
GPUS=$1; OUT=$2; shift 2
NP=$(awk -F, "{print NF}" <<<"$GPUS")
export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=4
cd ft/code
echo "$(date) start GPUs=$GPUS nproc=$NP out=$OUT args=$*"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
../../venv/bin/torchrun --nproc_per_node $NP --master_port 29517 train_lora.py --out $OUT "$@"
echo "$(date) exit $?"

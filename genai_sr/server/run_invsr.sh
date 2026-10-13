#!/bin/bash
# InvSR (CVPR 2025), 1 step, default config except seed=20261008 (README section 4).
# Usage: CUDA_VISIBLE_DEVICES=<idle> bash code/run_invsr.sh
set -e
source $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh
G=$FW/genai_sr
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "set CUDA_VISIBLE_DEVICES"; exit 1; }
cd $G/repos/InvSR
mkdir -p $G/models/invsr
sed 's/^seed: 12345/seed: 20261008/' configs/sample-sd-turbo.yaml > $G/code/invsr_sample-sd-turbo_seed20261008.yaml
grep -q "^seed: 20261008" $G/code/invsr_sample-sd-turbo_seed20261008.yaml
CK=$G/models/invsr/noise_predictor_sd_turbo_v5.pth
[ -f $CK ] || curl -sL -o $CK https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth
for C in D4 D8 D4J; do
  OUT=$G/outputs/_invsr_raw/$C
  mkdir -p $OUT
  if [ "$(ls $OUT | wc -l)" -ge "$(ls $G/inputs/lr/$C | wc -l)" ]; then echo "skip $C (done)"; continue; fi
  if [ -f $G/run/invsr_$C.lock ]; then echo "skip $C (running elsewhere)"; continue; fi
  touch $G/run/invsr_$C.lock
  /usr/bin/time -v $G/venv_invsr/bin/python inference_invsr.py -i $G/inputs/lr/$C -o $OUT --num_steps 1 \
     --cfg_path $G/code/invsr_sample-sd-turbo_seed20261008.yaml --sd_path $G/models/invsr/sd_turbo \
     --started_ckpt_path $CK 2>&1 | tail -n 25
  mv $G/run/invsr_$C.lock $G/run/invsr_$C.done
done
md5sum $CK > $G/logs/invsr_weights_md5.txt

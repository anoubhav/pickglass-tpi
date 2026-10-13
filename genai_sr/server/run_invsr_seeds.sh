#!/bin/bash
# EXPLORATORY seed sensitivity for InvSR: same as run_invsr.sh, seeds 1 and 2, conditions D4 and D8 only.
set -e
source $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh
G=$FW/genai_sr
cd $G/repos/InvSR
CK=$G/models/invsr/noise_predictor_sd_turbo_v5.pth
for S in 1 2; do
  sed "s/^seed: 12345/seed: $S/" configs/sample-sd-turbo.yaml > $G/code/invsr_sample-sd-turbo_seed$S.yaml
  for C in D4 D8; do
    OUT=$G/outputs/_invsr_s${S}_raw/$C
    mkdir -p $OUT
    if [ "$(ls $OUT | wc -l)" -ge "$(ls $G/inputs/lr/$C | wc -l)" ]; then echo "skip $S $C"; continue; fi
    $G/venv_invsr/bin/python inference_invsr.py -i $G/inputs/lr/$C -o $OUT --num_steps 1 \
       --cfg_path $G/code/invsr_sample-sd-turbo_seed$S.yaml --sd_path $G/models/invsr/sd_turbo --started_ckpt_path $CK 2>&1 | tail -n 3
  done
  $G/venv_gen/bin/python $G/code/finalize_raw.py invsr_s$S
done

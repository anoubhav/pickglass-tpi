#!/bin/bash
# Sequential GPU queue for genai_sr. Before EACH step waits for an idle GPU (../pick_gpu.sh: GPUs 0,1,3 only,
# <500 MiB used, <5% util, no compute process); logs step, GPU, PID, start/end to logs/gpu_usage.csv.
# Usage: bash code/gpu_queue.sh step1 step2 ...   (steps: simple seedvr2 flux2klein invsr cnn iq)
source $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh
G=$FW/genai_sr
cd $G
export WVAQ_ROOT=$G/wvaq_copy
[ -f logs/gpu_usage.csv ] || echo "step,gpu,pid,start_utc,end_utc,seconds,exit" > logs/gpu_usage.csv
for step in "$@"; do
  until GPU=$(bash $FW/pick_gpu.sh); do sleep 60; done
  export CUDA_VISIBLE_DEVICES=$GPU
  t0=$(date +%s); s0=$(date -u +%FT%TZ)
  case $step in
    simple)     venv_gen/bin/python code/restore_simple.py --restorers swinir,realesrgan > logs/step_simple.log 2>&1 & ;;
    seedvr2)    venv_gen/bin/python code/run_seedvr2.py > logs/step_seedvr2.log 2>&1 & ;;
    flux2klein) venv_gen/bin/python code/run_flux2klein.py > logs/step_flux2klein.log 2>&1 & ;;
    invsr)      (bash code/run_invsr.sh && venv_gen/bin/python code/finalize_raw.py invsr) > logs/step_invsr.log 2>&1 & ;;
    cnn)        venv_eval/bin/python code/score_cnn.py --restorers $CNN_RESTORERS > logs/step_cnn.log 2>&1 & ;;
    iq)         venv_eval/bin/python code/iq_metrics.py --restorers $IQ_RESTORERS > logs/step_iq.log 2>&1 & ;;
    *) echo "unknown step $step"; continue ;;
  esac
  PID=$!
  echo "$step $GPU $PID" > run/current_step.txt
  wait $PID; rc=$?
  t1=$(date +%s)
  echo "$step,$GPU,$PID,$s0,$(date -u +%FT%TZ),$((t1-t0)),$rc" >> logs/gpu_usage.csv
done
echo "QUEUE_DONE $(date -u +%FT%TZ)" >> logs/gpu_queue.log

#!/bin/bash
# prints the first of the preferred GPUs (0,1,3) that is idle: <500 MiB used, <5% util, no compute process
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
for i in 0 1 3; do
  read -r used util uuid < <(nvidia-smi -i $i --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits | tr -d ",")
  if [ "$used" -lt 500 ] && [ "$util" -lt 5 ] && ! grep -q "$uuid" <<<"$apps"; then echo $i; exit 0; fi
done
exit 1

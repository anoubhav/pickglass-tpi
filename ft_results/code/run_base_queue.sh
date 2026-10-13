#!/bin/bash
# base Molmo2-4B baselines on one GPU: synthetic val (100) then rect150; resumable
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/ft" || exit 1
G=$1
./run_infer.sh $G molmo2_4b_base none synval P1,P4,P4q,P4s --n-syn 100 --batch 8
./run_infer.sh $G molmo2_4b_base none rect150 P1,P4,P4q,P4s --batch 8
echo BASE_QUEUE_DONE

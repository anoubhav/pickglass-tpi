#!/bin/bash
# waits for run1 to finish, then evaluates the FINAL checkpoint (pre-declared) on two GPUs:
#   GPU A: synthetic val (100) -> rect150 shard 0/2     GPU B: legacy (20 crops) -> rect150 shard 1/2
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/ft" || exit 1
GA=$1; GB=$2
until grep -q "finished at step" ckpt/run1/train_rank0.log 2>/dev/null; do sleep 60; done
sleep 30
AD=../ckpt/run1/$(python3 -c "import json;print(json.load(open(\"ckpt/run1/latest.json\"))[\"dir\"])")
echo "$(date) final adapter $AD"
T=molmo2_4b_synthft
( ./run_infer.sh $GA $T $AD synval P1,P4,P4q,P4s --n-syn 100 --batch 12 > logs/infer_ft_synval.log 2>&1
  ./run_infer.sh $GA $T $AD rect150 P1,P4,P4q,P4s --batch 12 --shard 0/2 > logs/infer_ft_rect150_s0.log 2>&1 ) &
PA=$!
( ./run_infer.sh $GB $T $AD legacy P1,P4,P4q,P4s --batch 12 > logs/infer_ft_legacy.log 2>&1
  ./run_infer.sh $GB $T $AD rect150 P1,P4,P4q,P4s --batch 12 --shard 1/2 > logs/infer_ft_rect150_s1.log 2>&1 ) &
PB=$!
echo "$PA $PB" > run/ft_eval_children.pid
wait $PA $PB
echo "$(date) FT_EVAL_DONE"

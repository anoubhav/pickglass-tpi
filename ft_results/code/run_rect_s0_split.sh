#!/bin/bash
# rect150 shard-0 ids (even) split over GPU 3 (after the shard-1 run, pid 3603377, exits) and GPU 0 (after synval, pid 3591860)
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/ft" || exit 1
AD=../ckpt/run1/step01000; T=molmo2_4b_synthft
( while kill -0 3603377 2>/dev/null; do sleep 20; done; ./run_infer.sh 3 $T $AD rect150 P1,P4,P4q,P4s --batch 12 --ids 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148 --shard 1/2 > logs/infer_ft_rect150_s0b.log 2>&1 ) &
( while kill -0 3591860 2>/dev/null; do sleep 20; done; ./run_infer.sh 0 $T $AD rect150 P1,P4,P4q,P4s --batch 12 --ids 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148 --shard 0/2 > logs/infer_ft_rect150_s0a.log 2>&1 ) &
wait
echo "$(date) RECT_DONE"

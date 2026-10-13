#!/bin/zsh
# v2 benchmark chain for real150: build (sensor at the profile's own px/inch) -> ours on the renders -> audit summary.
# The isolated-stage audit and the px-per-yarn sweep do not depend on the v1/v2 change (they run at the
# source pitch or at an explicit pitch below it) and are reused.
cd "$(dirname "$0")/."
V="${PICKGLASS_PYTHON:-$(command -v python3)}"
W=${WORKERS:-4}
$V -u build_stress.py --set real150 --workers $W >> logs/build_real150_v2.log 2>&1 || exit 1
echo "v2 build real150 done $(date)" >> logs/chain.log
$V -u eval_stress_classical.py --set real150 --ests ours --workers $W >> logs/eval_real150_classical_v2.log 2>&1 || exit 1
echo "v2 eval real150 done $(date)" >> logs/chain.log
$V -u label_audit.py summarise > logs/audit_summary_v2.log 2>&1
echo "v2 summary done $(date)" >> logs/chain.log

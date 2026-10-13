#!/bin/zsh
# Background chain for the real150 part (build -> sweep -> ours on the stress renders -> isolated-stage audit
# -> summary).  Every step is resumable; re-running the chain continues where it stopped.
set -e
cd "$(dirname "$0")/."
V="${PICKGLASS_PYTHON:-$(command -v python3)}"
W=${WORKERS:-4}
$V -u build_stress.py --set real150 --workers $W >> logs/build_real150.log 2>&1
echo "build done $(date)" >> logs/chain.log
$V -u label_audit.py run --part sweep --workers $W >> logs/audit_sweep.log 2>&1
echo "sweep done $(date)" >> logs/chain.log
$V -u eval_stress_classical.py --set real150 --ests ours --workers $W >> logs/eval_real150_classical.log 2>&1
echo "eval real150 done $(date)" >> logs/chain.log
$V -u label_audit.py run --part real150 --workers $W >> logs/audit_real150.log 2>&1
echo "audit real150 done $(date)" >> logs/chain.log
$V -u label_audit.py summarise > logs/audit_summary.log 2>&1
echo "summary done $(date)" >> logs/chain.log

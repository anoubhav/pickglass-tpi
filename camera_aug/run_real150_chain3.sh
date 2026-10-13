#!/bin/zsh
# Continuation of run_real150_chain.sh with more workers once the legacy-13 evaluations had finished:
# waits for the running real150 build (pid given as $1), then runs the remaining resumable steps.
cd "$(dirname "$0")/."
V="${PICKGLASS_PYTHON:-$(command -v python3)}"
W=${WORKERS:-5}
# (no build to wait for)

$V -u build_stress.py --set real150 --workers $W >> logs/build_real150.log 2>&1 || exit 1
echo "build done $(date)" >> logs/chain.log
$V -u label_audit.py run --part sweep --workers $W >> logs/audit_sweep.log 2>&1 || exit 1
echo "sweep done $(date)" >> logs/chain.log
$V -u eval_stress_classical.py --set real150 --ests ours --workers $W >> logs/eval_real150_classical.log 2>&1 || exit 1
echo "eval real150 done $(date)" >> logs/chain.log
$V -u label_audit.py run --part real150 --workers $W >> logs/audit_real150.log 2>&1 || exit 1
echo "audit real150 done $(date)" >> logs/chain.log
$V -u label_audit.py summarise > logs/audit_summary.log 2>&1
echo "summary done $(date)" >> logs/chain.log

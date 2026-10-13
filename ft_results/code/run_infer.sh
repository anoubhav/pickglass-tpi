#!/bin/bash
# usage: run_infer.sh GPU TAG ADAPTER SETS METHODS [extra args]   (resumable)
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}" || exit 1
source ./env.sh
G=$1; TAG=$2; AD=$3; SETS=$4; METH=$5; shift 5
export CUDA_VISIBLE_DEVICES=$G
cd ft/code
exec ../../venv/bin/python infer_ft.py --tag $TAG --adapter $AD --sets $SETS --methods $METH "$@"

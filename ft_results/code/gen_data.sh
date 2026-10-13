#!/bin/bash
# synthetic point-labelled data: val (400, seeds 9100000+) then train (12000, seeds 3000000+); resumable
cd "$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}" || exit 1
source ./env.sh
export SYNTH_PROFILE=v2
cd ft/code
../../venv/bin/python synth_pts.py --out ../data/val --n 400 --seed0 9100000 --workers 20
../../venv/bin/python synth_pts.py --out ../data/train --n 12000 --seed0 3000000 --workers 20
echo GEN_ALL_DONE

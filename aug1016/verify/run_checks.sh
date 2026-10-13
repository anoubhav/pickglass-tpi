#!/bin/sh
# Independent verification of aug1016 (checks 1-4). Writes only under aug1016/verify/.
#   VENV2=<.../venv2/bin/python> VCNN=<.../venv_cnn/bin/python> sh run_checks.sh [--with-ncc]
set -e
cd "$(dirname "$0")"
if [ "$1" = "--with-ncc" ]; then
  "$VENV2" v1_make_sheets.py && "$VENV2" v1_ncc.py      # ~14 min: re-rectify + NCC on all 853
fi
"$VENV2" v1_compare.py                                    # check 1 (eye_calls.csv is the manual part)
"$VENV2" test_swap_synthetic.py                           # check 2
"$VENV2" test_propagate_fake3.py                          # check 3
"$VCNN"  test_train_cv_folds.py                           # check 4

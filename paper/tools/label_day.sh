#!/bin/sh
# label_day.sh -- one-command label day (thin wrapper around tools/label_day.py; all options pass through).
#
#   tools/label_day.sh --out-root <scratch>/ld                    # DRY RUN on the disk autosave (writes only under <scratch>/ld)
#   tools/label_day.sh --out-root <scratch>/ld ~/Downloads/annotations_FL_SA_*.json ~/Downloads/annotations_FL_SA_*.csv
#   tools/label_day.sh --real --cnn launch                         # final labels: real tables/PDFs/report; CNN CV on gpu-server
#   tools/label_day.sh --real --cnn-collect latest                 # later: pull the CNN runs, re-score, rebuild, report
#   tools/label_day.sh --help
#
# The Python needs numpy<2/pandas/scipy/matplotlib. Default (Oct 11): the DURABLE venv wvaq2027/.tools/venv (the old
# venv2 in /tmp was emptied by macOS's /tmp cleaner); override with LABEL_DAY_PY=/path/to/python.
# Tectonic + its cache default to wvaq2027/.tools/bin/tectonic and .tools/tectonic_cache (inside label_day.py).
# Other overrides: LABEL_DAY_TECTONIC, LABEL_DAY_TECTONIC_CACHE, LABEL_DAY_PDF_PY, LABEL_DAY_CNN_PY (see label_day.py).
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS="$(cd "$HERE/../.." && pwd)/.tools"
WF_VENV="${PICKGLASS_PYTHON:-$(command -v python3)}"
CHECK="import numpy, pandas, matplotlib; from scipy import stats; assert int(numpy.__version__.split('.')[0]) < 2"
PY=""
# first candidate that really imports the stack (a half-emptied venv still passes a plain -x test)
for c in ${LABEL_DAY_PY:+"$LABEL_DAY_PY"} "$TOOLS/venv/bin/python" "$WF_VENV"; do
  if [ -x "$c" ] && PYTHONDONTWRITEBYTECODE=1 "$c" -c "$CHECK" >/dev/null 2>&1; then PY="$c"; break; fi
  [ -n "$LABEL_DAY_PY" ] && [ "$c" = "$LABEL_DAY_PY" ] && echo "label_day.sh: LABEL_DAY_PY=$c cannot import numpy<2/pandas/scipy/matplotlib; trying the defaults" >&2
done
if [ -z "$PY" ]; then
  echo "label_day.sh: no python with numpy<2, pandas, scipy, matplotlib (tried LABEL_DAY_PY, $TOOLS/venv, the wf_paper venv); recreate:" >&2
  echo "  python3.11 -m venv $TOOLS/venv && $TOOLS/venv/bin/python -m pip install -r $TOOLS/requirements.txt" >&2
  exit 2
fi
PYTHONDONTWRITEBYTECODE=1 exec "$PY" "$HERE/label_day.py" "$@"

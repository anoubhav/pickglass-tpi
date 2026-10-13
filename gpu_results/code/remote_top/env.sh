# source this before EVERY python/pip/huggingface command: all caches/temp stay inside
# $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass} (never ~/.cache, ~/.local or /tmp) and no HF token is ever used
export FW=$HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}
export PIP_CACHE_DIR=$FW/cache/pip
export HF_HOME=$FW/hf_home
export HF_HUB_CACHE=$FW/hf_home/hub
export TORCH_HOME=$FW/cache/torch
export XDG_CACHE_HOME=$FW/cache/xdg
export TRITON_CACHE_DIR=$FW/cache/triton
export CUDA_CACHE_PATH=$FW/cache/cuda
export TMPDIR=$FW/tmp
export MPLCONFIGDIR=$FW/cache/matplotlib
export PYTHONPYCACHEPREFIX=$FW/cache/pycache
export PYTHONNOUSERSITE=1
export PIP_USER=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $PIP_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME $TRITON_CACHE_DIR $CUDA_CACHE_PATH $TMPDIR $MPLCONFIGDIR $PYTHONPYCACHEPREFIX
# fontconfig (spawned by matplotlib fc-list) also writes ~/.fontconfig by default -> private config whose only cachedir is ours
export FONTCONFIG_FILE=$FW/cache/fonts.conf
mkdir -p $FW/cache/fontconfig

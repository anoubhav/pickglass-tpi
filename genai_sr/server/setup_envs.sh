#!/bin/bash
# genai_sr server setup (gpu-server). Creates NEW venvs inside $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/genai_sr only;
# all caches via ../env.sh. Never touches existing venvs. Usage: bash setup_envs.sh gen|eval|invsr
set -e
source $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh
G=$FW/genai_sr
cd $G
TORCH_IDX=https://download.pytorch.org/whl/cu118
case "$1" in
gen)   # SeedVR2 (numz CLI), FLUX.2-klein (diffusers), SwinIR / Real-ESRGAN (spandrel)
  python3 -m venv --without-pip $G/venv_gen && $G/venv_gen/bin/python $FW/get-pip.py -q
  P=$G/venv_gen/bin/pip
  $P install -q --upgrade pip
  $P install -q torch==2.6.0 torchvision==0.21.0 --index-url $TORCH_IDX
  $P install -q "diffusers>=0.37" "transformers>=4.56" accelerate safetensors "peft>=0.17" "rotary_embedding_torch>=0.5.3" \
      "omegaconf>=2.3.0" gguf einops opencv-python-headless==4.11.0.86 spandrel sentencepiece psutil tqdm matplotlib numpy pandas
  $G/venv_gen/bin/python -c "import torch, diffusers, transformers, spandrel; print('gen OK', torch.__version__, diffusers.__version__, transformers.__version__, spandrel.__version__); from diffusers import Flux2KleinPipeline"
  ;;
eval)  # counters (ours / gabor), CNN, LPIPS, pyiqa, skimage -- numpy/cv2 pinned to the Mac versions
  python3 -m venv --without-pip $G/venv_eval && $G/venv_eval/bin/python $FW/get-pip.py -q
  P=$G/venv_eval/bin/pip
  $P install -q --upgrade pip
  $P install -q torch==2.6.0 torchvision==0.21.0 --index-url $TORCH_IDX
  $P install -q numpy==1.26.4 opencv-python-headless==4.11.0.86 scipy pandas scikit-image lpips pyiqa openpyxl matplotlib
  $G/venv_eval/bin/python -c "import numpy, cv2, scipy, torch, lpips, pyiqa, skimage; print('eval OK', numpy.__version__, cv2.__version__, scipy.__version__, torch.__version__, pyiqa.__version__, skimage.__version__)"
  ;;
invsr) # InvSR (CVPR 2025): its own vendored diffusers + pinned transformers
  mkdir -p $G/repos && cd $G/repos
  [ -d InvSR ] || git clone -q https://github.com/zsyOAOA/InvSR.git
  cd InvSR && git rev-parse HEAD > $G/logs/invsr_commit.txt
  python3 -m venv --without-pip $G/venv_invsr && $G/venv_invsr/bin/python $FW/get-pip.py -q
  P=$G/venv_invsr/bin/pip
  $P install -q --upgrade pip
  $P install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
  $P install -q xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
  $P install -q -e ".[torch]"
  $P install -q -r requirements.txt
  $G/venv_invsr/bin/python -c "import torch, diffusers, transformers; print('invsr OK', torch.__version__, diffusers.__version__, transformers.__version__)"
  ;;
esac

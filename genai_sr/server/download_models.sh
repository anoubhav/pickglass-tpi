#!/bin/bash
# CPU/network only: pre-download all open model weights into genai_sr/models or $HF_HOME (inside FW).
source $HOME/${PICKGLASS_REMOTE_ROOT:-pickglass}/env.sh
G=$FW/genai_sr
cd $G
set -x
# SeedVR2-3B fp16 + VAE (via the CLI's own downloader, integrity-checked)
(cd repos/seedvr2_numz && CUDA_VISIBLE_DEVICES="" $G/venv_gen/bin/python -c "
import sys; sys.argv=['x','--dit_model','seedvr2_ema_3b_fp16.safetensors']
sys.path.insert(0,'.')
from src.utils.downloads import download_weight
from src.utils.model_registry import DEFAULT_VAE
from src.utils.debug import Debug
print(download_weight(dit_model='seedvr2_ema_3b_fp16.safetensors', vae_model=DEFAULT_VAE, model_dir='$G/models/SEEDVR2', debug=Debug(enabled=True)))
")
# FLUX.2 [klein] 4B (Apache-2.0, not gated)
$G/venv_gen/bin/python -c "
from huggingface_hub import snapshot_download, HfApi
sha=HfApi().model_info('black-forest-labs/FLUX.2-klein-4B').sha; print('flux2 sha', sha)
print(snapshot_download('black-forest-labs/FLUX.2-klein-4B', revision=sha, ignore_patterns=['flux-2-klein-4b.safetensors','*.jpg','*.png']))
"
# InvSR: SD-Turbo (diffusers layout, no single-file ckpt) + noise predictor
$G/venv_invsr/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('stabilityai/sd-turbo', cache_dir='$G/models/invsr/sd_turbo', ignore_patterns=['sd_turbo.safetensors','*.png','*.jpg','*fp16*']))
"
mkdir -p $G/models/invsr
[ -f $G/models/invsr/noise_predictor_sd_turbo_v5.pth ] || curl -sL -o $G/models/invsr/noise_predictor_sd_turbo_v5.pth https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth
# SwinIR / Real-ESRGAN official release weights
curl -sL -o $G/models/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
curl -sL -o $G/models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ls -la $G/models $G/models/SEEDVR2 $G/models/invsr
md5sum $G/models/*.pth $G/models/SEEDVR2/* $G/models/invsr/*.pth > $G/logs/weights_md5.txt
echo DOWNLOADS_DONE

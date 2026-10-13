"""make_meta.py -- provenance of the genai_sr run: model ids / versions / licences / seeds, GPU-hours, API spend.

    <venv2>/bin/python make_meta.py   -> results/run_meta.json
"""
import json
import os

import pandas as pd

import gsr_common as G

LOG = os.path.join(G.GSR, "logs")
SRV = os.path.join(LOG, "server")


def j(p):
    return json.load(open(os.path.join(SRV, p))) if os.path.exists(os.path.join(SRV, p)) else None


def main():
    gu = pd.read_csv(os.path.join(SRV, "gpu_usage.csv"))
    main_steps = {"simple", "flux2klein", "seedvr2", "invsr"}
    seed_steps = {"seedvar_flux", "seedvar_invsr", "seedvar_seedvr2"}
    gpu_h = dict(total=round(gu.seconds.sum() / 3600, 3),
                 restoration_preregistered=round(gu[gu.step.isin(main_steps)].seconds.sum() / 3600, 3),
                 scoring_cnn_iq=round(gu[gu.step.isin({"cnn", "iq"})].seconds.sum() / 3600, 3),
                 seed_sensitivity_exploratory=round(gu[gu.step.isin(seed_steps)].seconds.sum() / 3600, 3),
                 hardware="1x NVIDIA RTX 4090 24 GB per step (gpu-server, idle GPUs only: 0 and 3)",
                 per_step=gu.to_dict(orient="records"))
    led = pd.read_csv(os.path.join(LOG, "openrouter_costs.csv"))
    spend = dict(total_usd=round(float(led.cost_usd.sum()), 4), cap_usd=6.0,
                 by_purpose=led.groupby("purpose").cost_usd.agg(["count", "sum"]).round(4).reset_index().to_dict(orient="records"),
                 nbpro_content_filter_refusals=int(((led.purpose == "nbpro") & (led.status == "fail")).sum()),
                 note="own ledger; the paper's global VLM ledger (vlm/logs/openrouter_costs.csv) was not modified")
    simple = j("restore_simple_swinir_realesrgan.json") or {}
    sv = j("restore_seedvr2.json") or {}
    fl = j("restore_flux2klein.json") or {}
    models = [
        dict(key="bicubic", family="interpolation (control)", impl="cv2.resize INTER_CUBIC (OpenCV 4.11.0)", licence="Apache-2.0"),
        dict(key="swinir", family="regression SR (non-generative)", model="SwinIR-M classical x4 (DF2K, L1)",
             weights=simple.get("swinir", {}).get("url"), weights_md5=simple.get("swinir", {}).get("weights_md5"),
             loader="spandrel 0.4.2", licence="Apache-2.0", paper="Liang et al., ICCVW 2021"),
        dict(key="realesrgan", family="GAN blind SR", model="Real-ESRGAN x4plus (RRDBNet)",
             weights=simple.get("realesrgan", {}).get("url"), weights_md5=simple.get("realesrgan", {}).get("weights_md5"),
             loader="spandrel 0.4.2", licence="BSD-3-Clause", paper="Wang et al., ICCVW 2021"),
        dict(key="seedvr2", family="one-step diffusion transformer (restoration)", model="SeedVR2-3B fp16",
             weights_repo=sv.get("weights_repo"), dit_md5=sv.get("dit_md5"), vae_md5=sv.get("vae_md5"),
             runner=f"numz/ComfyUI-SeedVR2_VideoUpscaler @ {sv.get('repo_commit')} (standalone CLI, in-process)",
             settings=dict(seed=G.SEED, batch_size=1, color_correction="lab (default)", resolution="reference side S",
                           vae_tiling="encode+decode 1024/128"),
             licence="Apache-2.0", paper="Wang et al., SeedVR2 (ByteDance Seed, 2025)", s_per_img=sv.get("s_per_img")),
        dict(key="invsr", family="diffusion inversion SR (SD-Turbo)", model="InvSR noise_predictor_sd_turbo_v5, 1 step",
             base="stabilityai/sd-turbo @ b261bac6fd2cf515557d5d0707481eafa0485ec2",
             repo="zsyOAOA/InvSR @ " + open(os.path.join(SRV, "invsr_commit.txt")).read().strip(),
             weights_md5=open(os.path.join(SRV, "invsr_weights_md5.txt")).read().split()[0],
             settings=dict(seed=G.SEED, num_steps=1, timesteps=[200], chopping_size=128, color_fix="none (default)"),
             licence="NTU S-Lab License 1.0 (non-commercial); SD-Turbo: Stability AI non-commercial research licence",
             paper="Yue et al., CVPR 2025"),
        dict(key="flux2klein", family="rectified-flow instruction editor (open 'AI enhance')", model=fl.get("repo_id"),
             revision=fl.get("revision"), settings=dict(prompt=G.PROMPT_ENHANCE, steps=4, guidance_scale=1.0, seed=G.SEED,
                                                       dtype="bfloat16", input="LR bicubic-upsampled to 1024^2", output="1024^2"),
             diffusers=fl.get("diffusers"), licence="Apache-2.0", peak_mem_gb=fl.get("peak_mem_gb"),
             s_per_img=fl.get("s_per_img")),
        dict(key="nbpro", family="commercial AI image editor", model="google/gemini-3-pro-image (Nano Banana Pro) via OpenRouter",
             served="Google (OpenRouter provider field)", settings=dict(prompt=G.PROMPT_ENHANCE, seed=G.SEED,
                                                                         image_config={"aspect_ratio": "1:1", "image_size": "2K"}),
             date="2026-10-07", licence="proprietary API (Google / OpenRouter terms)", n_ok=35, n_refused=4),
        dict(key="gpt-6.1-sol", family="VLM counter", model="openai/gpt-6.1-sol via OpenRouter, prompt P1 v1, reasoning 'minimal', T=0",
             date="2026-10-07"),
    ]
    meta = dict(protocol_md5=open(os.path.join(LOG, "prereg_md5.txt")).read().splitlines(), seed=G.SEED,
                reference_sets=dict(R_HR=G.HR_IDS, R_LEG=G.LEG_NAMES, t_c=G.T_C), conditions=G.CONDS,
                models=models, gpu_hours=gpu_h, openrouter=spend,
                counters=dict(score_counts=j("score_counts_env.json"), cnn=j("score_cnn_env.json"), iq=j("iq_env.json")),
                storage="full-resolution restored PNGs (16 GB) stay on gpu-server ~/pickglass/genai_sr/outputs "
                        "(md5 in results/outputs_manifest.csv); 768-px JPEG viewing copies in genai_sr/viewing/")
    G.write_json(os.path.join(G.GSR, "results", "run_meta.json"), meta)
    print(json.dumps(dict(gpu_hours={k: v for k, v in gpu_h.items() if k != "per_step"}, spend=spend), indent=1))


if __name__ == "__main__":
    main()

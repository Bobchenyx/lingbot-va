# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NU-VA-V0 is a distilled video-action world model for robot control, built on top of [LingBot-VA](https://arxiv.org/abs/2601.21998). It uses a distilled checkpoint with 2-step video / 4-step action inference for fast evaluation on RoboTwin-2.0.

## Key Commands

### Installation
```bash
# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# For CUDA 12.6:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```

### Code Formatting
```bash
make format   # runs: isort wan_va && yapf -i -r *.py wan_va
```

### Inference Server
```bash
# RoboTwin evaluation server (single GPU):
bash evaluation/robotwin/launch_server.sh
# Multi-GPU:
bash evaluation/robotwin/launch_server_multigpus.sh
# Image-to-Video-Action:
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

### Evaluation Client
```bash
task_name="adjust_bottle"
bash evaluation/robotwin/launch_client.sh ./results ${task_name}
# Multi-GPU (7 groups for 8 GPUs, task_group_id 0-6):
bash evaluation/robotwin/launch_client_multigpus.sh ./results ${task_group_id}
```

## Architecture

### Core Package: `wan_va/`

- **`wan_va_server.py`** — Inference server with async WebSocket execution. Entry point via `python -m wan_va.wan_va_server`.
- **`configs/`** — EasyDict-based hierarchical configs exported as `VA_CONFIGS` dict. Config names: `robotwin`, `robotwin_i2av`.
- **`modules/model.py`** — `WanTransformer3DModel` with 3D causal attention. Supports `"torch"` (SDPA) and `"flashattn"` (flash-attn) attention modes for inference.
- **`modules/utils.py`** — Model loading (`load_vae`, `load_text_encoder`, `load_tokenizer`, `load_transformer`) and `WanVAEStreamingWrapper`.
- **`distributed/`** — FSDP sharding (`shard_model`) for distributed inference, distributed init utilities.
- **`utils/`** — FlowMatchScheduler, logging, server utilities, and WebSocket policy deployment (`Simple_Remote_Infer/`).

### Evaluation: `evaluation/robotwin/`

Client-server evaluation pipeline for RoboTwin 2.0 benchmark. The client connects to the inference server via WebSocket, sends observations, and receives action predictions.

## Critical: attn_mode Configuration

The `attn_mode` field in `<model-path>/transformer/config.json` must be `"torch"` or `"flashattn"` for inference. The `"flex"` mode is training-only and will crash at inference time.

## Distilled Model

The current config (`va_robotwin_cfg.py`) is set up for the distilled checkpoint with:
- `num_inference_steps = 2` (video)
- `action_num_inference_steps = 4` (action)
- Model path: set `wan22_pretrained_model_name_or_path` in `va_robotwin_cfg.py`

## GPU Memory

- RoboTwin evaluation (single GPU with offload): ~24GB VRAM
- Image-to-Video-Action (single GPU with offload): ~18GB VRAM
- Requirements: Python 3.10, PyTorch >= 2.6.0, CUDA >= 12.4

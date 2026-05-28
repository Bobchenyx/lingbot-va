# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VA is an autoregressive (AR) diffusion framework for simultaneous **video world modeling** and **robot action inference**. A single interleaved sequence carries both visual-dynamics latents and action tokens, processed by a dual-stream Mixture-of-Transformers (MoT) built on the Wan2.2 video backbone. Targets are robotic manipulation benchmarks (RoboTwin-2.0, LIBERO) and real-world deployment.

## Critical: `attn_mode` must match the task

The transformer's attention backend is read from the **model checkpoint's `transformer/config.json`** (loaded via `from_pretrained`), *not* from this repo's configs. You must manually edit that JSON file before launching:

- **Training** → `"flex"` (FlexAttention; will NOT work for inference)
- **Inference / Evaluation** → `"torch"` or `"flashattn"` (`"flex"` errors at eval time)

Note `wan_va_server.py` forces `attn_mode="torch"` when calling `load_transformer`, but other paths rely on the checkpoint JSON. See `wan_va/modules/model.py:299` for the dispatch.

## Common Commands

There is no test suite. Formatting is the only tooling target.

```bash
# Format code (note: Makefile calls yapf, but pyproject configures black/isort)
make format            # isort + yapf over wan_va

# Install (base deps in requirements.txt / pyproject; flash-attn needs --no-build-isolation)
pip install flash-attn --no-build-isolation
```

### Inference server / client (Server–Client architecture)

The model runs as a server; the simulator/robot runs as a client, isolating dependencies. **Server and client must be on the same machine** for the sim evaluations.

```bash
# RoboTwin-2.0
bash evaluation/robotwin/launch_server.sh                       # single GPU
bash evaluation/robotwin/launch_server_multigpus.sh             # multi-GPU
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
bash evaluation/robotwin/launch_client_multigpus.sh ${save_root} ${task_group_id}   # task_group_id 0-6

# LIBERO
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh

# Image-to-Video-Action generation (standalone, no client)
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

### Training (post-training / fine-tuning)

```bash
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
NGPU=8 CONFIG_NAME='libero_train'   bash script/run_va_posttrain.sh
```

Uses FSDP via `torch.distributed.run`. Larger global batch (32/64) is recommended; raise `gradient_accumulation_steps` when GPU-limited. WandB env vars (`WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_TEAM_NAME`, `WANDB_PROJECT`) are set in `script/run_va_posttrain.sh` and must be filled in.

## Architecture

### Entry points
- **`wan_va/wan_va_server.py`** — inference. `VA_Server` loads VAE, tokenizer, text encoder, and transformer. `infer_mode` (from config) selects `server` (async websocket policy server via `run_async_server_mode`) or `i2va` (one-shot image→video-action generation via `model.generate()`).
- **`wan_va/train.py`** — `Trainer` class; FSDP training loop with flow-matching loss.

Both dispatch on `--config-name`, resolved through `VA_CONFIGS` in `wan_va/configs/__init__.py`.

### Config system (`wan_va/configs/`)
Plain `EasyDict` objects, not Hydra/YAML despite the `--config-name` flag. Layered by `.update()`:
- `shared_config.py` (`va_shared_cfg`) → base defaults (dtype, patch_size, host/port, `infer_mode`).
- Per-domain inference cfg, e.g. `va_robotwin_cfg.py` → model path, action dims, cameras, guidance scales, SNR shifts, action normalization stats.
- `_train_cfg.py` adds dataset path + optimizer/schedule; `_i2va.py` adds image path + prompt and sets `infer_mode='i2va'`.

All registered keys in `VA_CONFIGS`: `robotwin`, `franka`, `libero`, `demo` and their `*_train` / `*_i2av` variants. **When editing model behavior, check whether the value lives here or in the checkpoint's `transformer/config.json`.** Domain-critical keys to keep in sync with a checkpoint: `action_snr_shift`, `used_action_channel_ids`, `norm_stat`.

### Model (`wan_va/modules/model.py`)
`WanTransformer3DModel` (diffusers `ModelMixin`/`ConfigMixin`). MoT design: video-latent and action streams have **separate** parameters (`condition_embedder_action`, `action_norm*`, `action_embedder`/`action_proj_out`, scale-shift tables) but share transformer blocks. `_build_seq_ids` interleaves latent and action tokens with frame/modality ids; `forward_train` vs. the async/KV-cache inference path drive the two execution modes. `FlexAttnFunc` (compiled FlexAttention) implements the block-sparse causal attention windows (`attn_window`); `WanAttention` switches kernel by `attn_mode`.

### Distributed (`wan_va/distributed/`)
`fsdp.py` (`shard_model`, `apply_ac` activation checkpointing) and `util.py` (`_configure_model`, `init_distributed`, mesh helpers). Used by both train and server.

### Data (`wan_va/dataset/lerobot_latent_dataset.py`)
`MultiLatentLeRobotDataset` consumes **LeRobot-format** datasets that have been pre-processed into VAE latents. Custom data pipeline (see README "Custom Dataset Preparation"):
1. Raw → LeRobot format.
2. Add `action_config` (frame-segmented action text) to `meta/episodes.jsonl`.
3. Extract Wan2.2 VAE latents → `latents/.../episode_{idx}_{start}_{end}.pth` mirroring `videos/`. Each `.pth` holds `latent`, `text_emb`, frame/dim metadata, fps.
- **Action format: 30 dims** — L/R EEF (7+7), L/R joints (7+7), L/R gripper (1+1); missing dims padded with 0. `used_action_channel_ids` selects the active subset per domain.

### Remote inference plumbing
Websocket policy server/client lives in `wan_va/utils/Simple_Remote_Infer/deploy/` (msgpack-numpy serialization). Evaluation clients (`evaluation/robotwin/`, `evaluation/libero/`) connect to the running server.

## Reference material
- `example/{robotwin,libero,franka,demo}/` — sample inputs for i2va generation.
- `LingBot_VA_paper.pdf`, README.md "News" section — model variants and dataset specifics.

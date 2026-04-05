# NU-VA-V0

Distilled video-action world model for robot control, built on top of [LingBot-VA](https://arxiv.org/abs/2601.21998). Uses a **2-step video / 4-step action** distilled checkpoint for fast inference and evaluation.

## Installation

**Requirements:** Python 3.10, PyTorch >= 2.6.0, CUDA >= 12.4

```bash
# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# For CUDA 12.6:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```

## `attn_mode` Configuration

The `attn_mode` field in `<model-path>/transformer/config.json` must be set to `"torch"` or `"flashattn"` for inference. The `"flex"` mode is training-only and will cause errors at inference time.

## Evaluation on RoboTwin-2.0

### Preparing the RoboTwin Environment

Follow the official instructions: [RoboTwin-2.0 Installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

In summary:

1. Install Vulkan:
```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

2. Clone RoboTwin:
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin && git checkout 2eeec322
```

3. Modify `script/requirements.txt`:
```
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
```

4. Modify line 8 of `script/_install.sh`:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

5. Run installation and download assets:
```bash
bash script/_install.sh
bash script/_download_assets.sh
```

### Deploying the Inference Server

```bash
# single GPU
bash evaluation/robotwin/launch_server.sh

# multi-GPU
bash evaluation/robotwin/launch_server_multigpus.sh
```

### Executing the Inference Client

```bash
# single GPU
task_name="adjust_bottle"
save_root="results/"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}

# multi-GPU (7 groups for 8 GPUs, task_group_id 0-6)
save_root="results/"
task_group_id=0
bash evaluation/robotwin/launch_client_multigpus.sh ${save_root} ${task_group_id}
```

Results are saved in `/path/to/your/RoboTwin/${save_root}`. The inference server and client must be on the same machine. For multi-GPU, 50 tasks are padded to 56 and partitioned into 7 groups. See `evaluation/robotwin/launch_client_multigpus.sh` for grouping details.

> **GPU Memory**: ~24GB VRAM for single-GPU RoboTwin evaluation with offload enabled.

### Image to Video-Action Generation

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

> **GPU Memory**: ~18GB VRAM for single-GPU i2av inference with offload enabled.

## License

Apache License 2.0. See [LICENSE](LICENSE.txt).

## Acknowledgments

Built on top of [LingBot-VA](https://github.com/Robbyant/lingbot-va), which uses:
- [Wan-Video](https://github.com/Wan-Video) - Vision transformer backbone
- [MoT](https://github.com/facebookresearch/Mixture-of-Transformers) - Mixture-of-Transformers architecture

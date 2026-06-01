#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/server_env.bash"

# The base Docker image can contain a mixed CUDA 12 pip stack where torch loads
# libcusparse.so.12 with the wrong libnvJitLink.so.12. Pin the known-good CUDA
# 11.8 PyTorch wheels before installing Whisper/TTS dependencies.
pip uninstall -y \
  torch torchvision torchaudio triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 \
  >/dev/null 2>&1 || true

pip install --force-reinstall --no-cache-dir \
  torch==2.2.0+cu118 \
  torchvision==0.17.0+cu118 \
  torchaudio==2.2.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir \
  openai \
  openai-whisper \
  websockets \
  opencv-python \
  soundfile \
  scipy==1.11.4 \
  numpy==1.22.0 \
  networkx==2.8.8 \
  webdataset \
  TTS

pip install --force-reinstall --no-cache-dir numpy==1.22.0 networkx==2.8.8

# ROS Noetic message generation expects Empy 3.x. The unrelated PyPI
# package named "em", or Empy 4.x, breaks genmsg with:
#   AttributeError: module 'em' has no attribute 'RAW_OPT'
pip uninstall -y em empy >/dev/null 2>&1 || true
pip install --no-cache-dir empy==3.3.4

python - <<'PY'
import em
if not hasattr(em, "RAW_OPT"):
    raise RuntimeError("ROS message generation requires empy==3.3.4 providing em.RAW_OPT")
print("empy", getattr(em, "__version__", "unknown"), "ok")
PY

python - <<'PY'
import torch
if "+cu118" not in torch.__version__:
    raise RuntimeError(f"Expected torch CUDA 11.8 wheel, got {torch.__version__}")
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
PY

cd "$SERVER_ROOT"
catkin_make --pkg dual_arm_msgs
source "$SERVER_ROOT/devel/setup.bash"
catkin_make

python -c "import numpy, torch, whisper; print('numpy', numpy.__version__); print('torch', torch.__version__, torch.cuda.is_available()); print('whisper ok')"

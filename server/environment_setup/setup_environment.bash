#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/server_env.bash"

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

cd "$SERVER_ROOT"
catkin_make --pkg dual_arm_msgs
source "$SERVER_ROOT/devel/setup.bash"
catkin_make

python -c "import numpy, torch, whisper; print('numpy', numpy.__version__); print('torch', torch.__version__, torch.cuda.is_available()); print('whisper ok')"

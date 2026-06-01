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

cd "$SERVER_ROOT"
catkin_make --pkg dual_arm_msgs
source "$SERVER_ROOT/devel/setup.bash"
catkin_make

python -c "import numpy, torch, whisper; print('numpy', numpy.__version__); print('torch', torch.__version__, torch.cuda.is_available()); print('whisper ok')"

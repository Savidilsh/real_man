#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SERVER_TS_IP:=100.104.233.12}"
: "${CONDA_ENV_NAME:=rm}"

export ROS_MASTER_URI="http://${SERVER_TS_IP}:11311"
export ROS_IP="$SERVER_TS_IP"
export PYTHONPATH="$SERVER_ROOT:${PYTHONPATH:-}"

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
elif command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV_NAME"
fi

if [ -f "$SERVER_ROOT/devel/setup.bash" ]; then
  source "$SERVER_ROOT/devel/setup.bash"
fi

cd "$SERVER_ROOT"

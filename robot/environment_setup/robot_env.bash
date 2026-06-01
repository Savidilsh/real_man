#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SERVER_TS_IP:=100.104.233.12}"
: "${ROBOT_IP:=100.76.54.86}"

export ROS_MASTER_URI="http://${SERVER_TS_IP}:11311"
export ROS_IP="$ROBOT_IP"
export PYTHONPATH="$ROBOT_ROOT:${PYTHONPATH:-}"

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi

if [ -f "$ROBOT_ROOT/devel/setup.bash" ]; then
  source "$ROBOT_ROOT/devel/setup.bash"
fi

cd "$ROBOT_ROOT"

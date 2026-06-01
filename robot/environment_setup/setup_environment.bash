#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/robot_env.bash"

cd "$ROBOT_ROOT"
catkin_make --pkg dual_arm_msgs
source "$ROBOT_ROOT/devel/setup.bash"
catkin_make

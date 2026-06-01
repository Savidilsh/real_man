#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SERVER_TMUX_SESSION:-ros_server}"
ENV_CMD="source '$SERVER_ROOT/environment_setup/server_env.bash'"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attaching to existing session..."
  tmux attach -t "$SESSION"
  exit 0
fi

echo "Starting server-side ROS nodes..."

tmux new-session -d -s "$SESSION" -n main
PANE_0="$(tmux list-panes -t "$SESSION" -F '#{pane_id}' | head -n 1)"
tmux send-keys -t "$PANE_0" "$ENV_CMD; roscore" C-m

PANE_1="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_0")"
tmux send-keys -t "$PANE_1" "$ENV_CMD; sleep 5; roslaunch '$SERVER_ROOT/src/dual_arm_control/arm_control/launch/arm_control.launch'" C-m

PANE_2="$(tmux split-window -h -P -F '#{pane_id}' -t "$PANE_1")"
tmux send-keys -t "$PANE_2" "$ENV_CMD; sleep 5; roslaunch '$SERVER_ROOT/src/dual_arm_control/dual_arm_moveit_config/dual_75B_arm_moveit_config/launch/demo_realrobot.launch'" C-m

tmux select-layout -t "$SESSION" tiled

echo "Attaching to tmux session '$SESSION'..."
tmux attach -t "$SESSION"

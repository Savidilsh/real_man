#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${ROBOT_TMUX_SESSION:-ros_robot}"
ENV_CMD="source '$ROBOT_ROOT/environment_setup/robot_env.bash'"
SERVER_TS_IP="${SERVER_TS_IP:-100.104.233.12}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attaching to existing session..."
  tmux attach -t "$SESSION"
  exit 0
fi

echo "Starting robot-side ROS nodes..."

tmux new-session -d -s "$SESSION" -n main
PANE_0="$(tmux list-panes -t "$SESSION" -F '#{pane_id}' | head -n 1)"
tmux send-keys -t "$PANE_0" "$ENV_CMD; sleep 5; roslaunch '$ROBOT_ROOT/src/d435_control/rgb_depth_aligned.launch'" C-m

PANE_1="$(tmux split-window -h -P -F '#{pane_id}' -t "$PANE_0")"
tmux send-keys -t "$PANE_1" "$ENV_CMD; sleep 5; roslaunch '$ROBOT_ROOT/src/agv_control/agv_ros/launch/agv_start.launch'" C-m

PANE_2="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_0")"
tmux send-keys -t "$PANE_2" "$ENV_CMD; sleep 5; roslaunch '$ROBOT_ROOT/src/servo_control/servo_ros/launch/servo_start.launch'" C-m

PANE_3="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_1")"
tmux send-keys -t "$PANE_3" "$ENV_CMD; sleep 5; roslaunch '$ROBOT_ROOT/src/dual_arm_control/arm_driver/launch/dual_arm_75_driver.launch'" C-m

PANE_4="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_2")"
tmux send-keys -t "$PANE_4" "$ENV_CMD; sleep 10; python3 '$ROBOT_ROOT/audio_player_node.py'" C-m

PANE_5="$(tmux split-window -h -P -F '#{pane_id}' -t "$PANE_4")"
tmux send-keys -t "$PANE_5" "$ENV_CMD; sleep 10; python3 '$ROBOT_ROOT/wifi_watchdog.py' --verbose --ping-target '$SERVER_TS_IP'" C-m

tmux select-layout -t "$SESSION" tiled

echo "Attaching to tmux session '$SESSION'..."
tmux attach -t "$SESSION"

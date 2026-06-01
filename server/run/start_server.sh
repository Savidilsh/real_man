#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-realman_server}"
IMAGE_NAME="${IMAGE_NAME:-sayplan:graspnet-ready}"
SERVER_TS_IP="${SERVER_TS_IP:-100.104.233.12}"
ROBOT_IP="${ROBOT_IP:-100.76.54.86}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rm}"
WORKDIR="/root/ws_demo/catkin_ws"
SESSION="${SERVER_TMUX_SESSION:-ros_server}"

echo "================================================"
echo "  Starting Agent System - Server Side"
echo "================================================"
echo ""

if ! docker info >/dev/null 2>&1; then
    echo " Error: Docker is not running"
    echo "   Please start Docker and try again"
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo " Error: Docker image '$IMAGE_NAME' not found"
    echo "   Available related images:"
    docker images | grep -E "sayplan|demo_0|graspnet" || true
    exit 1
fi

echo "Docker image: $IMAGE_NAME"
echo "Container: $CONTAINER_NAME"
echo "Server root: $SERVER_ROOT"
echo ""

echo "[1/4] Starting Docker container..."
if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null || true
fi

existing_container="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"
if [ -n "$existing_container" ]; then
    container_image="$(docker inspect "$CONTAINER_NAME" --format '{{.Config.Image}}')"
    if [ "$container_image" != "$IMAGE_NAME" ]; then
        echo " Error: Container '$CONTAINER_NAME' exists but uses image '$container_image'"
        echo "   This launcher expects '$IMAGE_NAME'."
        echo ""
        echo "   Recreate it with:"
        echo "     docker stop $CONTAINER_NAME"
        echo "     docker rm $CONTAINER_NAME"
        echo "     bash server/run/start_server.sh"
        exit 1
    fi

    if ! docker ps -q -f "name=^/${CONTAINER_NAME}$" | grep -q .; then
        docker start "$CONTAINER_NAME" >/dev/null
    fi
else
    docker run -d --name "$CONTAINER_NAME" --gpus all --network host --shm-size=100g \
      -e SERVER_TS_IP="$SERVER_TS_IP" \
      -e ROBOT_IP="$ROBOT_IP" \
      -e CONDA_ENV_NAME="$CONDA_ENV_NAME" \
      -e ROS_IP="$SERVER_TS_IP" \
      -e ROS_MASTER_URI="http://${SERVER_TS_IP}:11311" \
      -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
      -e "ACCEPT_EULA=Y" \
      -e "PRIVACY_CONSENT=Y" \
      -e "DISPLAY=${DISPLAY:-}" \
      -e XDG_RUNTIME_DIR=/tmp/runtime-root \
      --entrypoint /bin/bash \
      -w "$WORKDIR" \
      -v /tmp/.X11-unix/:/tmp/.X11-unix \
      -v "$SERVER_ROOT:$WORKDIR" \
      -v "$SERVER_ROOT/environment_setup/container_bashrc:/root/.bashrc" \
      "$IMAGE_NAME" -lc "tail -f /dev/null" >/dev/null
fi

echo "      Container '$CONTAINER_NAME' is running"
echo ""

echo "[2/4] Setting up tmux session inside Docker..."
docker exec "$CONTAINER_NAME" bash -lc "
set -e

SESSION='$SESSION'
WORKDIR='$WORKDIR'

if tmux has-session -t \"\$SESSION\" 2>/dev/null; then
    echo \"tmux session '\$SESSION' already exists; reusing it.\"
    exit 0
fi

SETUP_CMD=\"source \${WORKDIR}/environment_setup/server_env.bash\"

tmux new-session -d -s \"\$SESSION\" -n roscore
PANE_0=\"\$(tmux list-panes -t \"\$SESSION\" -F '#{pane_id}' | head -n 1)\"
tmux send-keys -t \"\$PANE_0\" \"\$SETUP_CMD; roscore\" C-m

PANE_1=\"\$(tmux split-window -v -P -F '#{pane_id}' -t \"\$PANE_0\")\"
tmux send-keys -t \"\$PANE_1\" \"\$SETUP_CMD; sleep 5; roslaunch src/dual_arm_control/arm_control/launch/arm_control.launch\" C-m

PANE_2=\"\$(tmux split-window -h -P -F '#{pane_id}' -t \"\$PANE_0\")\"
tmux send-keys -t \"\$PANE_2\" \"\$SETUP_CMD; sleep 8; roslaunch src/dual_arm_control/dual_arm_moveit_config/dual_75B_arm_moveit_config/launch/demo_realrobot.launch\" C-m

PANE_3=\"\$(tmux split-window -h -P -F '#{pane_id}' -t \"\$PANE_1\")\"
tmux send-keys -t \"\$PANE_3\" \"\$SETUP_CMD; sleep 3; python sayplan/savindu/agent_cv.py\" C-m

tmux select-layout -t \"\$SESSION:0\" tiled

tmux new-window -t \"\$SESSION:1\" -n audio_stt
tmux send-keys -t \"\$SESSION:1\" \"\$SETUP_CMD; python sayplan/savindu/audio_to_str.py\" C-m
"

echo "      tmux windows created"
echo ""

echo "[3/4] Checking status..."
sleep 2
if docker exec "$CONTAINER_NAME" tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "      tmux session '$SESSION' is running"
else
    echo "      tmux session failed to start"
    exit 1
fi

echo ""
echo "[4/4] Server setup complete"
echo ""
echo "Attach with:"
echo "  docker exec -it $CONTAINER_NAME tmux attach -t $SESSION"
echo ""
echo "Robot side still needs to run on the robot:"
echo "  cd <real_man_repo>/robot"
echo "  bash run/ros_robot.bash"
echo ""
echo "Server ROS IP: $SERVER_TS_IP"
echo "Robot ROS IP:  $ROBOT_IP"
echo ""

read -p "Attach to tmux session now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker exec -it "$CONTAINER_NAME" tmux attach -t "$SESSION"
fi

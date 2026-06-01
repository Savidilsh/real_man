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

if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null || true
fi

existing_container="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"

if [ -n "$existing_container" ]; then
  container_image="$(docker inspect "$CONTAINER_NAME" --format '{{.Config.Image}}')"

  if [ "$container_image" != "$IMAGE_NAME" ]; then
    echo "Container '$CONTAINER_NAME' already exists, but it uses image '$container_image'."
    echo "This script is configured for '$IMAGE_NAME'."
    echo ""
    echo "To recreate it with the correct image, run:"
    echo "  docker stop $CONTAINER_NAME"
    echo "  docker rm $CONTAINER_NAME"
    echo "  bash server/environment_setup/docker_start.sh"
    exit 1
  fi

  if ! docker ps -q -f "name=^/${CONTAINER_NAME}$" | grep -q .; then
    docker start "$CONTAINER_NAME" >/dev/null
  fi

  docker exec -it "$CONTAINER_NAME" /bin/bash
  exit 0
fi

docker run --name "$CONTAINER_NAME" -it --gpus all --network host --shm-size=100g \
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
  "$IMAGE_NAME"

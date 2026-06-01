# How To Run

This repo has two sides:

- `server/`: run on the workstation/server with Docker.
- `robot/`: run on the robot machine.

Default IPs:

```bash
export SERVER_TS_IP=100.104.233.12
export ROBOT_IP=100.76.54.86
```

## 1. Server First-Time Setup

On the server/workstation:

```bash
git clone <your-real_man-repo-url>
cd real_man
export SERVER_TS_IP=100.104.233.12
export ROBOT_IP=100.76.54.86
export OPENAI_API_KEY="your_key_here"
```

Start and enter Docker:

```bash
bash server/environment_setup/docker_start.sh
```

Inside Docker, run the setup once:

```bash
bash environment_setup/setup_environment.bash
```

Then exit Docker:

```bash
exit
```

## 2. Robot First-Time Setup

On the robot machine:

```bash
git clone <your-real_man-repo-url>
cd real_man/robot
export SERVER_TS_IP=100.104.233.12
export ROBOT_IP=100.76.54.86
bash environment_setup/setup_environment.bash
```

## 3. Start Robot Side

On the robot machine:

```bash
cd real_man/robot
export SERVER_TS_IP=100.104.233.12
export ROBOT_IP=100.76.54.86
bash run/ros_robot.bash
```

This starts:

- D435 camera
- AGV bridge
- Servo controller
- Arm driver
- Audio player
- WiFi watchdog

## 4. Start Server Side

On the server/workstation:

```bash
cd real_man
export SERVER_TS_IP=100.104.233.12
export ROBOT_IP=100.76.54.86
export OPENAI_API_KEY="your_key_here"
bash server/run/start_server.sh
```

When it asks:

```text
Attach to tmux session now? (y/n):
```

Press:

```text
y
```

This starts:

- `roscore`
- `arm_control`
- MoveIt
- `agent_cv.py`
- `audio_to_str.py`

## Reattach To Running Sessions

Server:

```bash
docker exec -it realman_server tmux attach -t ros_server
```

Robot:

```bash
tmux attach -t ros_robot
```

## Stop

Inside tmux, close one pane process with:

```bash
Ctrl-c
```

Detach from tmux without stopping processes:

```text
Ctrl-b then d
```

Stop a whole tmux session:

```bash
tmux kill-session -t ros_server
tmux kill-session -t ros_robot
```

Stop the Docker container on the server:

```bash
docker stop realman_server
```

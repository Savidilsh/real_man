# real_man

Clean split deployment for the Realman voice agent.

The repo has two deployable folders:

- `server/`: workstation/Docker side. Runs ROS master, arm control, MoveIt, `agent_cv.py`, and `audio_to_str.py`.
- `robot/`: robot side. Runs camera, AGV bridge, servo controller, arm driver, audio player, and WiFi watchdog.

## Server

Enter the server Docker shell from the repo root:

```bash
bash server/environment_setup/docker_start.sh
```

Inside Docker, build/setup if needed:

```bash
bash environment_setup/setup_environment.bash
```

Start the full server side from the host:

```bash
bash server/run/start_server.sh
```

If you are already inside `server/`, use `bash run/start_server.sh`.

Attach later:

```bash
docker exec -it realman_server tmux attach -t ros_server
```

## Robot

Copy or clone this repo onto the robot, then run:

```bash
cd robot
bash environment_setup/setup_environment.bash
bash run/ros_robot.bash
```

Attach later:

```bash
tmux attach -t ros_robot
```

## IPs

Defaults are kept from the working setup:

```bash
SERVER_TS_IP=100.104.233.12
ROBOT_IP=100.76.54.86
```

Override them before running if needed:

```bash
export SERVER_TS_IP=<server-ip>
export ROBOT_IP=<robot-ip>
export OPENAI_API_KEY=<your-key>
```

## Notes

- Do not commit runtime JSON, generated audio, model caches, or real API keys.
- `server/` and `robot/` are separate catkin workspaces.
- The Docker image expected by default is `sayplan:graspnet-ready`.

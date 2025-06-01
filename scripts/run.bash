#!/bin/bash
IMAGE=$1
CONTAINER=$2
WS="${3:-"ws"}"

# The directory from which script has been run
SIM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Path to ${WS} volume folder: ${SIM_ROOT}"

# TODO: делать xhost -local:docker после удаления контейнера
xhost +local:docker && \
  docker run \
    -it \
    --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${SIM_ROOT}:/dbow3 \
    -v /dev:/dev \
    --ipc=host \
    --network=host \
    --gpus all \
    --device-cgroup-rule='c *:* rmw' \
    --name ${CONTAINER} $IMAGE
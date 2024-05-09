#!/bin/bash

if [[ ! $1 ]] ; then
    echo "Container name not passed" 
    exit 1
fi

if [[ $2 ]] ; then
    echo "Building container with GPU acceleration" 
fi

container_name=$1

xhost +local:docker

if [[ "$2" == "nvidia" ]] ; then
    # Build the docker image
    docker run -t -d --name $container_name -e DISPLAY=$DISPLAY -e LOCAL_USER_ID=1001  --gpus=all --runtime=nvidia -e "NVIDIA_DRIVER_CAPABILITIES=all" --network=host --pid=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:rw ros2-humble:latest
else
    # Build the docker image
    docker run -t -d --name $container_name -e DISPLAY=$DISPLAY -e LOCAL_USER_ID=1001  --network=host --pid=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:rw  osrf/ros:humble-desktop
fi
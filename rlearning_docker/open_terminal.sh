#!/bin/bash

if [[ ! $1 ]] ; then
    echo "Container name not passed" 
    exit 1
fi

container_name=$1

docker exec -it $container_name bash
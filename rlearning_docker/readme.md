## Install Docker
```sh
 sudo snap install docker
```
## Update permissions and check Docker
```sh
sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo chmod 666 /var/run/docker.sock
docker run hello-world
```

## Build image with docker file
```sh
docker build . -t <container_name>
```

## Create a container
```sh
sudo chmod +x ./create_container.sh
./create_container.sh {container_name}
```



## Start a docker container
```sh
docker run {container_name}
```
if that doesn't work:
```sh
docker run <container_name>
docker start {container_name}
```
## Open a terminal

```sh
sudo chmod +x ./open_terminal.sh
./open_terminal.sh {container_name}
```
## Check Ubuntu Version
```sh
lsb_release -a
```
## To start it every time
```sh
docker start {container_name}
./open_terminal.sh {container_name}
```

## Install Gazebo
```sh
sudo apt-get update
sudo apt install gazebo
```

## Install RViz
```sh
sudo apt-get update
sudo apt install ros-humble-rviz

```

## Install Dependencies
```sh
rosedp update
rosdep install -i --from-path src --rosdistro humble -y
```

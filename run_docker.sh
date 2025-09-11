
image_uri="nvcr.io/nvidia/pytorch:24.05-py3"

sudo docker run --shm-size=64g --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
 --network=host \
 -v /path/to/your/directory...:/workspace \
 -v /path/to/your/directory2...:/hdd \
 --name base_docker -it --rm ${image_uri}

docker run \
    -it --rm \
    --gpus all \
    --shm-size=32G \
    --publish 30022:30022 \
    --volume /home/jinwoo/:/workspace/ \
    --volume /old_home/datasets/Data/:/workspace/Data/ \
    jinwoo/research
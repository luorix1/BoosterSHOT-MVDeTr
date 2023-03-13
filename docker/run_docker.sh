docker run \
    -it --rm \
    --gpus all \
    --shm-size=32G \
    --publish 30027:30027 \
    --volume /home/jinwoo/:/workspace/ \
    --volume /old_home/Data/:/workspace/Data/ \
    jinwoo/research
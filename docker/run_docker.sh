docker run \
    -it --rm \
    --gpus all \
    --shm-size=32G \
    --publish 29022:29022 \
    --volume /home/jinwoo/:/workspace/ \
    --volume /data/:/workspace/Data/ \
    jinwoo/research
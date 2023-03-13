FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install pkg-config
RUN rm -rf /var/cache/apt/archives

RUN pip install pip --upgrade
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt
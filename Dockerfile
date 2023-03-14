FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install git python3 python3-pip ffmpeg libsm6 libxext6 -y
WORKDIR /usr/src/app

# Set up Python dependencies
COPY requirements.txt ./
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install --no-cache-dir -r requirements.txt

# Copy Yolov7_StrongSORT_OSNet to workdir
COPY Yolov7_StrongSORT_OSNet Yolov7_StrongSORT_OSNet

# Set up git
COPY .git .git
COPY .gitmodules .gitmodules 
RUN git submodule update --init --recursive

# Set up and run python3
COPY main.py main.py
COPY runner.py runner.py
COPY runner_utils.py runner_utils.py

CMD ["python3","-u","main.py","--c", "./config.yml"]

FROM python:3.9

WORKDIR /usr/src/app

COPY docker-requirements.txt ./
RUN pip install --no-cache-dir -r docker-requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

# Tech Stack / Dependencies

|                  | Name             | Version | Description                                       |
| :--------------- | :--------------- | :------ | ------------------------------------------------- |
| Language         | Python           | 3.10.0  | The programming language used for this project    |
| Framework        | Pytorch          | 1.13.1  | The deep learning framework used for this project |
| Computer Vision  | OpenCV           | 4.5.4   | The computer vision library used for this project |
| Object Detection | YOLOv7           | 1.0     | The object detection model used for this project  |
| Tracking & ReID  | StrongSORT_OSNet | 1.0     | The ReID model used for this project              |

## Repository Setup

```console
git clone git@github.com:JohnTrunix/moni.git
```

```console
cd moni
```

```console
git submodule update --init --recursive
```

### Check if all submodules and their submodules were set up correctly

Check submodules:

```console
git submodule status

>>> 852c9cf5c610a149331ce8e42d47be103cfd03ab Yolov7_StrongSORT_OSNet (v1.0-14-g852c9cf)
```

Check submodules of submodules:

```console
git submodule foreach git submodule status

>>> 4a0793780bd13f53ec2ca753a94dcef62dc9e955 strong_sort/deep/reid (v1.0.6-153-g4a07937)
>>> 3ab80fb707528cdc0aaad8e7cef39546a1ccc7f2 yolov7 (heads/main)
```

## Setup Environment

```console
virtualenv env --python=3.10
```

```console
env/Scripts/activate
```

### Install Pytorch / Torchvision

With CUDA support (follow the instructions on the [Pytorch website](https://pytorch.org/get-started/locally/))

eg. for Pytorch Stable (1.13.1), Windows, Pip, Python, CUDA 11.7:

```console
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

eg. without CUDA support:

```console
pip install torch>=1.7.0,!=1.12.0 torchvision>=0.8.1,!=0.13.0
```

### Install Requirements

```console
pip install -r requirements.txt
```

## Run

Before running the application, make sure:

-   to configure the `conf/config.yml` file correctly for your needs (see [example-config.yml](../conf/example-config.yml) for more information).
-   setup the [InfluxDB](https://www.influxdata.com/) database and configure the `conf/config.yml` file accordingly.
-   correctly set up your video source in the `conf/config.yml` file.
-   load the weights for the YOLOv7 and ReID models into the `weights` folder.

### Run locally

```console
python main.py
```

**Note:** If you want to run the application with a different configuration file, you can use the `--c` argument:

```console
python main.py --c conf/config.yml
```

**Note:** if you want to run the application with Docker, see the [Docker section](#docker) below.

## RTSP Stream

### Publish stream

Use ffmpeg to stream a local video feed to the rtsp server.

```sh
.\ffmpeg.exe -stream_loop -1 `
-re `
-i 'data\campus\campus4-c0.avi' `
-filter:v fps=5 `
-c:v libx264 `
-preset slow `
-crf 20 `
-c:a aac `
-b:a 160k `
-vf format=yuv420p `
-movflags +faststart `
-rtsp_transport tcp `
-f rtsp rtsp://<IP>:8554/input-stream-1
```

### Read stream

Use ffplay to read streams

```sh
.\ffplay.exe rtsp://<IP>:8554/input-stream-1
.\ffplay.exe rtsp://<IP>:8554/output-stream-1
```

## Docker

### Create config

Create a `conf\docker-config.yml` file. In this file you have to define all settings for the moni docker container to work correctly. At startup by docker-compose this file will be mounted into the container.

#### Start all containers

For local development you can start all containers with one command.
`docker compose --profile dev up --build --force-recreate -d`

Now you need to configure the InfluxDB (http://localhost:8086) and complete the initial setup (Create org, token, bucket, etc.). Put the InfluxDB config in the `docker-config.yml` and restart moni with `docker restart moni`.

#### Start moni container

If you have a seperate InfluxDB and RTSP server running, you can only start the moni container.
`docker compose --profile moni up --build --force-recreate -d`

### Container logs

You can use `docker logs -f moni` to check if the application is working correctly.

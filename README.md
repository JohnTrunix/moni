# Lilith-FlowTracker

Our project repository for the hackathon Thurgau challenge.

## Setup Repository

```console
git clone git@github.com:JohnTrunix/Lilith-FlowTracker.git
```

```console
cd Lilith-FlowTracker
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

For Pytorch Stable (1.13.1), Windows, Pip, Python, CUDA 11.7:

```console
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

Without CUDA support:

```console
pip install torch>=1.7.0!=1.12.0 torchvision>=0.8.1,!=0.13.0
```

### Install Requirements

```console
pip install -r requirements.txt
```

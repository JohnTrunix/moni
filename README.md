# Lilith-FlowTracker

Our project repository for the hackathon Thurgau challenge.

## Setup Repository

```console
$ git clone git@github.com:JohnTrunix/Lilith-FlowTracker.git
```

```console
$ cd Lilith-FlowTracker
```

```console
$ git submodule update --init --recursive
```

### Check if all submodules and their submodules were set up correctly

Check submodules:

```console
$ git submodule status

852c9cf5c610a149331ce8e42d47be103cfd03ab Yolov7_StrongSORT_OSNet (v1.0-14-g852c9cf)
```

Check submodules of submodules:

```console
$ git submodule foreach git submodule status

4a0793780bd13f53ec2ca753a94dcef62dc9e955 strong_sort/deep/reid (v1.0.6-153-g4a07937)
3ab80fb707528cdc0aaad8e7cef39546a1ccc7f2 yolov7 (heads/main)
```

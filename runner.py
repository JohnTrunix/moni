# autopep8: off
import os
import sys
from pathlib import Path
import yaml

import numpy as np
import random
import cv2
import torch
import torch.backends.cudnn as cudnn

from dotenv import load_dotenv
import subprocess

from runner_utils import plot_one_box, warpPoint
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'Yolov7_StrongSORT_OSNet'
WEIGHTS = FILE.parents[0] / 'weights'

#---------------------- Add Paths for custom modules ----------------------#

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'strong_sort' / 'deep' / 'reid') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort' / 'deep' / 'reid'))     # add reid ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from Yolov7_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT
from Yolov7_StrongSORT_OSNet.yolov7.models.experimental import attempt_load
from Yolov7_StrongSORT_OSNet.yolov7.utils.datasets import (LoadImages, LoadStreams)
from Yolov7_StrongSORT_OSNet.yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from Yolov7_StrongSORT_OSNet.yolov7.utils.torch_utils import (select_device, time_synchronized)

load_dotenv()
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'


@torch.no_grad()
def run_moni(
        stream_id,
        source,
        yaml_config,
        mp_barrier,  # multiprocessing barrier object for synchronization
        device, # Hardware device which is used for processing
        rtmp_url,
        t_matrix=None,  # transformation matrix for perspective transform
        imgsz=(640, 640),
        line_thickness=2,  # bounding boxes line thickness
        hide_labels=False,
        hide_conf=True,
        hide_class=False,
        half=False  # convert 32-bit tensor to 16-bit
):

    #---------------------- Load Configs from .yml file ----------------------#
    with open(yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # general flags
    save_influx = config['flags']['save_influx']
    rtmp_output = config['flags']['rtmp_output']
    show_video = config['flags']['show_video']
    show_matplot = config['flags']['show_matplot']
    detections_to_global = config['flags']['detections_to_global']
    save_video_local = config['flags']['save_video_local']
    save_coordinates_to_file = config['flags']['save_coordinates_to_file']

    # general configs
    classes = config['general']['classes']

    # yolo configs
    yolo_weights = config['yolo']['weights']
    conf_thres = config['yolo']['conf_thres']
    iou_thres = config['yolo']['iou_thres']

    # strong sort configs
    strong_sort_weights = config['strongsort']['weights']
    strong_sort_config = config['strongsort']['config']

    # InfluxDB config
    influx_config = config['influxdb']

    #---------------------- Initialize Objects ----------------------#
    if detections_to_global:
        if t_matrix is None:
            raise ValueError('t_matrix is None')
        else:
            t_matrix = np.array(t_matrix)
            if t_matrix.shape != (3, 3):
                raise ValueError('t_matrix is not 3x3')

    if save_influx:
        influx_writer = InfluxDBClient(
                            url = influx_config['url'],
                            token = influx_config['token'],
                            org = influx_config['org']).write_api(write_options=SYNCHRONOUS)

    if save_coordinates_to_file:
        if not os.path.exists('output'):
            os.makedirs('output')
        with open(f'output/coordinates_{stream_id}.csv', 'a') as f:
            f.write('frame,track_id,x,y\n')

    # get attributes from source before running
    if save_video_local or rtmp_output:
        frame_size = cv2.VideoCapture(source)
        frame_w = int(frame_size.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(frame_size.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(frame_size.get(cv2.CAP_PROP_FPS))
        frame_size.release()

    # initialize video writer
    if save_video_local:
        if not os.path.exists('output/streams'):
            os.makedirs('output/streams')
        vid_writer = cv2.VideoWriter(f'output/streams/{stream_id}.avi',
                                     cv2.VideoWriter_fourcc(*'XVID'), frame_fps, (frame_w, frame_h))

    # initialize rtmp stream writer if rtmp_output is True and rtmp_url is not None
    if rtmp_output:
        if rtmp_url is None:
            raise ValueError('rtmp_url is None')
        else:
            rtmp_process = subprocess.Popen(
                ['ffmpeg', 
                 '-y',
                 '-f', 'rawvideo',
                 '-vcodec', 'rawvideo',
                 '-pix_fmt', 'bgr24',
                 '-s', f'{frame_w}x{frame_h}',
                 '-i', '-',
                 '-an',
                 '-vcodec', 'mpeg4',
                 '-b:v', '5000k',
                 '-f', 'rtsp',
                 rtmp_url], stdin=subprocess.PIPE)

    #---------------------- Get File Type ----------------------#
    is_file = Path(source).suffix[1:] in VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_webcam = source.isnumeric() or (is_url and not is_file)

    #---------------------- Load Yolo Model ----------------------#
    device = select_device(str(device))
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)
    names = model.names
    stride = model.stride.max().cpu().numpy()
    imgsz = check_img_size(imgsz[0], s=stride)

    #---------------------- Load Strong Sort Model ----------------------#
    '''
    ToDo: Correctly load strong sort model and make detections shared between processes
    '''
    strong_sort = StrongSORT(
                        strong_sort_weights, 
                        device, 
                        half, 
                        max_dist=strong_sort_config['MAX_DIST'],
                        max_iou_distance=strong_sort_config['MAX_IOU_DISTANCE'],
                        max_age=strong_sort_config['MAX_AGE'],
                        n_init=strong_sort_config['N_INIT'],
                        nn_budget=strong_sort_config['NN_BUDGET'],
                        mc_lambda=strong_sort_config['MC_LAMBDA'],
                        ema_alpha=strong_sort_config['EMA_ALPHA'],
                    )

    outputs = [None]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    #---------------------- Load Video Stream ----------------------#
    if is_webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    #---------------------- RUN MONITORING LOOP FOR EACH FRAME ----------------------#
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None], [None]

    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_synchronized()
        dt[0] += t2 - t1

        #---------------------- YOLOv7 DETECTIONS ----------------------#
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        #---------------------- Non Max Suppression ----------------------#
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes)
        t4 = time_synchronized()
        dt[2] += t4 - t3

        #---------------------- Process detections ----------------------#
        for i, det in enumerate(pred):
            seen += 1
            if is_webcam:
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

                p = Path(p)

            else:
                p, im0, _ = path, im0s.copy(), getattr(
                    dataset, 'frame', 0)

                p = Path(p)

            curr_frames[i] = im0
            s += f'{im.shape[2]}x{im.shape[3]} '


            if strong_sort_config['ECC']:
                strong_sort.tracker.camera_update(
                    prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f'{n} {names[int(c)]}{"s" * (n > 1)}, '

                xywhs = xyxy2xywh(det[:, :4])
                confs = det[:, 4]
                clss = det[:, 5]

                t5 = time_synchronized()
                outputs[i] = strong_sort.update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t6 = time_synchronized()
                dt[3] += t6 - t5

                #---------------------- Draw detections ----------------------#
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0: 4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)
                        id = int(id)
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else
                                                          (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        plot_one_box(bboxes, im0, label=label, color=colors[int(
                            cls)], line_thickness=line_thickness)

                        #---------------------- Save to InfluxDB ----------------------#
                        if save_influx:
                            x_cord = (bboxes[0] + bboxes[2]) / 2
                            y_cord = bboxes[3]

                            #---------------------- Transform detections to global coordinates ----------------------#
                            if detections_to_global:
                                d_cord = np.array([x_cord, y_cord])
                                x_cord, y_cord = warpPoint(d_cord, t_matrix)

                            point = Point(influx_config['field']).tag('stream', stream_id
                                                                ).tag('id', id
                                                                ).field('x', x_cord
                                                                ).field('y', y_cord
                                                                ).field('frame', frame_idx)

                            influx_writer.write(bucket=influx_config['bucket'], org=influx_config['org'], record=point)

                        if save_coordinates_to_file:
                            x_cord = (bboxes[0] + bboxes[2]) / 2
                            y_cord = bboxes[3]

                            if detections_to_global:
                                d_cord = np.array([x_cord, y_cord])
                                x_cord, y_cord = warpPoint(d_cord, t_matrix)

                            with open(f'output/coordinates_{stream_id}.csv', 'a') as f:
                                f.write(f'{frame_idx},{id},{x_cord},{y_cord}\n')

                print(
                    f'{s}Done. YOLOv7:({t3 - t2:.3f}s) NMS:({t4 - t3:.3f}s) SORT:({t6 - t5:.3f}s)')

            else:
                strong_sort.increment_ages()
                print('No detections')

            if show_video:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_video_local:
                frame = cv2.resize(im0, (frame_w, frame_h), cv2.INTER_LANCZOS4)
                vid_writer.write(frame)

            if rtmp_output:
                fb = im0.tostring()
                rtmp_process.stdin.write(fb)

            if show_matplot:
                '''
                Not Implemented yet: Plotting of detections in images and global coordinates
                '''
                raise NotImplementedError('Matplotlib live visualisations not implemented yet')

        #---------------------- Update previous frames ----------------------#
            prev_frames[i] = curr_frames[i]

        if mp_barrier is not None:
            mp_barrier.wait()

    if rtmp_output:
        rtmp_process.stdin.close()
        rtmp_process.wait()

    if save_influx:
        influx_writer.close()

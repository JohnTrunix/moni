import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import subprocess
from numpy import random

load_dotenv()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'Yolov7_StrongSORT_OSNet'
WEIGHTS = FILE.parents[0] / 'weights'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

# autopep8: off

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'strong_sort'/ 'deep' / 'reid') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort' / 'deep' / 'reid'))  # add reid ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from Yolov7_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT
from Yolov7_StrongSORT_OSNet.strong_sort.utils.parser import get_config
from Yolov7_StrongSORT_OSNet.yolov7.models.experimental import attempt_load
from Yolov7_StrongSORT_OSNet.yolov7.utils.datasets import (LoadImages,
                                                           LoadStreams)
from Yolov7_StrongSORT_OSNet.yolov7.utils.general import (check_file,
                                                          check_img_size,
                                                          check_imshow,
                                                          check_requirements,
                                                          colorstr, cv2,
                                                          increment_path,
                                                          non_max_suppression,
                                                          scale_coords,
                                                          strip_optimizer,
                                                          xyxy2xywh)
from Yolov7_StrongSORT_OSNet.yolov7.utils.torch_utils import (
    select_device, time_synchronized)

# autopep8: on


class StreamTracking:
    def __init__(self,
                 stream_id,
                 source,
                 mp_barrier,
                 yolo_weights,  # model.pt path(s),
                 strong_sort_weights,
                 rtmp_enable=False,
                 rtmp_url=None,  # model.pt path,
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 show_vid=True,  # show results
                 show_matplot=False,
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=False,  # save cropped prediction boxes
                 save_vid=False,  # save confidences in --save-txt labels
                 nosave=False,  # do not save images/videos
                 classes=[0],  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project=ROOT / 'runs/track',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=2,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=True,  # hide confidences
                 hide_class=False,  # hide IDs
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 ):

        self.stream_id = stream_id
        self.source = str(source)
        self.rtmp_enable = rtmp_enable
        self.rtmp_url = rtmp_url
        self.mp_barrier = mp_barrier
        self.yolo_weights = yolo_weights
        self.strong_sort_weights = strong_sort_weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = str(device)
        self.show_vid = show_vid
        self.show_matplot = show_matplot
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.save_vid = save_vid
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.hide_class = hide_class
        self.half = half
        self.dnn = dnn

        # Setup Influx DB Client
        self.url = os.getenv('INFLUX_URL')
        self.token = os.getenv('INFLUX_TOKEN')
        self.org = os.getenv('INFLUX_ORG')
        self.bucket = os.getenv('INFLUX_BUCKET')
        self.client = InfluxDBClient(
            url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

        # Setup ffmpeg process
        if self.rtmp_url is not None and self.rtmp_enable:
            self.rtmp_process = subprocess.Popen([
                'ffmpeg',
                '-y',
                '-f',
                'rawvideo',
                '-pix_fmt',
                'bgr24',
                '-s',
                '360x288',
                '-i',
                '-',
                '-f',
                'flv',
                self.rtmp_url], stdin=subprocess.PIPE)

        # Organise Data
        self.save_img = not self.nosave and not self.source.endswith('.txt')
        is_file = Path(self.source).suffix[1:] in (VID_FORMATS)
        is_url = self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith(
            '.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)

        # Manage Directories
        if not isinstance(self.yolo_weights, list):
            exp_name = self.yolo_weights.stem
        elif type(self.yolo_weights) is list and len(self.yolo_weights) == 1:
            exp_name = Path(self.yolo_weights[0]).stem
        else:
            exp_name = 'ensemble'

        exp_name = self.name if self.name else exp_name + \
            '_' + self.strong_sort_weights.stem
        self.save_dir = increment_path(
            Path(self.project) / exp_name, exist_ok=self.exist_ok)
        self.save_dir = Path(self.save_dir)
        (self.save_dir / 'tracks' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Load Model
        self.device = select_device(self.device)
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = attempt_load(
            Path(self.yolo_weights), map_location=self.device)
        self.names, = self.model.names,
        self.stride = self.model.stride.max().cpu().numpy()
        self.imgsz = check_img_size(self.imgsz[0], s=self.stride)

        # Data Loaders
        if self.webcam:
            self.show_vid = check_imshow()
            cudnn.benchmark = True
            self.dataset = LoadStreams(
                self.source, img_size=self.imgsz, stride=self.stride)
            self.nr_sources = 1
        else:
            self.dataset = LoadImages(
                self.source, img_size=self.imgsz, stride=self.stride)
            self.nr_sources = 1

        self.vid_path, self.vid_writer, self.txt_path = [
            None] * self.nr_sources, [None] * self.nr_sources, [None] * self.nr_sources

        # init StrongSORT
        self.cfg = get_config()
        self._create_strongsort_instances()
        self.outputs = [None] * self.nr_sources
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def _create_strongsort_instances(self):
        self.strongsort_list = []
        for i in range(self.nr_sources):
            self.strongsort_list.append(
                StrongSORT(
                    self.strong_sort_weights,
                    self.device,
                    self.half,
                    mc_lambda=float(
                        os.getenv('STRONGSORT_MC_LAMBDA', '0.0')),
                    ema_alpha=float(
                        os.getenv('STRONGSORT_EMA_ALPHA', '0.0')),
                    max_dist=float(
                        os.getenv('STRONGSORT_MAX_DIST', '0.0')),
                    max_iou_distance=float(
                        os.getenv('STRONGSORT_MAX_IOU_DISTANCE', '0.0')),
                    max_age=int(
                        os.getenv('STRONGSORT_MAX_AGE', '0')),
                    n_init=int(
                        os.getenv('STRONGSORT_N_INIT', '0')),
                    nn_budget=int(
                        os.getenv('STRONGSORT_NN_BUDGET', '0'))
                )
            )
            self.strongsort_list[i].model.warmup()

    @torch.no_grad()
    def run(self):
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [
            None] * self.nr_sources, [None] * self.nr_sources

        for frame_idx, (path, im, im0s, vid_cap) in enumerate(self.dataset):
            s = ''
            t1 = time_synchronized()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            t2 = time_synchronized()
            dt[0] += t2 - t1

            # Inference
            self.visualize = increment_path(
                self.save_dir / Path(path[0]).stem, mkdir=True) if self.visualize else False
            pred = self.model(im)
            t3 = time_synchronized()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(
                pred[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms)
            dt[2] += time_synchronized() - t3

            # Process detections
            for i, det in enumerate(pred):
                seen += 1
                if self.webcam:
                    p, im0, _ = path[i], im0s[i].copy(), self.dataset.count
                    p = Path(p)
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(self.save_dir / p.name)
                else:
                    p, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    p = Path(p)

                    if self.source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(self.save_dir / p.name)

                    else:
                        txt_file_name = p.parent.name
                        save_path = str(self.save_dir / p.parent.name)

                curr_frames[i] = im0

                txt_path = str(self.save_dir / 'tracks' / txt_file_name)
                s += '%gx%g ' % im.shape[2:]
                imc = im0.copy() if self.save_crop else im0

                if (os.getenv('STRONGSORT_ECC', 'False') == 'True'):
                    self.strongsort_list[i].tracker.camera_update(
                        prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        im.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f'{n} {self.names[int(c)]}{"s" * (n > 1)}, '

                    xywhs = xyxy2xywh(det[:, :4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    t4 = time_synchronized()
                    self.outputs[i] = self.strongsort_list[i].update(
                        xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_synchronized()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(self.outputs[i]) > 0:

                        for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            # print('Stream: {} ID: {} \t Position: {}/{}'.format(self.stream_id, id, int((bboxes[0] + bboxes[2]) / 2), int(bboxes[3])))

                            point = Point('person').tag('stream', self.stream_id
                                                        ).tag('frame', frame_idx).tag('id', id
                                                                                      ).field('x', int((bboxes[0] + bboxes[2]) / 2)
                                                                                              ).field('y', int(bboxes[3]))
                            self.write_api.write(self.bucket, record=point)

                            if self.save_txt:
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]

                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id,
                                            bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if self.save_vid or self.save_crop or self.show_vid or True:
                                c = int(cls)
                                id = int(id)
                                label = None if self.hide_labels else (f'{id} {self.names[c]}' if self.hide_conf else
                                                                       (f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                                plot_one_box(bboxes, im0, label=label, color=self.colors[int(
                                    cls)], line_thickness=self.line_thickness)

                    print(
                        f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSort:({t5 - t4:.3f}s)')

                else:
                    self.strongsort_list[i].increment_ages()
                    print('No detections')

                if self.show_vid:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if self.rtmp_url is not None and self.rtmp_enable:
                    frame_bytes = im0.tostring()
                    self.rtmp_process.stdin.write(frame_bytes)

                if self.save_vid:
                    if self.vid_path[i] != save_path:
                        self.vid_path[i] = save_path
                        if isinstance(self.vid_writer[i], cv2.VideoWriter):
                            self.vid_writer[i].release()
                        if vid_cap:
                            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        self.vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4'), fps, (w, h))
                    self.vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]

            self.mp_barrier.wait()

        t = tuple(x / seen * 1E3 for x in dt)
        print(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, self.imgsz, self.imgsz)}' % t)
        if self.save_txt or self.save_crop:
            s = f"\n{len(list(self.save_dir.glob('tracks/*.txt')))} tracks saved to {self.save_dir / 'tracks'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.update:
            strip_optimizer(self.yolo_weights)

        if self.rtmp_url is not None and self.rtmp_enable:
            self.rtmp_process.stdin.close()
            self.rtmp_process.wait()


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # plot dot in lower center
    cv2.circle(img, (int((x[0] + x[2]) / 2), int(x[3])),
               8, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def run_stream(stream_id, barrier, stream_url, device, show_vid=False, show_matplot=False, rtmp_enable=False, rtmp_url=None):
    print(f'Running stream {stream_id}')
    d = StreamTracking(stream_id,
                       stream_url,
                       rtmp_enable=rtmp_enable,
                       rtmp_url=rtmp_url,
                       yolo_weights=WEIGHTS / 'yolov7.pt',
                       strong_sort_weights=WEIGHTS / 'osnet_x0_75_msmt17.pt',
                       mp_barrier=barrier,
                       device=device,
                       show_vid=show_vid,
                       show_matplot=show_matplot)
    d.run()


if __name__ == '__main__':
    from multiprocessing import Barrier, Process

    check_requirements(requirements=ROOT / 'requirements.txt',
                       exclude=('tensorboard', 'thop'))

    mp_barrier = Barrier(3)
    s1 = Process(target=run_stream, args=(
        1,
        mp_barrier,
        os.getenv('VIDEO_INPUT_1'),
        os.getenv('PROCESS_HARDWARE_1'),
        eval(os.getenv('PROCESS_SHOW_VIDEO', 'False')),
        eval(os.getenv('PROCESS_SHOW_MATPLOTLIB', 'False')),
        eval(os.getenv('VIDEO_OUTPUT_ENABLE', 'False')),
        os.getenv('VIDEO_OUTPUT_1')))

    s2 = Process(target=run_stream, args=(
        2,
        mp_barrier,
        os.getenv('VIDEO_INPUT_2'),
        os.getenv('PROCESS_HARDWARE_2'),
        eval(os.getenv('PROCESS_SHOW_VIDEO', 'False')),
        eval(os.getenv('PROCESS_SHOW_MATPLOTLIB', 'False')),
        eval(os.getenv('VIDEO_OUTPUT_ENABLE', 'False')),
        os.getenv('VIDEO_OUTPUT_2')))

    s3 = Process(target=run_stream, args=(
        3,
        mp_barrier,
        os.getenv('VIDEO_INPUT_3'),
        os.getenv('PROCESS_HARDWARE_3'),
        eval(os.getenv('PROCESS_SHOW_VIDEO', 'False')),
        eval(os.getenv('PROCESS_SHOW_MATPLOTLIB', 'False')),
        eval(os.getenv('VIDEO_OUTPUT_ENABLE', 'False')),
        os.getenv('VIDEO_OUTPUT_3')))

    s1.start()
    s2.start()
    s3.start()

    s1.join()
    s2.join()
    s3.join()

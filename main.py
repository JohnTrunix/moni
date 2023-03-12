from runner import run_moni
from runner_utils import check_packages

from multiprocessing import Process, Barrier

check_packages('./requirements.txt')

'''
run_moni(
        stream_id,
        source,
        yolo_weights,
        strong_sort_weights,
        mp_barrier,
        device='0',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        classes=[0],  # only detect and track persons (class 0)
        rtmp_enable=False,  # enable output stream to rtmp server
        rtmp_url=None,  # rtmp server url
        show_vid=True,  # shows video ouput with cv2
        show_plot=False,  # live plot output
        save_influx=False,  # save data to influxdb
        line_thickness=2,  # bounding boxes line thickness
        hide_labels=False,
        hide_conf=True,
        hide_class=False,
        half=False  # convert 32-bit tensor to 16-bit
):
'''

#----------------- Setup Multiprocessing -----------------#

mp_barrier = Barrier(3)  # 3 processes

s1 = Process(target=run_moni, args=())
s2 = Process(target=run_moni, args=())
s3 = Process(target=run_moni, args=())


#----------------- Run Multiprocessing -----------------#
s1.start()
s2.start()
s3.start()

s1.join()
s2.join()
s3.join()

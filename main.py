from runner import run_moni
from runner_utils import check_packages

from multiprocessing import Process, Barrier

check_packages('./requirements.txt')

'''
def run_moni(
        stream_id,
        source,
        yaml_config,
        mp_barrier=None,  # multiprocessing barrier object for synchronization
        device='0',
        imgsz=(640, 640),
        rtmp_url=None,
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

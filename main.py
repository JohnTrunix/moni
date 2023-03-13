from runner_utils import check_packages
check_packages('./requirements.txt')


import yaml
from runner import run_moni

from multiprocessing import Process, Barrier


#----------------- Function Documentation -----------------#
'''
def run_moni(
        stream_id,
        source,
        yaml_config,
        t_matrix=None,  # transformation matrix for perspective transform to global coordinates
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
yml_config = './conf/example-config.yml'

with open(yml_config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mp_barrier = Barrier(3)  # Barrier for 3 processes

s1 = Process(target=run_moni, args=(
    'Stream 1',
    config['rtmp']['video1'],
    yml_config,
    mp_barrier,
    config['hardware']['device1']
))

s2 = Process(target=run_moni, args=(
    'Stream 2',
    config['rtmp']['video2'],
    yml_config,
    mp_barrier,
    config['hardware']['device2']
))

s3 = Process(target=run_moni, args=(
    'Stream 3',
    config['rtmp']['video3'],
    yml_config,
    mp_barrier,
    config['hardware']['device3']
))


#----------------- Run Multiprocessing -----------------#
s1.start()
s2.start()
s3.start()

s1.join()
s2.join()
s3.join()

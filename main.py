from multiprocessing import Process, Barrier
from runner import run_moni
import yaml
from runner_utils import check_packages
check_packages('./requirements.txt')


# ----------------- Function Documentation -----------------#
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

# ----------------- Setup Multiprocessing -----------------#
yml_config = './conf/example-config.yml'

with open(yml_config, 'r', encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mp_barrier = Barrier(3)  # Barrier for 3 processes

s1 = Process(target=run_moni, args=(
    'Stream 1',
    config['rtmp']['input']['video_1'],
    yml_config,
    mp_barrier,
    config['hardware']['device_1'],
    config['rtmp']['output']['video_1']
))


s2 = Process(target=run_moni, args=(
    'Stream 2',
    config['rtmp']['input']['video_2'],
    yml_config,
    mp_barrier,
    config['hardware']['device_2'],
    config['rtmp']['output']['video_2']
))

s3 = Process(target=run_moni, args=(
    'Stream 3',
    config['rtmp']['input']['video_3'],
    yml_config,
    mp_barrier,
    config['hardware']['device_3'],
    config['rtmp']['output']['video_3']
))


# ----------------- Run Multiprocessing -----------------#
if __name__ == '__main__':
    s1.start()
    s2.start()
    s3.start()

    s1.join()
    s2.join()
    s3.join()

import argparse
import os
import yaml
from multiprocessing import Barrier, Process
from runner import run_moni
from runner_utils import check_packages


# ----------------- Argument parser -----------------#
def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist!".format(f))
    return f


parser = argparse.ArgumentParser(description='Moni')
parser.add_argument('--c',
                    default='./conf/example-config.yml',
                    type=validate_file,
                    dest='config_file',
                    help="path to config file.",
                    metavar="FILE")

args = parser.parse_args()


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
yml_config = args.config_file

with open(yml_config, 'r', encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mp_barrier = Barrier(3)  # Barrier for 3 processes

s1 = Process(target=run_moni, args=(
    'Stream 1',
    config['rtmp']['input']['video_1'],
    yml_config,
    mp_barrier,
    config['hardware']['device_1'],
    config['rtmp']['output']['video_1'],
    config['rtmp']['input']['t_matrix_1']
))


s2 = Process(target=run_moni, args=(
    'Stream 2',
    config['rtmp']['input']['video_2'],
    yml_config,
    mp_barrier,
    config['hardware']['device_2'],
    config['rtmp']['output']['video_2'],
    config['rtmp']['input']['t_matrix_2']
))

s3 = Process(target=run_moni, args=(
    'Stream 3',
    config['rtmp']['input']['video_3'],
    yml_config,
    mp_barrier,
    config['hardware']['device_3'],
    config['rtmp']['output']['video_3'],
    config['rtmp']['input']['t_matrix_3']
))


# ----------------- Run Multiprocessing -----------------#
if __name__ == '__main__':
    check_packages('./requirements.txt')

    s1.start()
    s2.start()
    s3.start()

    s1.join()
    s2.join()
    s3.join()

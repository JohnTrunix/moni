import os
import random
import time
from dotenv import load_dotenv
from multiprocessing import Process, Event

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

url = os.getenv('INFLUX_URL')
token = os.getenv('INFLUX_TOKEN')
org = os.getenv('INFLUX_ORG')
bucket = os.getenv('INFLUX_BUCKET')

client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)


def generate_data(stream_id, delay, event):
    i = 1
    while True:
        data = (stream_id, i, random.randint(
            0, 100), random.randint(0, 100))
        p = Point('frame4').tag('person_id', stream_id + 1).tag('frame_nr', data[1]).field(
            'x', data[2]).field('y', data[3])
        write_api.write(bucket, record=p)
        i += 1
        time.sleep(delay)
        event.set()
        event.clear()
        event.wait()


if __name__ == '__main__':
    event = Event()
    p1 = Process(target=generate_data, args=(1, 0.1, event))
    p2 = Process(target=generate_data, args=(2, 0.5, event))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

from influxdb_client import InfluxDBClient

load_dotenv()

url = os.getenv('INFLUX_URL')
token = os.getenv('INFLUX_TOKEN')
org = os.getenv('INFLUX_ORG')
bucket = os.getenv('INFLUX_BUCKET')
client = InfluxDBClient(url=url, token=token, org=org)
read_api = client.query_api()



H1 = np.array([[-1.6688907435, -6.9502305710, 940.69592392565],
               [1.1984806153, -10.7495778320, 868.29873467315],
               [0.0004069210, -0.0209324057, 0.42949125235]])


H2 = np.array([[0.6174778372, -0.4836875683, 147.00510919005],
               [0.5798503075, 3.8204849039, -386.096405131],
               [0.0000000001, 0.0077222239, -0.01593391935]])


H3 = np.array([[-0.2717592338, 1.0286363982, -17.6643219215],
                [-0.1373600672, -0.3326731339, 161.0109069274],
                [0.0000600052, 0.0030858398, -0.04195162855]])

def get_frame_data(i):
    query = f'from(bucket: "{bucket}") \
            |> range(start: -1h) \
            |> filter(fn: (r) => r._measurement == "person" and r.frame == "{i}") \
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") \
            |> keep(columns: ["frame", "id", "stream", "x", "y"])'


    table = read_api.query_data_frame(query)
    if table.empty:
        return 0, 0, 0, 0, 0, 0

    x1 = table[table['stream'] == '1']['x'].values if not table[table['stream'] == '1'].empty else 0
    y1 = table[table['stream'] == '1']['y'].values if not table[table['stream'] == '1'].empty else 0
    x2 = table[table['stream'] == '2']['x'].values if not table[table['stream'] == '2'].empty else 0
    y2 = table[table['stream'] == '2']['y'].values if not table[table['stream'] == '2'].empty else 0
    x3 = table[table['stream'] == '3']['x'].values if not table[table['stream'] == '3'].empty else 0
    y3 = table[table['stream'] == '3']['y'].values if not table[table['stream'] == '3'].empty else 0
    
    return x1, y1, x2, y2, x3, y3


def image_to_world_point(np_p, H):
    H_inv = np.linalg.inv(H)
    np_wp = np.dot(H_inv, np.append(np_p, [1])).astype(int)[:2]
    return np_wp


def plot(frame1, frame2, frame3, np_p1, np_p2, np_p3, pw1, pw2, pw3):
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.imshow(frame1)
    plt.scatter(np_p1[0][0], np_p1[0][1], c='r', s=10)

    plt.subplot(2, 2, 2)
    plt.imshow(frame2)
    plt.scatter(np_p2[0][0], np_p2[0][1], c='b', s=10)

    plt.subplot(2, 2, 3)
    plt.imshow(frame3)
    plt.scatter(np_p3[0][0], np_p3[0][1], c='g', s=10)

    # draw point in world
    plt.subplot(2, 2, 4)
    plt.scatter(pw1[0], pw1[1], c='r', s=10)
    plt.scatter(pw2[0], pw2[1], c='b', s=10)
    plt.scatter(pw3[0], pw3[1], c='g', s=10)
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)


    plt.draw()
    plt.pause(0.001)



cap1 = cv2.VideoCapture('./data/terrace/terrace1-c0.avi')
cap2 = cv2.VideoCapture('./data/terrace/terrace1-c1.avi')
cap3 = cv2.VideoCapture('./data/terrace/terrace1-c2.avi')

for i in range(200, 5000):
    x1, y1, x2, y2, x3, y3 = get_frame_data(i)

    np_p1 = np.array([[int((x1 + x1) / 2), int(y1)]])
    np_p2 = np.array([[int((x2 + x2) / 2), int(y2)]])
    np_p3 = np.array([[int((x3 + x3) / 2), int(y3)]])

    pw1 = image_to_world_point(np_p1, H1)
    pw2 = image_to_world_point(np_p2, H2)
    pw3 = image_to_world_point(np_p3, H3)



    cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret1, frame1 = cap1.read()


    cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret2, frame2 = cap2.read()


    cap3.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret3, frame3 = cap3.read()

    print('eucl distance pw1-pw2: ', np.linalg.norm(pw1 - pw2))
    print('eucl distance pw1-pw3: ', np.linalg.norm(pw1 - pw3))
    print('eucl distance pw2-pw3: ', np.linalg.norm(pw2 - pw3))
    print('------------------------------------------------')


    plot(frame1, frame2, frame3, np_p1, np_p2, np_p3, pw1, pw2, pw3)


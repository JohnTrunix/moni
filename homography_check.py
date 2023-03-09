import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

url = os.getenv('INFLUX_URL')
token = os.getenv('INFLUX_TOKEN')
org = os.getenv('INFLUX_ORG')
bucket = os.getenv('INFLUX_BUCKET')
client = InfluxDBClient(url=url, token=token, org=org)
read_api = client.query_api()

with open('./data/calibration_terrace.json') as f:
    data = json.load(f)



H1 = np.array(data['cameras'][0]['H_Ground_Plane'])
H2 = np.array(data['cameras'][1]['H_Ground_Plane'])
H3 = np.array(data['cameras'][2]['H_Ground_Plane'])

H1_inv = np.linalg.inv(H1)
H2_inv = np.linalg.inv(H2)
H3_inv = np.linalg.inv(H3)


def get_frame_data(i):
    query = f'from(bucket: "{bucket}") \
            |> range(start: -1000h) \
            |> filter(fn: (r) => r._measurement == "person" and r.frame == "{i}") \
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") \
            |> keep(columns: ["frame", "id", "stream", "x", "y"])'


    table = read_api.query_data_frame(query)
    if table.empty:
        return None
    else:
        df1 = table[table['stream'] == '1']
        df2 = table[table['stream'] == '2']
        df3 = table[table['stream'] == '3']
        return df1, df2, df3 


def local_to_global(np_p, H_inv):
    np_p = np.hstack((np_p, 1))
    wp = np.dot(H_inv, np_p)
    wp = wp[:-1] / wp[-1]
    return wp


def plot(frame1, frame2, frame3, d1, d2, d3, d1_new, d2_new, d3_new):
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.imshow(frame1)
    if d1 is not None:
        plt.scatter(d1[0], d1[1], c='r', s=10)

    plt.subplot(2, 2, 2)
    plt.imshow(frame2)
    if d2 is not None:
        plt.scatter(d2[0], d2[1], c='b', s=10)

    plt.subplot(2, 2, 3)
    plt.imshow(frame3)
    if d3 is not None:
        plt.scatter(d3[0], d3[1], c='g', s=10)

    # draw point in world
    plt.subplot(2, 2, 4)
    if d1_new is not None:
        plt.scatter(d1_new[0], d1_new[1], c='r', s=10)
    if d2_new is not None:
        plt.scatter(d2_new[0], d2_new[1], c='b', s=10)
    if d3_new is not None:
        plt.scatter(d3_new[0], d3_new[1], c='g', s=10)




    plt.draw()
    plt.pause(0.001)



cap1 = cv2.VideoCapture('./data/terrace/terrace1-c0.avi')
cap2 = cv2.VideoCapture('./data/terrace/terrace1-c1.avi')
cap3 = cv2.VideoCapture('./data/terrace/terrace1-c2.avi')

for i in range(200, 5000, 10):
    df1, df2, df3 = get_frame_data(i)
    d1x, d1y = [], []
    d1_newx, d1_newy = [], []
    d2x, d2y = [], []
    d2_newx, d2_newy = [], []
    d3x, d3y = [], []
    d3_newx, d3_newy = [], []


    # for each df x, y tuple call local_to_global
    for index, row in df1.iterrows():
        x1, y1 = row['x'], row['y']
        d1x.append(x1)
        d1y.append(y1)
        p = local_to_global(np.array([x1, y1]), H1_inv)
        d1_newx.append(p[0])
        d1_newy.append(p[1])

    for index, row in df2.iterrows():
        x2, y2 = row['x'], row['y']
        d2x.append(x2)
        d2y.append(y2)
        p = local_to_global(np.array([x2, y2]), H2_inv)
        d2_newx.append(p[0])
        d2_newy.append(p[1])

    for index, row in df3.iterrows():
        x3, y3 = row['x'], row['y']
        d3x.append(x3)
        d3y.append(y3)
        p = local_to_global(np.array([x3, y3]), H3_inv)
        d3_newx.append(p[0])
        d3_newy.append(p[1])

    cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret1, frame1 = cap1.read()


    cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret2, frame2 = cap2.read()


    cap3.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret3, frame3 = cap3.read()

    d1 = [d1x, d1y]
    d2 = [d2x, d2y]
    d3 = [d3x, d3y]
    d1_new = [d1_newx, d1_newy]
    d2_new = [d2_newx, d2_newy]
    d3_new = [d3_newx, d3_newy]

    plot(frame1, frame2, frame3, d1, d2, d3, d1_new, d2_new, d3_new)


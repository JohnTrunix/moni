import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
import plotly.express as px


load_dotenv()

url = os.getenv('INFLUX_URL')
token = os.getenv('INFLUX_TOKEN')
org = os.getenv('INFLUX_ORG')
bucket = os.getenv('INFLUX_BUCKET')
client = InfluxDBClient(url=url, token=token, org=org)
read_api = client.query_api()



# Funktion zum Abrufen der Daten
def get_influx_data(frame):
    query = f'from(bucket: "{bucket}") \
            |> range(start: -1h) \
            |> filter(fn: (r) => r._measurement == "person" and r.frame == "{frame}") \
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") \
            |> keep(columns: ["frame", "id", "stream", "x", "y"])'


    table = read_api.query_data_frame(query)
    if table.empty:
        return pd.DataFrame(columns=['frame', 'id', 'stream', 'x', 'y'])
    return table

if __name__ == '__main__':
    # Erstelle leeres Streamlit-Widget
    col1, col2 = st.columns(2)

    with col1:
        placeholder1 = st.empty()
    with col2:
        placeholder2 = st.empty()
    
    for i in range(100, 1000):
        # Rufe Funktion auf, um Dataframe zu erhalten
        data = get_influx_data(i)
        fig = px.scatter(data, x='x', y='y', color='stream')
        fig.update_layout(xaxis=dict(range=[0, 500]), yaxis=dict(range=[0, 500]))
        
        # Überschreibe Inhalt des Widgets mit Dataframe
        placeholder1.dataframe(data)
        placeholder2.plotly_chart(fig)
        
        # Warte für 1 Sekunde
        time.sleep(0.01)
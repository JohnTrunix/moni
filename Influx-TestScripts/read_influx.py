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


query = 'from(bucket:"position_data") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "frame4") |> filter(fn: (r) => r.person_id == "2")'

result = read_api.query(org=org, query=query)
results = []
for table in result:
    print(table)

print(results)

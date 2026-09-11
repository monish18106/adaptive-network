from kafka import KafkaProducer
import pandas as pd
import json
import time

TOPIC = "raw_traffic"
BOOTSTRAP_SERVERS = "localhost:9092"

CSV_PATH = "data/processed/model_dataset.csv"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows Loaded: {len(df)}")

for idx, row in df.iterrows():

    record = row.to_dict()

    producer.send(
        TOPIC,
        value=record
    )

    if idx % 1000 == 0:
        print(f"Sent {idx} records")

    if idx > 0 and idx % 10000 == 0:
        producer.flush()

    time.sleep(0.01)  # optional for demo

producer.flush()
producer.close()

print("Streaming Complete")
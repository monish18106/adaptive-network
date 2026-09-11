from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "raw_traffic",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest"
)

for msg in consumer:
    print(msg.value)
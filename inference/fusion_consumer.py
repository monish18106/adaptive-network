from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "fusion_scores",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    group_id="fusion-worker-v2",
    auto_offset_reset="latest"
)

print("Fusion Consumer Started...")

for msg in consumer:
    print(msg.value)
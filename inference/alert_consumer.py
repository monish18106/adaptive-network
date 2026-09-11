from kafka import KafkaConsumer
import json


INPUT_TOPIC = "alerts"


consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="alert-viewer-v1"
)

print("Alert Consumer Started...")

for msg in consumer:

    data = msg.value

    print(
        f"STATUS: {data.get('status')} | "
        f"IF: {data.get('if_score'):.4f} | "
        f"AE: {data.get('ae_score'):.4f} | "
        f"FINAL: {data.get('final_score'):.4f} | "
        f"THRESHOLD: {data.get('threshold'):.2f}"
    )
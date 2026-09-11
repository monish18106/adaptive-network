from kafka import KafkaConsumer, KafkaProducer
import json
import time


INPUT_TOPIC = "anomaly_scores"
OUTPUT_TOPIC = "fusion_scores"


consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="fusion-worker-v2"
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

processed = 0

print("Fusion Worker Started...")

for msg in consumer:

    data = msg.value

    # Skip old messages
    if "ae_score" not in data:
        continue

    if_score = float(data["if_score"])
    ae_score = float(data["ae_score"])

    final_score = (
        0.2 * if_score +
        0.8 * ae_score
    )

    result = {
        "features": data.get("features"),
        "if_score": if_score,
        "ae_score": ae_score,
        "final_score": final_score,
        "timestamp": time.time()
    }

    producer.send(
        OUTPUT_TOPIC,
        value=result
    )

    processed += 1

    if processed % 1000 == 0:
        print(f"Fused {processed} records")
        producer.flush()
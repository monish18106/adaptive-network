from kafka import KafkaConsumer, KafkaProducer
import json
import time


INPUT_TOPIC = "fusion_scores"
OUTPUT_TOPIC = "alerts"

THRESHOLD = 0.85


consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="alert-worker-v1"
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

processed = 0

print("Alert Worker Started...")


for msg in consumer:

    data = msg.value

    if "final_score" not in data:
        continue

    final_score = float(data["final_score"])

    if final_score >= THRESHOLD:
        status = "ANOMALY"
    else:
        status = "NORMAL"

    result = {
        "features": data.get("features"),
        "if_score": data.get("if_score"),
        "ae_score": data.get("ae_score"),
        "final_score": final_score,
        "threshold": THRESHOLD,
        "status": status,
        "timestamp": time.time()
    }

    producer.send(
        OUTPUT_TOPIC,
        value=result
    )

    processed += 1

    if processed % 1000 == 0:
        producer.flush()
        print(f"Processed {processed} alerts")
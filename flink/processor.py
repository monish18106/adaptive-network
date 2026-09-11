from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import joblib
import json

INPUT_TOPIC = "raw_traffic"
OUTPUT_TOPIC = "processed_features"
BOOTSTRAP_SERVERS = "localhost:9092"

print("Loading scaler...")
scaler = joblib.load("models/saved_models/scaler.pkl")

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="feature-processor"
)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Processor started...")

processed_count = 0

for msg in consumer:

    record = msg.value

    # Remove label
    record.pop("Label", None)

    # Convert to dataframe
    df = pd.DataFrame([record])

    # Scale
    scaled = scaler.transform(df)

    output = {
        "features": scaled[0].tolist()
    }

    producer.send(
        OUTPUT_TOPIC,
        value=output
    )

    processed_count += 1

    if processed_count % 1000 == 0:
        print(f"Processed {processed_count}")
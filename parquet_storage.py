from kafka import KafkaConsumer
import json
import os
import time
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_TOPIC = "alerts"
BOOTSTRAP_SERVERS = "localhost:9092"

OUTPUT_DIR = "data/history"

BATCH_SIZE = 100


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(
        x.decode("utf-8")
    ),
    auto_offset_reset="latest",
    group_id="parquet-storage-worker-v1"
)


# ============================================================
# VARIABLES
# ============================================================

records = []

file_counter = 0


print("=" * 80)
print("PARQUET STORAGE WORKER")
print("=" * 80)

print(f"Kafka Topic : {INPUT_TOPIC}")
print(f"Output Dir  : {OUTPUT_DIR}")
print(f"Batch Size  : {BATCH_SIZE}")

print("\nWaiting for alert records...\n")


# ============================================================
# SAVE BATCH TO PARQUET
# ============================================================

def save_batch(batch):

    global file_counter

    if not batch:
        return

    df = pd.DataFrame(batch)

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"alerts_{timestamp}_{file_counter:04d}.parquet"
    )

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_parquet(
        filepath,
        index=False
    )

    file_counter += 1

    print(
        f"Saved {len(df)} records → {filepath}"
    )


# ============================================================
# CONSUME ALERTS
# ============================================================

processed = 0

try:

    for msg in consumer:

        data = msg.value

        # ----------------------------------------------------
        # Validate alert record
        # ----------------------------------------------------

        if "final_score" not in data:
            continue

        if "status" not in data:
            continue

        # ----------------------------------------------------
        # Store record
        # ----------------------------------------------------

        record = {
            "features": data.get("features"),
            "if_score": data.get("if_score"),
            "ae_score": data.get("ae_score"),
            "final_score": data.get("final_score"),
            "threshold": data.get("threshold"),
            "status": data.get("status"),
            "timestamp": data.get(
                "timestamp",
                time.time()
            )
        }

        records.append(record)

        processed += 1

        # ----------------------------------------------------
        # Save batch
        # ----------------------------------------------------

        if len(records) >= BATCH_SIZE:

            save_batch(records)

            records.clear()

            print(
                f"Total processed: {processed}"
            )


except KeyboardInterrupt:

    print("\nStopping worker...")

    # Save remaining records
    if records:
        save_batch(records)

    print(
        f"\nTotal records processed: {processed}"
    )

    print("Parquet Storage Worker stopped.")
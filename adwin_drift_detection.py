import os
import glob
import json
import time

import pandas as pd
from river.drift import ADWIN


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HISTORY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "history"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_drift_results.csv"
)

STATE_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_state.json"
)

TRIGGER_FILE = os.path.join(
    PROCESSED_DIR,
    "retraining_trigger.json"
)


# ============================================================
# ADWIN CONFIGURATION
# ============================================================

DELTA = 0.002

# Run ADWIN after every 10,000 records.
CHECKPOINT_SIZE = 10000

# Number of drift events required before retraining.
# Keep this configurable for demonstration.
DRIFT_TRIGGER_COUNT = 50


# ============================================================
# HELPERS
# ============================================================

def save_json(path, data):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def load_json(path):

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ============================================================
# LOAD PARQUET HISTORY
# ============================================================

def load_history():

    files = sorted(
        glob.glob(
            os.path.join(
                HISTORY_DIR,
                "*.parquet"
            )
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No parquet files found in {HISTORY_DIR}"
        )

    print(
        f"Parquet files found: {len(files)}"
    )

    frames = []

    for file in files:

        try:

            df = pd.read_parquet(file)

            if len(df) > 0:
                frames.append(df)

        except Exception as e:

            print(
                f"Warning: failed to read {file}: {e}"
            )

    if not frames:

        raise ValueError(
            "No valid parquet history found."
        )

    history = pd.concat(
        frames,
        ignore_index=True
    )

    return history


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("ADWIN DRIFT DETECTION")
print("=" * 80)

print()
print(
    f"Checkpoint size       : {CHECKPOINT_SIZE}"
)

print(
    f"ADWIN delta           : {DELTA}"
)

print(
    f"Drift trigger count   : {DRIFT_TRIGGER_COUNT}"
)

print()


# ============================================================
# LOAD HISTORY
# ============================================================

history = load_history()

print(
    f"Total historical records: {len(history)}"
)


# ============================================================
# VERIFY SCORE
# ============================================================

if "final_score" not in history.columns:

    raise ValueError(
        "final_score column not found in parquet history."
    )


history["final_score"] = pd.to_numeric(
    history["final_score"],
    errors="coerce"
)

history = history.dropna(
    subset=["final_score"]
).reset_index(drop=True)


print(
    f"Valid score records      : {len(history)}"
)


# ============================================================
# LOAD PREVIOUS STATE
# ============================================================

previous_state = load_json(
    STATE_FILE
)

previous_processed_records = int(
    previous_state.get(
        "processed_records",
        0
    )
)

previous_drift_count = int(
    previous_state.get(
        "total_drifts",
        0
    )
)

previous_triggered_count = int(
    previous_state.get(
        "retraining_trigger_count",
        0
    )
)


# ============================================================
# DETERMINE PROCESSING RANGE
# ============================================================

current_records = len(history)

if current_records < CHECKPOINT_SIZE:

    print()
    print(
        f"Waiting for {CHECKPOINT_SIZE} records."
    )

    print(
        f"Current records: {current_records}"
    )

    state = {
        "status": "WAITING_FOR_CHECKPOINT",
        "processed_records": current_records,
        "total_records": current_records,
        "checkpoint_size": CHECKPOINT_SIZE,
        "next_checkpoint": CHECKPOINT_SIZE,
        "total_drifts": previous_drift_count,
        "drift_trigger_count": DRIFT_TRIGGER_COUNT,
        "retraining_trigger_count":
            previous_triggered_count,
        "last_updated": time.time()
    }

    save_json(
        STATE_FILE,
        state
    )

    print()
    print(
        "ADWIN checkpoint not reached yet."
    )

    raise SystemExit(0)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_number = (
    current_records // CHECKPOINT_SIZE
)

checkpoint_end = (
    checkpoint_number * CHECKPOINT_SIZE
)


print()
print("=" * 80)
print("ADWIN CHECKPOINT")
print("=" * 80)

print(
    f"Records available : {current_records}"
)

print(
    f"Checkpoint        : {checkpoint_number}"
)

print(
    f"Processing through: {checkpoint_end}"
)


# ============================================================
# IMPORTANT:
# REPLAY HISTORY FROM START
#
# This guarantees deterministic ADWIN results even when
# this script is restarted.
# ============================================================

adwin = ADWIN(
    delta=DELTA
)


drift_results = []

drift_count = 0


# ============================================================
# PROCESS HISTORY
# ============================================================

for index in range(checkpoint_end):

    score = float(
        history.iloc[index]["final_score"]
    )

    adwin.update(
        score
    )

    drift_detected = False

    if adwin.drift_detected:

        drift_detected = True

        drift_count += 1

        print(
            f"DRIFT DETECTED "
            f"at record {index}"
        )

    drift_results.append({

        "record_index":
            index,

        "final_score":
            score,

        "drift_detected":
            drift_detected,

        "adwin_width":
            float(adwin.width),

        "adwin_estimation":
            float(adwin.estimation),

        "checkpoint":
            checkpoint_number

    })


# ============================================================
# CONVERT RESULTS
# ============================================================

result_df = pd.DataFrame(
    drift_results
)


# ============================================================
# SAVE DRIFT RESULTS
# ============================================================

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# ACTUAL DRIFT COUNT
# ============================================================

actual_drift_count = int(
    (
        result_df["drift_detected"] == True
    ).sum()
)


# ============================================================
# RETRAINING TRIGGER LOGIC
# ============================================================

retraining_required = (
    actual_drift_count >=
    DRIFT_TRIGGER_COUNT
)


trigger_created = False


if retraining_required:

    trigger_data = {

        "trigger":
            "ADWIN_DRIFT_THRESHOLD_REACHED",

        "status":
            "RETRAINING_REQUIRED",

        "drift_count":
            actual_drift_count,

        "required_drift_count":
            DRIFT_TRIGGER_COUNT,

        "records_processed":
            checkpoint_end,

        "checkpoint":
            checkpoint_number,

        "timestamp":
            time.time()

    }

    save_json(
        TRIGGER_FILE,
        trigger_data
    )

    trigger_created = True

    print()
    print("=" * 80)
    print("RETRAINING TRIGGER")
    print("=" * 80)

    print(
        "ADWIN drift threshold reached."
    )

    print(
        f"Drifts detected : {actual_drift_count}"
    )

    print(
        f"Required        : {DRIFT_TRIGGER_COUNT}"
    )

    print(
        "Retraining trigger created."
    )

    print(
        f"Trigger file    : {TRIGGER_FILE}"
    )

else:

    # Remove stale trigger if threshold is not reached.
    if os.path.exists(TRIGGER_FILE):

        try:
            os.remove(
                TRIGGER_FILE
            )
        except Exception:
            pass


# ============================================================
# STATE
# ============================================================

next_checkpoint = (
    (checkpoint_number + 1)
    * CHECKPOINT_SIZE
)


state = {

    "status":
        "RETRAINING_REQUIRED"
        if retraining_required
        else "MONITORING",

    "processed_records":
        checkpoint_end,

    "total_records":
        current_records,

    "checkpoint_size":
        CHECKPOINT_SIZE,

    "current_checkpoint":
        checkpoint_number,

    "next_checkpoint":
        next_checkpoint,

    "total_drifts":
        actual_drift_count,

    "drift_trigger_count":
        DRIFT_TRIGGER_COUNT,

    "retraining_required":
        retraining_required,

    "retraining_trigger_created":
        trigger_created,

    "retraining_trigger_count":
        previous_triggered_count
        + (1 if trigger_created else 0),

    "adwin_delta":
        DELTA,

    "last_drift_record":
        int(
            result_df[
                result_df[
                    "drift_detected"
                ] == True
            ]["record_index"].iloc[-1]
        )
        if actual_drift_count > 0
        else None,

    "last_drift_score":
        float(
            result_df[
                result_df[
                    "drift_detected"
                ] == True
            ]["final_score"].iloc[-1]
        )
        if actual_drift_count > 0
        else None,

    "last_updated":
        time.time()
}


save_json(
    STATE_FILE,
    state
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("ADWIN ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Records processed : {checkpoint_end}"
)

print(
    f"Total history     : {current_records}"
)

print(
    f"Checkpoint        : {checkpoint_number}"
)

print(
    f"Drifts detected   : {actual_drift_count}"
)

print(
    f"Trigger threshold : {DRIFT_TRIGGER_COUNT}"
)

print(
    f"Retraining needed : "
    f"{retraining_required}"
)

print(
    f"Results saved     : {OUTPUT_FILE}"
)

print(
    f"State saved       : {STATE_FILE}"
)


# ============================================================
# SHOW DRIFT EVENTS
# ============================================================

drifts = result_df[
    result_df["drift_detected"] == True
]


print()
print("=" * 80)
print("DRIFT EVENTS")
print("=" * 80)


if len(drifts) == 0:

    print(
        "No drift detected in this checkpoint."
    )

else:

    print(
        drifts.tail(20).to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("NEXT CHECKPOINT")
print("=" * 80)

print(
    f"Next ADWIN checkpoint: "
    f"{next_checkpoint} records"
)

print()


# ============================================================
# IMPORTANT NOTE
# ============================================================

if retraining_required:

    print(
        "Automatic orchestrator should now "
        "start PySpark retraining."
    )

else:

    print(
        "Continue monitoring until the "
        "next 10,000-record checkpoint."
    )

print("=" * 80)
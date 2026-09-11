# ============================================================
# SELF-CORRECTION CONTROLLER
# ============================================================
#
# Automatic flow:
#
#   Parquet reaches 10,000
#          ↓
#       ADWIN
#          ↓
#     Drift detected?
#       /       \
#     NO         YES
#     │           │
#   Wait      PySpark
#               ↓
#          Evaluation
#               ↓
#          Lifecycle
#
# Then continue to:
#
#   20,000
#   30,000
#   40,000
#   ...
#
# ============================================================

import os
import sys
import json
import time
import subprocess


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

STATE_FILE = os.path.join(
    PROCESSED_DIR,
    "self_correction_state.json"
)

ADWIN_RESULT_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_latest_result.json"
)

ADWIN_SCRIPT = os.path.join(
    BASE_DIR,
    "adwin_drift_detection.py"
)

PYSPARK_SCRIPT = os.path.join(
    BASE_DIR,
    "pyspark_retraining.py"
)

EVALUATION_SCRIPT = os.path.join(
    BASE_DIR,
    "model_evaluation.py"
)

LIFECYCLE_SCRIPT = os.path.join(
    BASE_DIR,
    "model_lifecycle.py"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_SIZE = 10000

POLL_SECONDS = 5

PYTHON_EXECUTABLE = sys.executable


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# STATE
# ============================================================

DEFAULT_STATE = {

    "last_completed_checkpoint": 0,

    "total_adwin_runs": 0,

    "total_drift_events": 0,

    "total_retraining_runs": 0,

    "last_drift_detected": False,

    "last_status": "STARTING",

    "last_run_timestamp": None

}


# ============================================================
# STATE FUNCTIONS
# ============================================================

def load_state():

    if not os.path.exists(
        STATE_FILE
    ):

        return DEFAULT_STATE.copy()

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        for key, value in DEFAULT_STATE.items():

            if key not in state:

                state[key] = value

        return state

    except Exception as e:

        print(
            f"Warning: state load failed: {e}"
        )

        return DEFAULT_STATE.copy()


def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            indent=4
        )


# ============================================================
# HISTORY COUNT
# ============================================================

def get_record_count():

    import glob
    import pandas as pd

    history_dir = os.path.join(
        BASE_DIR,
        "data",
        "history"
    )

    files = sorted(
        glob.glob(
            os.path.join(
                history_dir,
                "*.parquet"
            )
        )
    )

    if not files:

        return 0

    total = 0

    for file in files:

        try:

            df = pd.read_parquet(
                file,
                columns=["final_score"]
            )

            total += len(df)

        except Exception:

            try:

                df = pd.read_parquet(
                    file
                )

                total += len(df)

            except Exception:

                pass

    return total


# ============================================================
# RUN SCRIPT
# ============================================================

def run_script(
    script,
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    result = subprocess.run(
        [
            PYTHON_EXECUTABLE,
            script
        ],
        cwd=BASE_DIR,
        check=False
    )

    if result.returncode != 0:

        print(
            f"ERROR: {title} failed."
        )

        return False

    print(
        f"{title} completed successfully."
    )

    return True


# ============================================================
# READ ADWIN RESULT
# ============================================================

def read_adwin_result():

    if not os.path.exists(
        ADWIN_RESULT_FILE
    ):

        return {}

    try:

        with open(
            ADWIN_RESULT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ============================================================
# RETRAINING PIPELINE
# ============================================================

def run_self_correction(
    state,
    checkpoint
):

    print()
    print("=" * 80)
    print("DRIFT DETECTED - STARTING AUTOMATIC SELF-CORRECTION")
    print("=" * 80)

    # --------------------------------------------------------
    # STEP 1 - PYSPARK
    # --------------------------------------------------------

    success = run_script(
        PYSPARK_SCRIPT,
        "STEP 1 - PYSPARK RETRAINING"
    )

    if not success:

        state["last_status"] = (
            "RETRAINING_FAILED"
        )

        save_state(
            state
        )

        return False


    # --------------------------------------------------------
    # STEP 2 - EVALUATION
    # --------------------------------------------------------

    success = run_script(
        EVALUATION_SCRIPT,
        "STEP 2 - MODEL EVALUATION"
    )

    if not success:

        state["last_status"] = (
            "EVALUATION_FAILED"
        )

        save_state(
            state
        )

        return False


    # --------------------------------------------------------
    # STEP 3 - MODEL LIFECYCLE
    # --------------------------------------------------------

    success = run_script(
        LIFECYCLE_SCRIPT,
        "STEP 3 - MODEL LIFECYCLE"
    )

    if not success:

        state["last_status"] = (
            "LIFECYCLE_FAILED"
        )

        save_state(
            state
        )

        return False


    state["total_retraining_runs"] += 1

    state["last_status"] = (
        "SELF_CORRECTION_COMPLETE"
    )

    state["last_run_timestamp"] = (
        time.time()
    )

    save_state(
        state
    )

    print()
    print("=" * 80)
    print("AUTOMATIC SELF-CORRECTION COMPLETE")
    print("=" * 80)

    print(
        f"Checkpoint : {checkpoint}"
    )

    print(
        f"Retraining runs : "
        f"{state['total_retraining_runs']}"
    )

    print("=" * 80)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    state = load_state()

    print("=" * 80)
    print("AUTOMATIC SELF-CORRECTION CONTROLLER")
    print("=" * 80)

    print(
        f"Checkpoint size : {CHECKPOINT_SIZE}"
    )

    print(
        f"Polling         : {POLL_SECONDS}s"
    )

    print(
        f"Last checkpoint : "
        f"{state['last_completed_checkpoint']}"
    )

    print(
        f"ADWIN runs      : "
        f"{state['total_adwin_runs']}"
    )

    print(
        f"Retraining runs : "
        f"{state['total_retraining_runs']}"
    )

    print("=" * 80)

    while True:

        try:

            total_records = get_record_count()

            # ------------------------------------------------
            # DETERMINE COMPLETED CHECKPOINT
            # ------------------------------------------------

            completed_checkpoint = (
                total_records // CHECKPOINT_SIZE
            ) * CHECKPOINT_SIZE

            last_checkpoint = int(
                state[
                    "last_completed_checkpoint"
                ]
            )

            # ------------------------------------------------
            # NOTHING NEW YET
            # ------------------------------------------------

            if (
                completed_checkpoint
                <= last_checkpoint
            ):

                print(
                    f"\rRecords: "
                    f"{total_records:,} | "
                    f"Next checkpoint: "
                    f"{last_checkpoint + CHECKPOINT_SIZE:,}",
                    end="",
                    flush=True
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # NEW CHECKPOINT
            # ------------------------------------------------

            checkpoint = completed_checkpoint

            print()
            print()
            print("=" * 80)
            print(
                f"NEW CHECKPOINT REACHED: "
                f"{checkpoint:,} RECORDS"
            )
            print("=" * 80)

            # ------------------------------------------------
            # RUN ADWIN
            # ------------------------------------------------

            success = run_script(
                ADWIN_SCRIPT,
                f"ADWIN ANALYSIS - CHECKPOINT {checkpoint:,}"
            )

            if not success:

                state["last_status"] = (
                    "ADWIN_FAILED"
                )

                save_state(
                    state
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            state["total_adwin_runs"] += 1

            # ------------------------------------------------
            # READ RESULT
            # ------------------------------------------------

            result = read_adwin_result()

            result_checkpoint = int(
                result.get(
                    "checkpoint",
                    0
                )
            )

            drift_detected = bool(
                result.get(
                    "drift_detected",
                    False
                )
            )

            drift_count = int(
                result.get(
                    "drift_count",
                    0
                )
            )

            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if result_checkpoint != checkpoint:

                print(
                    "WARNING: ADWIN checkpoint mismatch."
                )

                print(
                    f"Expected: {checkpoint}"
                )

                print(
                    f"Received: {result_checkpoint}"
                )

                state["last_status"] = (
                    "ADWIN_CHECKPOINT_MISMATCH"
                )

                save_state(
                    state
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # SAVE DRIFT INFORMATION
            # ------------------------------------------------

            state[
                "total_drift_events"
            ] += drift_count

            state[
                "last_drift_detected"
            ] = drift_detected

            # ------------------------------------------------
            # DRIFT
            # ------------------------------------------------

            if drift_detected:

                state["last_status"] = (
                    "DRIFT_DETECTED"
                )

                save_state(
                    state
                )

                run_self_correction(
                    state,
                    checkpoint
                )

            # ------------------------------------------------
            # NO DRIFT
            # ------------------------------------------------

            else:

                state["last_status"] = (
                    "NO_DRIFT"
                )

                state["last_run_timestamp"] = (
                    time.time()
                )

                save_state(
                    state
                )

                print()
                print("=" * 80)
                print("NO DRIFT DETECTED")
                print("=" * 80)

                print(
                    f"Checkpoint : "
                    f"{checkpoint:,}"
                )

                print(
                    "Production model remains unchanged."
                )

                print("=" * 80)

            # ------------------------------------------------
            # MARK CHECKPOINT COMPLETE
            # ------------------------------------------------

            state[
                "last_completed_checkpoint"
            ] = checkpoint

            save_state(
                state
            )

            print()
            print(
                f"Next ADWIN checkpoint: "
                f"{checkpoint + CHECKPOINT_SIZE:,}"
            )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print()
            print(
                "Self-correction controller stopped."
            )

            save_state(
                state
            )

            break

        except Exception as e:

            print()
            print(
                f"Controller error: {e}"
            )

            state["last_status"] = (
                "CONTROLLER_ERROR"
            )

            save_state(
                state
            )

            time.sleep(
                POLL_SECONDS
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
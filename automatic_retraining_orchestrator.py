import os
import sys
import json
import time
import glob
import subprocess
from datetime import datetime

import pandas as pd


# ============================================================
# PHASE 16
# AUTOMATIC DRIFT → RETRAINING → EVALUATION → LIFECYCLE
# ============================================================
#
# Continuous pipeline:
#
# Parquet history
#       ↓
# 10,000-record checkpoint
#       ↓
# ADWIN
#       ↓
# Drift?
#   NO  → continue monitoring
#   YES
#       ↓
# PySpark retraining
#       ↓
# Model evaluation
#       ↓
# Model lifecycle
#       ↓
# Promote / Reject / Rollback
#
# ADWIN checkpoints:
#   10,000
#   20,000
#   30,000
#   ...
#
# PySpark trains using ALL available historical NORMAL traffic.
#
# ============================================================


# ============================================================
# BASE DIRECTORIES
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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

PRODUCTION_DIR = os.path.join(
    MODELS_DIR,
    "production"
)

CANDIDATES_DIR = os.path.join(
    MODELS_DIR,
    "candidates"
)


# ============================================================
# STATE
# ============================================================

STATE_FILE = os.path.join(
    PROCESSED_DIR,
    "automatic_retraining_state.json"
)


# ============================================================
# DASHBOARD PIPELINE STATUS
# ============================================================

PIPELINE_STATUS_FILE = os.path.join(
    PROCESSED_DIR,
    "automatic_pipeline_status.json"
)


# ============================================================
# PIPELINE HISTORY
# ============================================================

PIPELINE_HISTORY_FILE = os.path.join(
    PROCESSED_DIR,
    "automatic_pipeline_history.json"
)


# ============================================================
# PIPELINE LOG
# ============================================================

PIPELINE_LOG = os.path.join(
    PROCESSED_DIR,
    "retraining_pipeline.log"
)


# ============================================================
# RETRAINING LOCK
# ============================================================

RETRAINING_LOCK_FILE = os.path.join(
    PROCESSED_DIR,
    "retraining_in_progress.lock"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY_FILE = os.path.join(
    MODELS_DIR,
    "model_registry.json"
)


# ============================================================
# EXTERNAL SCRIPTS
# ============================================================

ADWIN_SCRIPT = os.path.join(
    BASE_DIR,
    "adwin_drift_detection.py"
)

RETRAIN_SCRIPT = os.path.join(
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
# ADWIN OUTPUT
# ============================================================

ADWIN_RESULT_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_drift_results.csv"
)

ADWIN_LATEST_RESULT_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_latest_result.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_SIZE = 10_000

CHECK_INTERVAL_SECONDS = 10

PYTHON_EXECUTABLE = sys.executable


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# PRINT / LOG HELPERS
# ============================================================

def separator():

    print(
        "=" * 80,
        flush=True
    )


def log(message):

    message = str(message)

    print(
        message,
        flush=True
    )

    append_pipeline_log(
        message
    )


def append_pipeline_log(message):

    try:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            PIPELINE_LOG,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"[{timestamp}] {message}\n"
            )

    except Exception:
        pass


# ============================================================
# JSON HELPERS
# ============================================================

def safe_read_json(path):

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except Exception as error:

        log(
            f"WARNING: Could not read JSON "
            f"{path}: {error}"
        )

        return {}


def safe_write_json(path, data):

    try:

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        temporary_path = (
            path
            + ".tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                default=str
            )

        os.replace(
            temporary_path,
            path
        )

        return True

    except Exception as error:

        log(
            f"WARNING: Could not write "
            f"{path}: {error}"
        )

        return False


# ============================================================
# DASHBOARD PIPELINE STATUS
# ============================================================

def save_pipeline_status(

    status="RUNNING",

    stage="",

    checkpoint=0,

    drift_count=0,

    retraining=False,

    evaluation="NOT_STARTED",

    lifecycle="NOT_STARTED",

    final_result="NONE",

    message="",

    training_times=None,

    drift_events=None,

    extra=None

):

    if training_times is None:
        training_times = {}

    if drift_events is None:
        drift_events = []

    if extra is None:
        extra = {}

    data = {

        "status":
            status,

        "stage":
            stage,

        "checkpoint":
            int(checkpoint),

        "drift_count":
            int(drift_count),

        "retraining":
            bool(retraining),

        "evaluation":
            evaluation,

        "lifecycle":
            lifecycle,

        "final_result":
            final_result,

        "message":
            message,

        "training_times":
            training_times,

        "drift_events":
            drift_events,

        "timestamp":
            time.time(),

        "updated_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    }

    data.update(
        extra
    )

    safe_write_json(
        PIPELINE_STATUS_FILE,
        data
    )


# ============================================================
# PIPELINE HISTORY
# ============================================================

def append_pipeline_history(status_data):

    history = []

    if os.path.exists(
        PIPELINE_HISTORY_FILE
    ):

        try:

            with open(
                PIPELINE_HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history = json.load(
                    file
                )

            if not isinstance(
                history,
                list
            ):

                history = []

        except Exception:

            history = []

    history.append(
        status_data
    )

    # Keep dashboard history manageable.
    history = history[-100:]

    safe_write_json(
        PIPELINE_HISTORY_FILE,
        history
    )


def save_status_and_history(

    status="RUNNING",

    stage="",

    checkpoint=0,

    drift_count=0,

    retraining=False,

    evaluation="NOT_STARTED",

    lifecycle="NOT_STARTED",

    final_result="NONE",

    message="",

    training_times=None,

    drift_events=None,

    extra=None,

    add_history=False

):

    save_pipeline_status(

        status=status,

        stage=stage,

        checkpoint=checkpoint,

        drift_count=drift_count,

        retraining=retraining,

        evaluation=evaluation,

        lifecycle=lifecycle,

        final_result=final_result,

        message=message,

        training_times=training_times,

        drift_events=drift_events,

        extra=extra

    )

    if add_history:

        current = safe_read_json(
            PIPELINE_STATUS_FILE
        )

        if current:
            append_pipeline_history(
                current
            )


# ============================================================
# DEFAULT STATE
# ============================================================

def default_state():

    return {

        "last_processed_checkpoint":
            0,

        "last_adwin_record_count":
            0,

        "total_retraining_triggers":
            0,

        "last_trigger_record_count":
            None,

        "last_trigger_reason":
            None,

        "last_run_timestamp":
            None,

        "last_drift_count":
            0,

        "last_drift_indices":
            [],

        "known_drift_indices":
            [],

        "total_checkpoints_processed":
            0

    }


# ============================================================
# LOAD STATE
# ============================================================

def load_state():

    state = default_state()

    if not os.path.exists(
        STATE_FILE
    ):

        return state

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            saved = json.load(
                file
            )

        if isinstance(
            saved,
            dict
        ):

            state.update(
                saved
            )

    except Exception as error:

        log(
            f"WARNING: Could not load state: "
            f"{error}"
        )

    return state


# ============================================================
# SAVE STATE
# ============================================================

def save_state(state):

    safe_write_json(
        STATE_FILE,
        state
    )


# ============================================================
# COUNT HISTORY
# ============================================================

def count_history_records():

    files = sorted(
        glob.glob(
            os.path.join(
                HISTORY_DIR,
                "*.parquet"
            )
        )
    )

    total = 0

    for file in files:

        try:

            df = pd.read_parquet(
                file,
                columns=["final_score"]
            )

            total += len(df)

        except Exception as error:

            log(
                f"WARNING: Could not read "
                f"{file}: {error}"
            )

    return total


# ============================================================
# COMPLETED CHECKPOINT
# ============================================================

def get_completed_checkpoint(
    record_count
):

    if record_count < CHECKPOINT_SIZE:
        return 0

    return (
        record_count
        // CHECKPOINT_SIZE
    ) * CHECKPOINT_SIZE


# ============================================================
# CLEAR OLD ADWIN OUTPUT
# ============================================================

def clear_old_adwin_results():

    for path in [

        ADWIN_RESULT_FILE,

        ADWIN_LATEST_RESULT_FILE

    ]:

        if os.path.exists(path):

            try:

                os.remove(path)

                log(
                    "Removed old ADWIN output: "
                    + os.path.basename(path)
                )

            except Exception as error:

                log(
                    f"WARNING: Could not remove "
                    f"{path}: {error}"
                )


# ============================================================
# RUN EXTERNAL SCRIPT
# ============================================================

def run_script(

    script_path,

    script_name,

    extra_env=None,

    status_callback=None

):

    separator()

    log(
        f"STARTING: {script_name}"
    )

    log(
        f"Script: {script_path}"
    )

    separator()

    if not os.path.exists(
        script_path
    ):

        log(
            f"ERROR: Script not found: "
            f"{script_path}"
        )

        return False, 0.0

    start_time = time.time()

    env = os.environ.copy()

    if extra_env:

        env.update(
            extra_env
        )

    try:

        process = subprocess.Popen(

            [
                PYTHON_EXECUTABLE,
                script_path
            ],

            cwd=BASE_DIR,

            env=env,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1

        )

        last_status_update = time.time()

        while True:

            line = process.stdout.readline()

            if line:

                line = line.rstrip()

                print(
                    line,
                    flush=True
                )

                append_pipeline_log(
                    line
                )

                # Keep dashboard timestamp/stage alive
                # while long-running PySpark executes.
                if (
                    status_callback
                    and
                    time.time()
                    - last_status_update
                    >= 5
                ):

                    try:

                        status_callback()

                    except Exception:
                        pass

                    last_status_update = (
                        time.time()
                    )

            elif process.poll() is not None:

                break

            else:

                time.sleep(
                    0.1
                )

        return_code = (
            process.returncode
        )

        elapsed = (
            time.time()
            - start_time
        )

        if return_code == 0:

            log(
                f"{script_name} completed "
                f"successfully in "
                f"{elapsed:.3f} seconds."
            )

            return True, elapsed

        log(
            f"{script_name} FAILED "
            f"with return code "
            f"{return_code} after "
            f"{elapsed:.3f} seconds."
        )

        return False, elapsed

    except Exception as error:

        elapsed = (
            time.time()
            - start_time
        )

        log(
            f"ERROR running {script_name}: "
            f"{error}"
        )

        return False, elapsed


# ============================================================
# LOAD ADWIN CSV
# ============================================================

def load_adwin_results():

    if not os.path.exists(
        ADWIN_RESULT_FILE
    ):

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            ADWIN_RESULT_FILE
        )

    except Exception as error:

        log(
            f"WARNING: Could not read ADWIN CSV: "
            f"{error}"
        )

        return pd.DataFrame()

    required = [

        "record_index",

        "drift_detected"

    ]

    for column in required:

        if column not in df.columns:

            log(
                f"WARNING: ADWIN result missing "
                f"column: {column}"
            )

            return pd.DataFrame()

    return df


# ============================================================
# LOAD ADWIN LATEST RESULT
# ============================================================

def load_adwin_latest_result():

    return safe_read_json(
        ADWIN_LATEST_RESULT_FILE
    )


# ============================================================
# GET CURRENT CHECKPOINT DRIFT
# ============================================================

def get_current_drift_events(
    checkpoint
):

    latest = load_adwin_latest_result()

    # --------------------------------------------------------
    # Verify latest result belongs to checkpoint.
    # --------------------------------------------------------

    latest_checkpoint = latest.get(
        "checkpoint"
    )

    if latest_checkpoint is not None:

        try:

            latest_checkpoint = int(
                latest_checkpoint
            )

        except Exception:

            latest_checkpoint = None

    if (

        latest_checkpoint is not None

        and
        latest_checkpoint != checkpoint

    ):

        log(
            "WARNING: ADWIN latest result "
            f"belongs to checkpoint "
            f"{latest_checkpoint:,}, "
            f"not {checkpoint:,}."
        )

        return []


    # --------------------------------------------------------
    # If latest result explicitly says no drift,
    # trust it.
    # --------------------------------------------------------

    if latest:

        latest_drift = latest.get(
            "drift_detected"
        )

        if str(
            latest_drift
        ).lower() in [

            "false",

            "0",

            "no"

        ]:

            return []


    # --------------------------------------------------------
    # Load CSV.
    # --------------------------------------------------------

    df = load_adwin_results()

    if df.empty:

        return []


    # --------------------------------------------------------
    # Normalize record index.
    # --------------------------------------------------------

    df["record_index"] = pd.to_numeric(

        df["record_index"],

        errors="coerce"

    )

    df = df.dropna(
        subset=[
            "record_index"
        ]
    )

    df["record_index"] = (
        df["record_index"]
        .astype(int)
    )


    # --------------------------------------------------------
    # Current checkpoint only.
    # --------------------------------------------------------

    df = df[
        df["record_index"]
        < checkpoint
    ]


    # --------------------------------------------------------
    # Drift rows.
    # --------------------------------------------------------

    mask = (

        df["drift_detected"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )

    )

    drift_df = df[
        mask
    ].copy()

    if drift_df.empty:

        return []


    # --------------------------------------------------------
    # Convert to dashboard-friendly records.
    # --------------------------------------------------------

    events = []

    detection_time = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    for _, row in drift_df.iterrows():

        event = {

            "record_index":
                int(
                    row["record_index"]
                ),

            "final_score":
                float(
                    row["final_score"]
                )
                if "final_score" in row
                and pd.notna(
                    row["final_score"]
                )
                else None,

            "adwin_width":
                float(
                    row["adwin_width"]
                )
                if "adwin_width" in row
                and pd.notna(
                    row["adwin_width"]
                )
                else None,

            "adwin_estimation":
                float(
                    row["adwin_estimation"]
                )
                if "adwin_estimation" in row
                and pd.notna(
                    row["adwin_estimation"]
                )
                else None,

            "detected_at":
                detection_time

        }

        events.append(
            event
        )

    return events


# ============================================================
# FILTER ONLY NEW DRIFT
# ============================================================

def find_new_drift_events(

    state,

    checkpoint

):

    events = get_current_drift_events(
        checkpoint
    )

    if not events:
        return []

    known = set()

    for value in state.get(
        "known_drift_indices",
        []
    ):

        try:

            known.add(
                int(value)
            )

        except Exception:
            pass

    new_events = []

    for event in events:

        index = int(
            event["record_index"]
        )

        if index not in known:

            new_events.append(
                event
            )

    return new_events


# ============================================================
# REGISTER DRIFT EVENTS
# ============================================================

def register_drift_events(

    state,

    events

):

    known = set()

    for value in state.get(
        "known_drift_indices",
        []
    ):

        try:

            known.add(
                int(value)
            )

        except Exception:
            pass

    for event in events:

        try:

            known.add(
                int(
                    event[
                        "record_index"
                    ]
                )
            )

        except Exception:
            pass

    # Keep a reasonable history.
    state[
        "known_drift_indices"
    ] = sorted(
        known
    )[-5000:]

    state[
        "last_drift_indices"
    ] = [

        int(
            event[
                "record_index"
            ]
        )

        for event in events

    ]

    state[
        "last_drift_count"
    ] = len(
        events
    )


# ============================================================
# READ MODEL REGISTRY
# ============================================================

def read_model_registry():

    return safe_read_json(
        MODEL_REGISTRY_FILE
    )


# ============================================================
# FINAL LIFECYCLE RESULT
# ============================================================

def get_final_lifecycle_result():

    registry = read_model_registry()

    decision = str(
        registry.get(
            "last_decision",
            ""
        )
    ).upper()

    if decision == "PROMOTED":
        return "PROMOTED"

    if decision == "REJECTED":
        return "REJECTED"

    if decision == "ROLLBACK":
        return "ROLLBACK"

    if decision:
        return decision

    return "UNKNOWN"


# ============================================================
# RETRAINING PIPELINE
# ============================================================

def run_retraining_pipeline(

    checkpoint,

    drift_events

):

    if os.path.exists(
        RETRAINING_LOCK_FILE
    ):

        log(
            "WARNING: Retraining lock exists."
        )

        log(
            "Another retraining process may "
            "already be running."
        )

        return False


    # --------------------------------------------------------
    # CREATE LOCK
    # --------------------------------------------------------

    with open(
        RETRAINING_LOCK_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            str(time.time())
        )


    training_times = {}


    try:

        separator()

        log(
            "AUTOMATIC RETRAINING TRIGGERED"
        )

        separator()

        log(
            f"Checkpoint       : "
            f"{checkpoint:,}"
        )

        log(
            f"New drift events : "
            f"{len(drift_events)}"
        )


        # ----------------------------------------------------
        # DRIFT DETAILS
        # ----------------------------------------------------

        for event in drift_events:

            log(
                "  Drift record index : "
                f"{event.get('record_index')}"
            )

            log(
                "  Final score        : "
                f"{event.get('final_score')}"
            )

            log(
                "  Detected at        : "
                f"{event.get('detected_at')}"
            )


        # ----------------------------------------------------
        # DASHBOARD: DRIFT DETECTED
        # ----------------------------------------------------

        save_status_and_history(

            status="RETRAINING",

            stage="DRIFT DETECTED",

            checkpoint=checkpoint,

            drift_count=len(
                drift_events
            ),

            retraining=True,

            evaluation="WAITING",

            lifecycle="WAITING",

            final_result="PENDING",

            message=(
                "ADWIN detected drift. "
                "Automatic PySpark retraining "
                "will proceed."
            ),

            training_times=training_times,

            drift_events=drift_events,

            add_history=True

        )


        # ----------------------------------------------------
        # STEP 1: PYSPARK
        # ----------------------------------------------------

        save_pipeline_status(

            status="RETRAINING",

            stage="PYSPARK RETRAINING",

            checkpoint=checkpoint,

            drift_count=len(
                drift_events
            ),

            retraining=True,

            evaluation="WAITING",

            lifecycle="WAITING",

            final_result="PENDING",

            message=(
                "Drift detected → "
                "PySpark candidate model "
                "training started."
            ),

            training_times=training_times,

            drift_events=drift_events

        )

        log(
            "DRIFT DETECTED → RETRAINING PROCEEDED"
        )


        def pyspark_heartbeat():

            save_pipeline_status(

                status="RETRAINING",

                stage="PYSPARK RETRAINING",

                checkpoint=checkpoint,

                drift_count=len(
                    drift_events
                ),

                retraining=True,

                evaluation="WAITING",

                lifecycle="WAITING",

                final_result="PENDING",

                message=(
                    "PySpark retraining is "
                    "still running."
                ),

                training_times=training_times,

                drift_events=drift_events

            )


        success, elapsed = run_script(

            RETRAIN_SCRIPT,

            "PySpark Retraining",

            status_callback=
                pyspark_heartbeat

        )

        training_times[
            "pyspark_retraining_seconds"
        ] = round(
            elapsed,
            3
        )


        if not success:

            save_pipeline_status(

                status="FAILED",

                stage="PYSPARK RETRAINING FAILED",

                checkpoint=checkpoint,

                drift_count=len(
                    drift_events
                ),

                retraining=False,

                evaluation="NOT_STARTED",

                lifecycle="NOT_STARTED",

                final_result="FAILED",

                message=(
                    "PySpark retraining failed. "
                    "Evaluation and lifecycle "
                    "were not started."
                ),

                training_times=training_times,

                drift_events=drift_events

            )

            return False


        log(
            "PYSPARK RETRAINING COMPLETED"
        )


        # ----------------------------------------------------
        # STEP 2: MODEL EVALUATION
        # ----------------------------------------------------

        save_pipeline_status(

            status="RETRAINING",

            stage="MODEL EVALUATION",

            checkpoint=checkpoint,

            drift_count=len(
                drift_events
            ),

            retraining=True,

            evaluation="RUNNING",

            lifecycle="WAITING",

            final_result="PENDING",

            message=(
                "PySpark training completed. "
                "Candidate model evaluation started."
            ),

            training_times=training_times,

            drift_events=drift_events

        )

        log(
            "MODEL EVALUATION STARTED"
        )


        def evaluation_heartbeat():

            save_pipeline_status(

                status="RETRAINING",

                stage="MODEL EVALUATION",

                checkpoint=checkpoint,

                drift_count=len(
                    drift_events
                ),

                retraining=True,

                evaluation="RUNNING",

                lifecycle="WAITING",

                final_result="PENDING",

                message=(
                    "Candidate model evaluation "
                    "is running."
                ),

                training_times=training_times,

                drift_events=drift_events

            )


        success, elapsed = run_script(

            EVALUATION_SCRIPT,

            "Model Evaluation",

            status_callback=
                evaluation_heartbeat

        )

        training_times[
            "model_evaluation_seconds"
        ] = round(
            elapsed,
            3
        )


        if not success:

            save_pipeline_status(

                status="FAILED",

                stage="MODEL EVALUATION FAILED",

                checkpoint=checkpoint,

                drift_count=len(
                    drift_events
                ),

                retraining=False,

                evaluation="FAILED",

                lifecycle="NOT_STARTED",

                final_result="FAILED",

                message=(
                    "Candidate model evaluation "
                    "failed."
                ),

                training_times=training_times,

                drift_events=drift_events

            )

            return False


        log(
            "MODEL EVALUATION COMPLETED"
        )


        # ----------------------------------------------------
        # STEP 3: MODEL LIFECYCLE
        # ----------------------------------------------------

        save_pipeline_status(

            status="RETRAINING",

            stage="MODEL LIFECYCLE",

            checkpoint=checkpoint,

            drift_count=len(
                drift_events
            ),

            retraining=True,

            evaluation="COMPLETED",

            lifecycle="RUNNING",

            final_result="PENDING",

            message=(
                "Candidate evaluated. "
                "Promotion/rejection decision "
                "is running."
            ),

            training_times=training_times,

            drift_events=drift_events

        )

        log(
            "MODEL LIFECYCLE STARTED"
        )


        success, elapsed = run_script(

            LIFECYCLE_SCRIPT,

            "Model Lifecycle"

        )

        training_times[
            "model_lifecycle_seconds"
        ] = round(
            elapsed,
            3
        )


        if not success:

            save_pipeline_status(

                status="FAILED",

                stage="MODEL LIFECYCLE FAILED",

                checkpoint=checkpoint,

                drift_count=len(
                    drift_events
                ),

                retraining=False,

                evaluation="COMPLETED",

                lifecycle="FAILED",

                final_result="FAILED",

                message=(
                    "Model lifecycle "
                    "decision failed."
                ),

                training_times=training_times,

                drift_events=drift_events

            )

            return False


        log(
            "MODEL LIFECYCLE COMPLETED"
        )


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        final_result = (
            get_final_lifecycle_result()
        )

        registry = (
            read_model_registry()
        )

        current_version = (
            registry.get(
                "current_production_version"
            )
        )

        candidate_version = (
            registry.get(
                "latest_candidate_version"
            )
        )


        # ----------------------------------------------------
        # FINAL DASHBOARD STATUS
        # ----------------------------------------------------

        save_status_and_history(

            status="COMPLETED",

            stage="PIPELINE COMPLETE",

            checkpoint=checkpoint,

            drift_count=len(
                drift_events
            ),

            retraining=False,

            evaluation="COMPLETED",

            lifecycle=final_result,

            final_result=final_result,

            message=(
                "Drift detected → "
                "PySpark retraining → "
                "Evaluation → "
                "Lifecycle completed."
            ),

            training_times=training_times,

            drift_events=drift_events,

            extra={

                "current_production_version":
                    current_version,

                "latest_candidate_version":
                    candidate_version

            },

            add_history=True

        )


        # ----------------------------------------------------
        # TERMINAL SUMMARY
        # ----------------------------------------------------

        separator()

        log(
            "AUTOMATIC MODEL PIPELINE COMPLETE"
        )

        separator()

        log(
            "ADWIN"
            " → PySpark"
            " → Evaluation"
            " → Lifecycle"
        )

        log(
            f"Final result           : "
            f"{final_result}"
        )

        log(
            f"Current production    : "
            f"v{current_version}"
        )

        log(
            f"Latest candidate      : "
            f"v{candidate_version}"
        )

        log(
            "Execution / training times:"
        )

        for name, value in (
            training_times.items()
        ):

            log(
                f"  {name:<32}: "
                f"{value:.3f} seconds"
            )

        separator()

        return True


    finally:

        # ----------------------------------------------------
        # ALWAYS REMOVE LOCK
        # ----------------------------------------------------

        if os.path.exists(
            RETRAINING_LOCK_FILE
        ):

            try:

                os.remove(
                    RETRAINING_LOCK_FILE
                )

            except Exception:
                pass


# ============================================================
# PROCESS ONE CHECKPOINT
# ============================================================

def process_checkpoint(

    checkpoint,

    state

):

    separator()

    log(
        f"NEW CHECKPOINT REACHED: "
        f"{checkpoint:,} RECORDS"
    )

    separator()


    # --------------------------------------------------------
    # DASHBOARD: ADWIN START
    # --------------------------------------------------------

    save_pipeline_status(

        status="RUNNING",

        stage="ADWIN DRIFT DETECTION",

        checkpoint=checkpoint,

        drift_count=0,

        retraining=False,

        evaluation="NOT_STARTED",

        lifecycle="NOT_STARTED",

        final_result="NONE",

        message=(
            f"Running ADWIN at "
            f"{checkpoint:,}-record checkpoint."
        )

    )


    # --------------------------------------------------------
    # Remove stale result files.
    # --------------------------------------------------------

    clear_old_adwin_results()


    # --------------------------------------------------------
    # Run ADWIN.
    # --------------------------------------------------------

    adwin_start = time.time()

    success, elapsed = run_script(

        ADWIN_SCRIPT,

        "ADWIN Drift Detection",

        extra_env={

            "ORCHESTRATOR_CHECKPOINT":
                str(checkpoint)

        }

    )


    if not success:

        save_pipeline_status(

            status="FAILED",

            stage="ADWIN FAILED",

            checkpoint=checkpoint,

            drift_count=0,

            retraining=False,

            evaluation="NOT_STARTED",

            lifecycle="NOT_STARTED",

            final_result="FAILED",

            message=(
                "ADWIN drift detection failed."
            ),

            training_times={

                "adwin_seconds":
                    round(
                        elapsed,
                        3
                    )

            }

        )

        return False


    # --------------------------------------------------------
    # Read ADWIN result.
    # --------------------------------------------------------

    latest = (
        load_adwin_latest_result()
    )

    drift_events = (
        find_new_drift_events(
            state,
            checkpoint
        )
    )


    # --------------------------------------------------------
    # Register all current drift events.
    # --------------------------------------------------------

    all_current_events = (
        get_current_drift_events(
            checkpoint
        )
    )

    register_drift_events(
        state,
        all_current_events
    )

    state[
        "last_adwin_record_count"
    ] = checkpoint

    state[
        "total_checkpoints_processed"
    ] = (

        int(
            state.get(
                "total_checkpoints_processed",
                0
            )
        )

        + 1

    )


    # --------------------------------------------------------
    # ADWIN SUMMARY.
    # --------------------------------------------------------

    actual_drift_count = len(
        all_current_events
    )

    latest_drift = (
        latest.get(
            "drift_detected"
        )
        if latest
        else False
    )

    log(
        "ADWIN checkpoint result:"
    )

    log(
        f"  Checkpoint       : "
        f"{checkpoint:,}"
    )

    log(
        f"  Records analyzed : "
        f"{latest.get('records_analyzed', checkpoint):,}"
    )

    log(
        f"  Drift events     : "
        f"{latest.get('drift_events', actual_drift_count)}"
    )

    log(
        f"  Drift detected   : "
        f"{latest_drift}"
    )


    # --------------------------------------------------------
    # NO NEW DRIFT
    # --------------------------------------------------------

    if not drift_events:

        log(
            "No NEW drift detected."
        )

        log(
            "Retraining will NOT be triggered."
        )

        save_status_and_history(

            status="RUNNING",

            stage="NO DRIFT",

            checkpoint=checkpoint,

            drift_count=actual_drift_count,

            retraining=False,

            evaluation="NOT_STARTED",

            lifecycle="NOT_STARTED",

            final_result="NO RETRAINING",

            message=(
                "ADWIN completed. "
                "No new drift detected. "
                "Retraining not required."
            ),

            training_times={

                "adwin_seconds":
                    round(
                        elapsed,
                        3
                    )

            },

            drift_events=all_current_events,

            add_history=True

        )

        state[
            "last_processed_checkpoint"
        ] = checkpoint

        state[
            "last_run_timestamp"
        ] = time.time()

        save_state(
            state
        )

        return True


    # --------------------------------------------------------
    # NEW DRIFT FOUND
    # --------------------------------------------------------

    log(
        f"NEW DRIFT EVENTS: "
        f"{len(drift_events)}"
    )

    log(
        "Triggering automatic PySpark retraining..."
    )


    # --------------------------------------------------------
    # Run automatic pipeline.
    # --------------------------------------------------------

    success = run_retraining_pipeline(

        checkpoint,

        drift_events

    )


    # --------------------------------------------------------
    # Update state.
    # --------------------------------------------------------

    if success:

        state[
            "total_retraining_triggers"
        ] = (

            int(
                state.get(
                    "total_retraining_triggers",
                    0
                )
            )

            + 1

        )

        state[
            "last_trigger_record_count"
        ] = checkpoint

        state[
            "last_trigger_reason"
        ] = (

            f"ADWIN detected "
            f"{len(drift_events)} "
            f"new drift event(s)"

        )

    else:

        log(
            "Automatic retraining pipeline failed."
        )


    state[
        "last_processed_checkpoint"
    ] = checkpoint

    state[
        "last_run_timestamp"
    ] = time.time()

    save_state(
        state
    )

    return success


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():

    separator()

    print(
        "AUTOMATIC DRIFT / RETRAINING ORCHESTRATOR",
        flush=True
    )

    separator()

    print()

    log(
        f"History directory     : "
        f"{HISTORY_DIR}"
    )

    log(
        f"Checkpoint size       : "
        f"{CHECKPOINT_SIZE:,}"
    )

    log(
        f"Check interval        : "
        f"{CHECK_INTERVAL_SECONDS}s"
    )

    log(
        "Pipeline:"
    )

    log(
        "ADWIN → PySpark → Evaluation → Lifecycle"
    )

    print()


    # --------------------------------------------------------
    # Initial dashboard state.
    # --------------------------------------------------------

    save_pipeline_status(

        status="RUNNING",

        stage="MONITORING",

        checkpoint=0,

        drift_count=0,

        retraining=False,

        evaluation="NOT_STARTED",

        lifecycle="NOT_STARTED",

        final_result="NONE",

        message=(
            "Automatic orchestrator is monitoring "
            "Parquet history for 10,000-record checkpoints."
        )

    )


    # --------------------------------------------------------
    # Load state.
    # --------------------------------------------------------

    state = load_state()

    log(
        "State loaded."
    )

    log(
        "Last processed checkpoint: "
        f"{state.get('last_processed_checkpoint', 0):,}"
    )

    log(
        "Total automatic retrainings: "
        f"{state.get('total_retraining_triggers', 0)}"
    )

    print()

    separator()

    log(
        "ORCHESTRATOR RUNNING"
    )

    separator()


    # ========================================================
    # CONTINUOUS MONITORING
    # ========================================================

    while True:

        try:

            record_count = (
                count_history_records()
            )

            completed_checkpoint = (
                get_completed_checkpoint(
                    record_count
                )
            )

            last_checkpoint = int(
                state.get(
                    "last_processed_checkpoint",
                    0
                )
            )


            print()

            log(
                f"History records        : "
                f"{record_count:,}"
            )

            log(
                f"Completed checkpoint   : "
                f"{completed_checkpoint:,}"
            )

            log(
                f"Last processed         : "
                f"{last_checkpoint:,}"
            )


            # ------------------------------------------------
            # Process every missed checkpoint.
            # ------------------------------------------------

            if (
                completed_checkpoint
                > last_checkpoint
            ):

                next_checkpoint = (

                    last_checkpoint
                    + CHECKPOINT_SIZE

                )

                while (

                    next_checkpoint
                    <= completed_checkpoint

                ):

                    process_checkpoint(

                        next_checkpoint,

                        state

                    )

                    next_checkpoint += (
                        CHECKPOINT_SIZE
                    )


            else:

                log(
                    "No new 10,000-record checkpoint."
                )

                save_pipeline_status(

                    status="RUNNING",

                    stage="MONITORING",

                    checkpoint=last_checkpoint,

                    drift_count=0,

                    retraining=False,

                    evaluation="NOT_STARTED",

                    lifecycle="NOT_STARTED",

                    final_result="NONE",

                    message=(
                        "Waiting for the next "
                        "10,000-record checkpoint."
                    )

                )


            time.sleep(
                CHECK_INTERVAL_SECONDS
            )


        # ----------------------------------------------------
        # CTRL+C
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print()

            separator()

            log(
                "ORCHESTRATOR STOPPED BY USER."
            )

            save_pipeline_status(

                status="STOPPED",

                stage="STOPPED",

                checkpoint=state.get(
                    "last_processed_checkpoint",
                    0
                ),

                drift_count=0,

                retraining=False,

                evaluation="NOT_STARTED",

                lifecycle="NOT_STARTED",

                final_result="STOPPED",

                message=(
                    "Automatic orchestrator "
                    "stopped manually."
                )

            )

            separator()

            break


        # ----------------------------------------------------
        # CONTINUE AFTER ERROR
        # ----------------------------------------------------

        except Exception as error:

            print()

            log(
                f"ERROR in orchestrator: "
                f"{error}"
            )

            log(
                "System will continue monitoring."
            )

            time.sleep(
                CHECK_INTERVAL_SECONDS
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
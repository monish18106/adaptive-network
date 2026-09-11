# ============================================================
# DASHBOARD DATA MODULE
# REAL-TIME NETWORK INTRUSION DETECTION
# ============================================================
#
# Single normalized data layer for the dashboard.
#
# Pipeline:
# Kafka
#   -> Flink
#   -> Inference
#   -> Fusion
#   -> Alert
#   -> Parquet History
#   -> ADWIN
#   -> PySpark Retraining
#   -> Evaluation
#   -> Model Lifecycle
#
# IMPORTANT:
# - Historical ADWIN drift count is kept separate from
#   current-checkpoint NEW drift count.
# - Latest drift is taken from drift history, not only from
#   adwin_latest_result.json.
# - PySpark retraining metadata is preferred.
# - Old Phase-11 report is only a fallback.
# - Model registry is the source of truth for production/
#   candidate/lifecycle state.
#
# ============================================================


import os
import glob
import json
import math
import time

import pandas as pd


# ============================================================
# BASE PATHS
# ============================================================

# dashboard_data.py is expected to be inside the dashboard
# directory.
#
# Project:
# D:\adaptive network\
#
# dashboard:
# D:\adaptive network\dashboard\
#
# Therefore:
# dirname(dirname(__file__))
#       -> project root
#
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


SAVED_MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)


PRODUCTION_DIR = os.path.join(
    SAVED_MODELS_DIR,
    "production"
)


CANDIDATES_DIR = os.path.join(
    SAVED_MODELS_DIR,
    "candidates"
)


ARCHIVE_DIR = os.path.join(
    SAVED_MODELS_DIR,
    "archive"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

REGISTRY_PATH = os.path.join(
    SAVED_MODELS_DIR,
    "model_registry.json"
)


# ============================================================
# PROCESSED DATA FILES
# ============================================================

PHASE12_COMPARISON = os.path.join(
    PROCESSED_DIR,
    "phase12_model_comparison.csv"
)


PHASE12_PROMOTION = os.path.join(
    PROCESSED_DIR,
    "phase12_promotion_report.csv"
)


PHASE12_METADATA = os.path.join(
    PROCESSED_DIR,
    "phase12_model_metadata.json"
)


# Old Phase-11 report.
RETRAINING_REPORT = os.path.join(
    PROCESSED_DIR,
    "phase11_retraining_report.csv"
)


# Current PySpark retraining report.
PYSPARK_RETRAINING_REPORT = os.path.join(
    PROCESSED_DIR,
    "pyspark_retraining_report.csv"
)


ADWIN_RESULTS = os.path.join(
    PROCESSED_DIR,
    "adwin_drift_results.csv"
)


ADWIN_LATEST_FILE = os.path.join(
    PROCESSED_DIR,
    "adwin_latest_result.json"
)


PIPELINE_STATUS_FILE = os.path.join(
    PROCESSED_DIR,
    "automatic_pipeline_status.json"
)


SELF_CORRECTION_FILE = os.path.join(
    PROCESSED_DIR,
    "self_correction_state.json"
)


AUTOMATIC_STATE_FILE = os.path.join(
    PROCESSED_DIR,
    "automatic_retraining_state.json"
)


PIPELINE_LOG_FILE = os.path.join(
    PROCESSED_DIR,
    "retraining_pipeline.log"
)


RETRAINING_LOCK_FILE = os.path.join(
    PROCESSED_DIR,
    "retraining_in_progress.lock"
)


# ============================================================
# CONSTANTS
# ============================================================

CHECKPOINT_SIZE = 10_000

ANOMALY_LIMIT = 100

DRIFT_HISTORY_LIMIT = 100

PIPELINE_LOG_LIMIT = 150

FEATURE_PREVIEW_LIMIT = 12


# ============================================================
# SAFE FILE HELPERS
# ============================================================

def safe_read_json(path):
    """
    Read JSON safely.

    Returns:
        dict/list depending on JSON content,
        or {} if unavailable.
    """

    if not path:
        return {}

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


def safe_read_csv(path):
    """
    Read CSV safely.
    """

    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception:

        return pd.DataFrame()


def file_exists(path):
    return bool(
        path and os.path.exists(path)
    )


def file_mtime(path):
    """
    Return file modification timestamp.
    """

    if not file_exists(path):
        return None

    try:
        return os.path.getmtime(path)

    except Exception:
        return None


# ============================================================
# VALUE CLEANING
# ============================================================

def clean_value(value):
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:

        pass

    # numpy scalar
    if hasattr(value, "item"):

        try:
            value = value.item()

        except Exception:
            pass

    # arrays / series
    if hasattr(value, "tolist"):

        try:
            value = value.tolist()

        except Exception:
            pass

    if isinstance(
        value,
        float
    ):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

    return value


def clean_record(record):
    """
    Clean every value in a dictionary.
    """

    if not isinstance(
        record,
        dict
    ):

        return record

    return {
        key: clean_value(value)
        for key, value in record.items()
    }


def safe_float(value):
    """
    Convert to float safely.
    """

    if value is None:
        return None

    try:

        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if not value:
                return None

        result = float(value)

        if math.isnan(result):
            return None

        if math.isinf(result):
            return None

        return result

    except Exception:

        return None


def safe_int(value):
    """
    Convert to integer safely.
    """

    if value is None:
        return None

    try:

        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if not value:
                return None

        return int(float(value))

    except Exception:

        return None


def first_value(
    source,
    *keys,
    default=None
):
    """
    Return first existing non-null value.
    """

    if not isinstance(
        source,
        dict
    ):

        return default

    for key in keys:

        if key in source:

            value = source.get(key)

            if value is not None:
                return value

    return default


# ============================================================
# HISTORICAL PARQUET DATA
# ============================================================

def load_history():
    """
    Load all parquet history files.

    Supports:
        data/history/*.parquet
        data/history/**/*.parquet
    """

    if not os.path.exists(
        HISTORY_DIR
    ):

        return pd.DataFrame()

    files = glob.glob(
        os.path.join(
            HISTORY_DIR,
            "**",
            "*.parquet"
        ),
        recursive=True
    )

    if not files:
        return pd.DataFrame()

    frames = []

    for path in sorted(files):

        try:

            frame = pd.read_parquet(
                path
            )

            if not frame.empty:
                frames.append(frame)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    try:

        return pd.concat(
            frames,
            ignore_index=True
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# HISTORY TIMESTAMP SORTING
# ============================================================

def sort_by_timestamp(
    df,
    ascending=False
):
    """
    Sort a dataframe by timestamp if available.
    """

    if df.empty:
        return df

    if "timestamp" not in df.columns:
        return df

    result = df.copy()

    try:

        result["_dashboard_timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce"
        )

        result = result.sort_values(
            "_dashboard_timestamp",
            ascending=ascending,
            na_position="last"
        )

        result = result.drop(
            columns=[
                "_dashboard_timestamp"
            ],
            errors="ignore"
        )

    except Exception:

        pass

    return result


# ============================================================
# SYSTEM SUMMARY
# ============================================================

def get_system_summary():
    """
    Dashboard system counters.

    IMPORTANT:
    These values come from the actual Parquet history,
    not from ADWIN or pipeline state.
    """

    df = load_history()

    total_logs = int(
        len(df)
    )

    normal_count = 0
    anomaly_count = 0

    if not df.empty:

        if "status" in df.columns:

            status = (
                df["status"]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            normal_count = int(
                (
                    status == "NORMAL"
                ).sum()
            )

            anomaly_count = int(
                (
                    status == "ANOMALY"
                ).sum()
            )

        elif "prediction" in df.columns:

            prediction = (
                df["prediction"]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            anomaly_count = int(
                (
                    prediction == "ANOMALY"
                ).sum()
            )

            normal_count = (
                total_logs -
                anomaly_count
            )

    registry = safe_read_json(
        REGISTRY_PATH
    )

    pipeline = safe_read_json(
        PIPELINE_STATUS_FILE
    )

    production_version = first_value(
        registry,
        "current_production_version"
    )

    candidate_version = first_value(
        registry,
        "latest_candidate_version"
    )

    decision = first_value(
        registry,
        "last_decision",
        default="NOT_AVAILABLE"
    )

    return {

        "total_logs":
            total_logs,

        "logs_processed":
            total_logs,

        "normal_logs":
            normal_count,

        "anomalies_detected":
            anomaly_count,

        "current_production_version":
            production_version,

        "latest_candidate_version":
            candidate_version,

        "last_model_decision":
            decision,

        "system_status":
            "RUNNING",

        "pipeline_stage":
            first_value(
                pipeline,
                "stage",
                default="MONITORING"
            ),

        "timestamp":
            time.time()
    }


# ============================================================
# ANOMALIES
# ============================================================

def get_anomalies(
    limit=ANOMALY_LIMIT
):
    """
    Return recent anomaly records.

    These are intended for manual forensic verification.
    """

    df = load_history()

    if df.empty:
        return []

    working = df.copy()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if "status" in working.columns:

        working["status"] = (
            working["status"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        anomalies = working[
            working["status"] == "ANOMALY"
        ].copy()

    elif "prediction" in working.columns:

        working["prediction"] = (
            working["prediction"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        anomalies = working[
            working["prediction"] == "ANOMALY"
        ].copy()

    else:

        return []

    if anomalies.empty:
        return []

    anomalies = sort_by_timestamp(
        anomalies,
        ascending=False
    )

    anomalies = anomalies.head(
        limit
    )

    records = []

    for _, row in anomalies.iterrows():

        features = row.get(
            "features"
        )

        if features is None:

            # Try common alternate field.
            features = row.get(
                "feature_vector",
                []
            )

        if hasattr(
            features,
            "tolist"
        ):

            try:
                features = features.tolist()

            except Exception:
                features = []

        if not isinstance(
            features,
            list
        ):

            try:

                features = list(
                    features
                )

            except Exception:

                features = []

        record = {

            "timestamp":
                clean_value(
                    row.get(
                        "timestamp"
                    )
                ),

            "status":
                "ANOMALY",

            "if_score":
                safe_float(
                    row.get(
                        "if_score"
                    )
                ),

            "ae_score":
                safe_float(
                    row.get(
                        "ae_score"
                    )
                ),

            "final_score":
                safe_float(
                    row.get(
                        "final_score"
                    )
                ),

            "threshold":
                safe_float(
                    row.get(
                        "threshold"
                    )
                ),

            "features":
                [
                    clean_value(x)
                    for x in features[
                        :FEATURE_PREVIEW_LIMIT
                    ]
                ]
        }

        # ----------------------------------------------------
        # FORENSIC LOG FIELDS
        # ----------------------------------------------------

        for column in [

            "source_ip",
            "destination_ip",

            "src_ip",
            "dst_ip",

            "protocol",

            "src_port",
            "dst_port",

            "source_port",
            "destination_port",

            "label",
            "prediction",
            "attack_type",

            "packet_count",
            "bytes",

            "flow_duration",

            "model_version"

        ]:

            if column in row.index:

                record[column] = clean_value(
                    row.get(column)
                )

        records.append(
            clean_record(
                record
            )
        )

    return records


# ============================================================
# ADWIN DRIFT
# ============================================================

def get_drift():
    """
    Return complete ADWIN drift information.

    CRITICAL DISTINCTION:

        total_drifts
            = historical number of drift events

        checkpoint
            = latest checkpoint examined

        new_drift_count
            = new events belonging to current pipeline
              checkpoint

    The dashboard must never use new_drift_count as the
    historical total.
    """

    df = safe_read_csv(
        ADWIN_RESULTS
    )

    latest_result = safe_read_json(
        ADWIN_LATEST_FILE
    )

    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if df.empty:

        return {

            "available":
                False,

            "total_drifts":
                0,

            "new_drift_count":
                0,

            "latest_drift":
                None,

            "latest_result":
                latest_result,

            "checkpoint":
                first_value(
                    latest_result,
                    "checkpoint",
                    "records_analyzed"
                ),

            "records_analyzed":
                first_value(
                    latest_result,
                    "records_analyzed"
                ),

            "records":
                []
        }

    working = df.copy()

    # --------------------------------------------------------
    # FIND DRIFT COLUMN
    # --------------------------------------------------------

    drift_column = None

    possible_columns = [

        "drift_detected",

        "drift",

        "is_drift",

        "change_detected"

    ]

    for column in possible_columns:

        if column in working.columns:

            drift_column = column
            break

    if drift_column is None:

        for column in working.columns:

            if "drift" in column.lower():

                drift_column = column
                break

    if drift_column is None:

        return {

            "available":
                True,

            "total_drifts":
                0,

            "new_drift_count":
                0,

            "latest_drift":
                None,

            "latest_result":
                latest_result,

            "checkpoint":
                first_value(
                    latest_result,
                    "checkpoint"
                ),

            "records_analyzed":
                first_value(
                    latest_result,
                    "records_analyzed"
                ),

            "records":
                []
        }

    values = (
        working[drift_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    drift_mask = values.isin(
        [
            "true",
            "1",
            "yes",
            "detected"
        ]
    )

    drift_df = working[
        drift_mask
    ].copy()

    # --------------------------------------------------------
    # RECORD INDEX
    # --------------------------------------------------------

    if "record_index" in drift_df.columns:

        drift_df["record_index"] = pd.to_numeric(
            drift_df["record_index"],
            errors="coerce"
        )

        drift_df = drift_df.dropna(
            subset=[
                "record_index"
            ]
        )

        drift_df["record_index"] = (
            drift_df["record_index"]
            .astype(int)
        )

        drift_df = drift_df.sort_values(
            "record_index"
        )

    else:

        drift_df = sort_by_timestamp(
            drift_df,
            ascending=True
        )

    # --------------------------------------------------------
    # ALL HISTORICAL DRIFTS
    # --------------------------------------------------------

    total_drifts = int(
        len(drift_df)
    )

    # --------------------------------------------------------
    # LAST 100 EVENTS
    # --------------------------------------------------------

    records = (
        drift_df
        .tail(
            DRIFT_HISTORY_LIMIT
        )
        .to_dict(
            orient="records"
        )
    )

    records = [
        clean_record(record)
        for record in records
    ]

    # --------------------------------------------------------
    # LATEST HISTORICAL DRIFT
    # --------------------------------------------------------

    latest_drift = (
        records[-1]
        if records
        else None
    )

    # --------------------------------------------------------
    # LATEST RESULT / CHECKPOINT
    # --------------------------------------------------------

    checkpoint = first_value(
        latest_result,
        "checkpoint",
        "records_analyzed",
        "record_count"
    )

    records_analyzed = first_value(
        latest_result,
        "records_analyzed",
        "record_count",
        "checkpoint"
    )

    # If latest result does not contain checkpoint,
    # infer it from the highest drift record.
    if checkpoint is None and latest_drift:

        checkpoint = latest_drift.get(
            "checkpoint"
        )

        if checkpoint is None:

            checkpoint = latest_drift.get(
                "record_index"
            )

    # --------------------------------------------------------
    # NEW DRIFT COUNT
    #
    # The ADWIN CSV may contain events across multiple
    # checkpoints. The latest pipeline status is authoritative
    # for the current new-drift count.
    # --------------------------------------------------------

    pipeline = safe_read_json(
        PIPELINE_STATUS_FILE
    )

    new_drift_count = first_value(
        pipeline,
        "new_drift_count",
        "new_drift_events"
    )

    if new_drift_count is None:

        # Older orchestrator writes drift_count.
        # This is only considered current/new count here.
        new_drift_count = first_value(
            pipeline,
            "drift_count"
        )

    new_drift_count = (
        safe_int(
            new_drift_count
        )
        if new_drift_count is not None
        else 0
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "available":
            True,

        "total_drifts":
            total_drifts,

        "new_drift_count":
            new_drift_count,

        "latest_drift":
            latest_drift,

        "latest_result":
            clean_record(
                latest_result
            )
            if isinstance(
                latest_result,
                dict
            )
            else latest_result,

        "checkpoint":
            safe_int(
                checkpoint
            ),

        "records_analyzed":
            safe_int(
                records_analyzed
            ),

        "records":
            records
    }


# ============================================================
# RETRAINING REPORT SELECTION
# ============================================================

def choose_retraining_report():
    """
    Choose the most appropriate retraining report.

    Priority:
        1. pyspark_retraining_report.csv
        2. phase11_retraining_report.csv

    The current PySpark report is preferred because the
    dashboard is intended to represent the automatic
    self-correction pipeline.
    """

    if file_exists(
        PYSPARK_RETRAINING_REPORT
    ):

        return (
            PYSPARK_RETRAINING_REPORT,
            "pyspark_retraining_report.csv"
        )

    if file_exists(
        RETRAINING_REPORT
    ):

        return (
            RETRAINING_REPORT,
            "phase11_retraining_report.csv"
        )

    return (
        None,
        None
    )


# ============================================================
# RETRAINING METADATA FILE SEARCH
# ============================================================

def find_candidate_metadata():
    """
    Find candidate_training_metadata.json or similar
    metadata files.

    This is a fallback for projects where the CSV does not
    contain all metadata.
    """

    candidates = []

    search_roots = [

        CANDIDATES_DIR,

        PROCESSED_DIR,

        SAVED_MODELS_DIR

    ]

    filenames = [

        "candidate_training_metadata.json",

        "training_metadata.json",

        "model_metadata.json"

    ]

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for filename in filenames:

            pattern = os.path.join(
                root,
                "**",
                filename
            )

            candidates.extend(
                glob.glob(
                    pattern,
                    recursive=True
                )
            )

    if not candidates:
        return None

    # Newest metadata file first.
    candidates.sort(
        key=lambda p: (
            file_mtime(p) or 0
        ),
        reverse=True
    )

    return candidates[0]


# ============================================================
# RETRAINING DATA
# ============================================================

def get_retraining():
    """
    Current retraining information.

    This returns:
        current status
        latest training records
        model-specific training rows
        source report
    """

    report_path, report_source = (
        choose_retraining_report()
    )

    df = (
        safe_read_csv(
            report_path
        )
        if report_path
        else pd.DataFrame()
    )

    pipeline = safe_read_json(
        PIPELINE_STATUS_FILE
    )

    lock_active = file_exists(
        RETRAINING_LOCK_FILE
    )

    # --------------------------------------------------------
    # REPORT RECORDS
    # --------------------------------------------------------

    records = []

    if not df.empty:

        records = (
            df
            .to_dict(
                orient="records"
            )
        )

        records = [
            clean_record(record)
            for record in records
        ]

    # --------------------------------------------------------
    # PIPELINE CURRENT STATUS
    # --------------------------------------------------------

    pipeline_stage = str(
        first_value(
            pipeline,
            "stage",
            default=""
        )
    ).upper()

    pipeline_status = str(
        first_value(
            pipeline,
            "status",
            default=""
        )
    ).upper()

    retraining_flag = first_value(
        pipeline,
        "retraining"
    )

    # --------------------------------------------------------
    # DETERMINE CURRENT RETRAINING STATUS
    # --------------------------------------------------------

    if lock_active:

        current_status = "ONGOING"

    elif (
        "PYSPARK" in pipeline_stage
        and "FAILED" not in pipeline_stage
    ):

        if pipeline_status in [
            "RETRAINING",
            "RUNNING",
            "IN_PROGRESS"
        ]:

            current_status = "ONGOING"

        else:

            current_status = "COMPLETED"

    elif pipeline_stage in [
        "DRIFT DETECTED"
    ]:

        current_status = "TRIGGERED"

    elif pipeline_stage in [
        "NO DRIFT",
        "WAITING",
        "MONITORING"
    ]:

        current_status = "NOT_TRIGGERED"

    elif "FAILED" in pipeline_stage:

        current_status = "FAILED"

    elif pipeline_status in [
        "RETRAINING",
        "RUNNING",
        "IN_PROGRESS"
    ]:

        current_status = "ONGOING"

    elif retraining_flag is True:

        current_status = "ONGOING"

    elif records:

        current_status = "COMPLETED"

    else:

        current_status = "NOT_TRIGGERED"

    # --------------------------------------------------------
    # TRAINING RECORD COUNT
    # --------------------------------------------------------

    training_records = None
    feature_count = None

    for record in records:

        value = first_value(
            record,
            "training_records",
            "records"
        )

        if value is not None:

            training_records = safe_int(
                value
            )

            if training_records is not None:
                break

    for record in records:

        value = first_value(
            record,
            "features",
            "feature_count"
        )

        if value is not None:

            feature_count = safe_int(
                value
            )

            if feature_count is not None:
                break

    # --------------------------------------------------------
    # NEW DRIFT COUNT
    # --------------------------------------------------------

    drift_count = first_value(
        pipeline,
        "new_drift_count",
        "new_drift_events",
        "drift_count"
    )

    drift_count = (
        safe_int(
            drift_count
        )
        if drift_count is not None
        else 0
    )

    # --------------------------------------------------------
    # TRAINING TIMES
    # --------------------------------------------------------

    training_times = first_value(
        pipeline,
        "training_times",
        default={}
    )

    if not isinstance(
        training_times,
        dict
    ):

        training_times = {}

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "available":
            bool(records),

        "source":
            report_source,

        "source_path":
            report_path,

        "current_status":
            current_status,

        "pipeline_stage":
            pipeline_stage,

        "pipeline_status":
            pipeline_status,

        "training_records":
            training_records,

        "features":
            feature_count,

        "drift_count":
            drift_count,

        "training_times":
            clean_record(
                training_times
            ),

        "records":
            records,

        "latest":
            records[-1]
            if records
            else None
    }


# ============================================================
# RETRAINING HISTORY
# ============================================================

def get_retraining_history():
    """
    Return complete model training history from the selected
    retraining report.
    """

    retraining = get_retraining()

    records = retraining.get(
        "records",
        []
    )

    return {

        "available":
            bool(records),

        "source":
            retraining.get(
                "source"
            ),

        "records":
            records,

        "count":
            len(records),

        "latest":
            records[-1]
            if records
            else None
    }


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_models():
    """
    Return normalized model registry information.
    """

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    models = registry.get(
        "models",
        []
    )

    if not isinstance(
        models,
        list
    ):

        models = []

    cleaned_models = []

    for model in models:

        if not isinstance(
            model,
            dict
        ):

            continue

        model_copy = dict(
            model
        )

        metrics = model_copy.get(
            "metrics",
            {}
        )

        if not isinstance(
            metrics,
            dict
        ):

            metrics = {}

        model_copy["metrics"] = {
            key:
                clean_value(value)
            for key, value
            in metrics.items()
        }

        if "metric_changes" in model_copy:

            changes = model_copy.get(
                "metric_changes"
            )

            if not isinstance(
                changes,
                dict
            ):

                changes = {}

            model_copy[
                "metric_changes"
            ] = {
                key:
                    clean_value(value)
                for key, value
                in changes.items()
            }

        # Normalize common metric names.
        model_copy[
            "f1"
        ] = safe_float(
            first_value(
                model_copy["metrics"],
                "f1",
                "F1"
            )
        )

        model_copy[
            "accuracy"
        ] = safe_float(
            first_value(
                model_copy["metrics"],
                "accuracy",
                "Accuracy"
            )
        )

        model_copy[
            "precision"
        ] = safe_float(
            first_value(
                model_copy["metrics"],
                "precision",
                "Precision"
            )
        )

        model_copy[
            "detection_rate"
        ] = safe_float(
            first_value(
                model_copy["metrics"],
                "detection_rate",
                "detection"
            )
        )

        model_copy[
            "false_alarm_rate"
        ] = safe_float(
            first_value(
                model_copy["metrics"],
                "false_alarm_rate",
                "far"
            )
        )

        cleaned_models.append(
            clean_record(
                model_copy
            )
        )

    # Newest version first.
    cleaned_models.sort(
        key=lambda model: (
            safe_int(
                model.get(
                    "version"
                )
            )
            or -1
        ),
        reverse=True
    )

    return {

        "current_production_version":
            registry.get(
                "current_production_version"
            ),

        "latest_candidate_version":
            registry.get(
                "latest_candidate_version"
            ),

        "previous_production_version":
            registry.get(
                "previous_production_version"
            ),

        "last_decision":
            registry.get(
                "last_decision"
            ),

        "last_updated":
            registry.get(
                "last_updated"
            ),

        "models":
            cleaned_models
    }


# ============================================================
# TOP PERFORMING MODEL
# ============================================================

def get_top_model():
    """
    Production model is the top model for the dashboard
    because it is the model currently serving production.

    If the registry contains explicit ACTIVE model metrics,
    those are used.
    """

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    models = registry.get(
        "models",
        []
    )

    if not isinstance(
        models,
        list
    ):

        models = []

    production_version = registry.get(
        "current_production_version"
    )

    production_model = None

    # --------------------------------------------------------
    # Find ACTIVE production version.
    # --------------------------------------------------------

    for model in models:

        if not isinstance(
            model,
            dict
        ):
            continue

        version = model.get(
            "version"
        )

        status = str(
            model.get(
                "status",
                ""
            )
        ).upper()

        if (
            production_version is not None
            and str(version)
            == str(production_version)
        ):

            production_model = model
            break

        if status == "ACTIVE":

            production_model = model
            break

    if production_model is None:

        return {

            "available":
                False,

            "model":
                "UNKNOWN",

            "version":
                production_version,

            "status":
                "UNKNOWN",

            "f1":
                None,

            "accuracy":
                None,

            "precision":
                None,

            "detection_rate":
                None,

            "false_alarm_rate":
                None,

            "training_time_seconds":
                None,

            "reason":
                registry.get(
                    "last_decision"
                )
        }

    metrics = production_model.get(
        "metrics",
        {}
    )

    if not isinstance(
        metrics,
        dict
    ):

        metrics = {}

    metadata = {}

    version = production_model.get(
        "version"
    )

    # --------------------------------------------------------
    # Production metadata
    # --------------------------------------------------------

    if version is not None:

        metadata_path = os.path.join(

            PRODUCTION_DIR,

            f"v{version}",

            "model_metadata.json"

        )

        metadata = safe_read_json(
            metadata_path
        )

    return {

        "available":
            True,

        "model":
            first_value(
                production_model,
                "model",
                "model_name",
                "name",
                default="Isolation Forest + Autoencoder"
            ),

        "model_name":
            first_value(
                production_model,
                "model_name",
                "model",
                "name",
                default="Isolation Forest + Autoencoder"
            ),

        "version":
            version,

        "status":
            production_model.get(
                "status",
                "ACTIVE"
            ),

        "f1":
            safe_float(
                first_value(
                    metrics,
                    "f1",
                    "F1"
                )
            ),

        "accuracy":
            safe_float(
                first_value(
                    metrics,
                    "accuracy",
                    "Accuracy"
                )
            ),

        "precision":
            safe_float(
                first_value(
                    metrics,
                    "precision",
                    "Precision"
                )
            ),

        "detection_rate":
            safe_float(
                first_value(
                    metrics,
                    "detection_rate",
                    "detection"
                )
            ),

        "false_alarm_rate":
            safe_float(
                first_value(
                    metrics,
                    "false_alarm_rate",
                    "far"
                )
            ),

        "training_time_seconds":
            safe_float(
                first_value(
                    production_model,
                    "training_time_seconds"
                )
            ),

        "reason":
            first_value(
                production_model,
                "reason",
                "promotion_reason",
                default=registry.get(
                    "last_decision"
                )
            ),

        "metadata":
            clean_record(
                metadata
            )
            if isinstance(
                metadata,
                dict
            )
            else {}
    }


# ============================================================
# MODEL METADATA
# ============================================================

def get_model_metadata():
    """
    Return production and candidate model training metadata.

    Candidate values are taken from:
        1. PySpark retraining report
        2. candidate metadata JSON
        3. registry/model metadata

    Production values are taken from:
        production/vX/model_metadata.json
    """

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    models = registry.get(
        "models",
        []
    )

    if not isinstance(
        models,
        list
    ):

        models = []

    # --------------------------------------------------------
    # RETRAINING REPORT
    # --------------------------------------------------------

    report_path, report_source = (
        choose_retraining_report()
    )

    retraining_df = (
        safe_read_csv(
            report_path
        )
        if report_path
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # CANDIDATE DEFAULTS
    # --------------------------------------------------------

    candidate_records = None
    candidate_features = None

    candidate_if_time = None
    candidate_if_estimators = None

    candidate_ae_time = None
    candidate_ae_epochs = None
    candidate_ae_batch_size = None
    candidate_ae_learning_rate = None
    candidate_ae_threshold = None
    candidate_best_loss = None
    candidate_device = None

    candidate_created_at = None

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    if not retraining_df.empty:

        for _, row in retraining_df.iterrows():

            model_name = str(
                row.get(
                    "model",
                    ""
                )
            ).lower()

            # ------------------------------------------------
            # Common metadata
            # ------------------------------------------------

            if candidate_records is None:

                candidate_records = safe_int(
                    first_value(
                        row.to_dict(),
                        "training_records",
                        "records"
                    )
                )

            if candidate_features is None:

                candidate_features = safe_int(
                    first_value(
                        row.to_dict(),
                        "features",
                        "feature_count"
                    )
                )

            # ------------------------------------------------
            # Isolation Forest
            # ------------------------------------------------

            if "isolation" in model_name:

                candidate_if_time = safe_float(
                    first_value(
                        row.to_dict(),
                        "training_time_seconds",
                        "if_training_time_seconds"
                    )
                )

                candidate_if_estimators = safe_int(
                    first_value(
                        row.to_dict(),
                        "estimators",
                        "if_estimators"
                    )
                )

            # ------------------------------------------------
            # Autoencoder
            # ------------------------------------------------

            elif (
                "autoencoder" in model_name
                or model_name.strip() == "ae"
            ):

                candidate_ae_time = safe_float(
                    first_value(
                        row.to_dict(),
                        "training_time_seconds",
                        "ae_training_time_seconds"
                    )
                )

                candidate_ae_epochs = safe_int(
                    first_value(
                        row.to_dict(),
                        "epochs",
                        "ae_epochs"
                    )
                )

                candidate_ae_batch_size = safe_int(
                    first_value(
                        row.to_dict(),
                        "batch_size",
                        "ae_batch_size"
                    )
                )

                candidate_ae_learning_rate = safe_float(
                    first_value(
                        row.to_dict(),
                        "learning_rate",
                        "ae_learning_rate"
                    )
                )

                candidate_ae_threshold = safe_float(
                    first_value(
                        row.to_dict(),
                        "ae_threshold",
                        "threshold"
                    )
                )

                candidate_best_loss = safe_float(
                    first_value(
                        row.to_dict(),
                        "best_training_loss",
                        "best_loss"
                    )
                )

                candidate_device = first_value(
                    row.to_dict(),
                    "device"
                )

    # --------------------------------------------------------
    # CANDIDATE METADATA JSON FALLBACK
    # --------------------------------------------------------

    candidate_metadata_path = (
        find_candidate_metadata()
    )

    candidate_metadata = (
        safe_read_json(
            candidate_metadata_path
        )
        if candidate_metadata_path
        else {}
    )

    if not isinstance(
        candidate_metadata,
        dict
    ):

        candidate_metadata = {}

    # --------------------------------------------------------
    # Common candidate values
    # --------------------------------------------------------

    if candidate_records is None:

        candidate_records = safe_int(
            first_value(
                candidate_metadata,
                "training_records",
                "records"
            )
        )

    if candidate_features is None:

        candidate_features = safe_int(
            first_value(
                candidate_metadata,
                "feature_count",
                "features",
                "input_dim"
            )
        )

    candidate_created_at = first_value(
        candidate_metadata,
        "created_at",
        "timestamp"
    )

    # --------------------------------------------------------
    # Nested IF metadata
    # --------------------------------------------------------

    candidate_if_meta = candidate_metadata.get(
        "isolation_forest",
        {}
    )

    if not isinstance(
        candidate_if_meta,
        dict
    ):

        candidate_if_meta = {}

    if candidate_if_estimators is None:

        candidate_if_estimators = safe_int(
            first_value(
                candidate_if_meta,
                "estimators",
                "n_estimators"
            )
        )

    if candidate_if_time is None:

        candidate_if_time = safe_float(
            first_value(
                candidate_if_meta,
                "training_time_seconds",
                "training_time"
            )
        )

    # --------------------------------------------------------
    # Nested AE metadata
    # --------------------------------------------------------

    candidate_ae_meta = candidate_metadata.get(
        "autoencoder",
        {}
    )

    if not isinstance(
        candidate_ae_meta,
        dict
    ):

        candidate_ae_meta = {}

    if candidate_ae_epochs is None:

        candidate_ae_epochs = safe_int(
            first_value(
                candidate_ae_meta,
                "epochs"
            )
        )

    if candidate_ae_batch_size is None:

        candidate_ae_batch_size = safe_int(
            first_value(
                candidate_ae_meta,
                "batch_size"
            )
        )

    if candidate_ae_learning_rate is None:

        candidate_ae_learning_rate = safe_float(
            first_value(
                candidate_ae_meta,
                "learning_rate"
            )
        )

    if candidate_ae_time is None:

        candidate_ae_time = safe_float(
            first_value(
                candidate_ae_meta,
                "training_time_seconds",
                "training_time"
            )
        )

    if candidate_ae_threshold is None:

        candidate_ae_threshold = safe_float(
            first_value(
                candidate_ae_meta,
                "ae_threshold",
                "threshold"
            )
        )

    if candidate_best_loss is None:

        candidate_best_loss = safe_float(
            first_value(
                candidate_ae_meta,
                "best_training_loss",
                "best_loss"
            )
        )

    if candidate_device is None:

        candidate_device = first_value(
            candidate_ae_meta,
            "device"
        )

    # --------------------------------------------------------
    # Candidate registry model
    # --------------------------------------------------------

    candidate_version = registry.get(
        "latest_candidate_version"
    )

    candidate_registry_model = None

    for model in models:

        if not isinstance(
            model,
            dict
        ):
            continue

        if (
            candidate_version is not None
            and str(
                model.get("version")
            )
            == str(candidate_version)
        ):

            candidate_registry_model = model
            break

    candidate_metrics = {}

    if candidate_registry_model:

        candidate_metrics = candidate_registry_model.get(
            "metrics",
            {}
        )

        if not isinstance(
            candidate_metrics,
            dict
        ):

            candidate_metrics = {}

    # --------------------------------------------------------
    # Production metadata
    # --------------------------------------------------------

    production_version = registry.get(
        "current_production_version"
    )

    production_metadata = {}

    production_metadata_path = None

    if production_version is not None:

        production_metadata_path = os.path.join(

            PRODUCTION_DIR,

            f"v{production_version}",

            "model_metadata.json"

        )

        production_metadata = safe_read_json(
            production_metadata_path
        )

        if not isinstance(
            production_metadata,
            dict
        ):

            production_metadata = {}

    # --------------------------------------------------------
    # Production metrics
    # --------------------------------------------------------

    production_model = None

    for model in models:

        if not isinstance(
            model,
            dict
        ):
            continue

        if (
            production_version is not None
            and str(
                model.get("version")
            )
            == str(production_version)
        ):

            production_model = model
            break

    production_metrics = {}

    if production_model:

        production_metrics = production_model.get(
            "metrics",
            {}
        )

        if not isinstance(
            production_metrics,
            dict
        ):

            production_metrics = {}

    # --------------------------------------------------------
    # Production nested model metadata
    # --------------------------------------------------------

    production_if = production_metadata.get(
        "isolation_forest",
        {}
    )

    if not isinstance(
        production_if,
        dict
    ):

        production_if = {}

    production_ae = production_metadata.get(
        "autoencoder",
        {}
    )

    if not isinstance(
        production_ae,
        dict
    ):

        production_ae = {}

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "available":
            bool(
                production_metadata
                or candidate_metadata
                or not retraining_df.empty
            ),

        "timestamp":
            time.time(),

        "last_updated":
            time.time(),

        "source":
            report_source,

        "source_path":
            report_path,

        "candidate_version":
            candidate_version,

        "production_version":
            production_version,

        "candidate": {

            "version":
                candidate_version,

            "model_type":
                "candidate",

            "training_records":
                candidate_records,

            "features":
                candidate_features,

            "training_data":
                first_value(
                    candidate_metadata,
                    "training_data",
                    default="NORMAL historical traffic"
                ),

            "training_reason":
                first_value(
                    candidate_metadata,
                    "training_reason",
                    default="ADWIN/self-correction candidate retraining"
                ),

            "created_at":
                candidate_created_at,

            "isolation_forest": {

                "estimators":
                    candidate_if_estimators,

                "training_time_seconds":
                    candidate_if_time

            },

            "autoencoder": {

                "epochs":
                    candidate_ae_epochs,

                "batch_size":
                    candidate_ae_batch_size,

                "learning_rate":
                    candidate_ae_learning_rate,

                "training_time_seconds":
                    candidate_ae_time,

                "best_training_loss":
                    candidate_best_loss,

                "ae_threshold":
                    candidate_ae_threshold,

                "device":
                    candidate_device

            },

            "metrics":
                clean_record(
                    candidate_metrics
                )

        },

        # ----------------------------------------------------
        # Compatibility keys expected by current dashboard JS
        # ----------------------------------------------------

        "isolation_forest": {

            "training_records":
                candidate_records,

            "features":
                candidate_features,

            "estimators":
                candidate_if_estimators,

            "training_time_seconds":
                candidate_if_time

        },

        "autoencoder": {

            "training_records":
                candidate_records,

            "features":
                candidate_features,

            "epochs":
                candidate_ae_epochs,

            "batch_size":
                candidate_ae_batch_size,

            "learning_rate":
                candidate_ae_learning_rate,

            "training_time_seconds":
                candidate_ae_time,

            "best_training_loss":
                candidate_best_loss,

            "ae_threshold":
                candidate_ae_threshold,

            "device":
                candidate_device

        },

        "production": {

            "version":
                production_version,

            "status":
                (
                    production_model.get(
                        "status",
                        "ACTIVE"
                    )
                    if production_model
                    else "ACTIVE"
                ),

            "model_type":
                (
                    production_model.get(
                        "model_type"
                    )
                    if production_model
                    else "Isolation Forest + Autoencoder"
                ),

            "metadata":
                clean_record(
                    production_metadata
                ),

            "metrics":
                clean_record(
                    production_metrics
                ),

            "isolation_forest":
                clean_record(
                    production_if
                ),

            "autoencoder":
                clean_record(
                    production_ae
                )

        }
    }


# ============================================================
# MODEL VERSION MANAGEMENT
# ============================================================

def get_version_management():
    """
    Model lifecycle/version history.
    """

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    models = registry.get(
        "models",
        []
    )

    if not isinstance(
        models,
        list
    ):

        models = []

    versions = []

    for model in models:

        if not isinstance(
            model,
            dict
        ):

            continue

        metrics = model.get(
            "metrics",
            {}
        )

        if not isinstance(
            metrics,
            dict
        ):

            metrics = {}

        metric_changes = model.get(
            "metric_changes",
            {}
        )

        if not isinstance(
            metric_changes,
            dict
        ):

            metric_changes = {}

        versions.append({

            "version":
                model.get(
                    "version"
                ),

            "model_type":
                model.get(
                    "model_type"
                ),

            "status":
                model.get(
                    "status"
                ),

            "path":
                model.get(
                    "path"
                ),

            "reason":
                model.get(
                    "reason"
                ),

            "created_at":
                first_value(
                    model,
                    "created_at",
                    "timestamp",
                    "registered_at"
                ),

            "updated_at":
                first_value(
                    model,
                    "updated_at",
                    "last_updated"
                ),

            "promoted_at":
                first_value(
                    model,
                    "promoted_at"
                ),

            "rejected_at":
                first_value(
                    model,
                    "rejected_at"
                ),

            "metrics":
                {
                    key:
                        clean_value(value)
                    for key, value
                    in metrics.items()
                },

            "metric_changes":
                {
                    key:
                        clean_value(value)
                    for key, value
                    in metric_changes.items()
                }

        })

    versions.sort(
        key=lambda model: (
            safe_int(
                model.get(
                    "version"
                )
            )
            or -1
        ),
        reverse=True
    )

    return {

        "current_production_version":
            registry.get(
                "current_production_version"
            ),

        "latest_candidate_version":
            registry.get(
                "latest_candidate_version"
            ),

        "previous_production_version":
            registry.get(
                "previous_production_version"
            ),

        "last_decision":
            registry.get(
                "last_decision"
            ),

        "last_updated":
            registry.get(
                "last_updated"
            ),

        "versions":
            versions
    }


# ============================================================
# PHASE 12 EVALUATION
# ============================================================

def get_model_evaluation():
    """
    Return Phase-12 old-vs-candidate evaluation.

    Also provides normalized summary fields so the dashboard
    does not have to understand the raw CSV structure.
    """

    comparison = safe_read_csv(
        PHASE12_COMPARISON
    )

    promotion = safe_read_csv(
        PHASE12_PROMOTION
    )

    metadata = safe_read_json(
        PHASE12_METADATA
    )

    comparison_records = (
        comparison
        .to_dict(
            orient="records"
        )
        if not comparison.empty
        else []
    )

    promotion_records = (
        promotion
        .to_dict(
            orient="records"
        )
        if not promotion.empty
        else []
    )

    comparison_records = [
        clean_record(record)
        for record in comparison_records
    ]

    promotion_records = [
        clean_record(record)
        for record in promotion_records
    ]

    # --------------------------------------------------------
    # Registry decision
    # --------------------------------------------------------

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    decision = str(
        registry.get(
            "last_decision",
            ""
        )
    ).upper()

    # --------------------------------------------------------
    # Determine decision from promotion CSV if available.
    # --------------------------------------------------------

    promotion_decision = None
    decision_reason = None

    for record in reversed(
        promotion_records
    ):

        promotion_decision = first_value(
            record,
            "decision",
            "promotion_decision",
            "result",
            "status"
        )

        decision_reason = first_value(
            record,
            "decision_reason",
            "reason",
            "message"
        )

        if promotion_decision is not None:
            break

    if promotion_decision is not None:

        decision = str(
            promotion_decision
        ).upper()

    # --------------------------------------------------------
    # Registry model metrics
    # --------------------------------------------------------

    production_version = registry.get(
        "current_production_version"
    )

    candidate_version = registry.get(
        "latest_candidate_version"
    )

    old_metrics = {}
    candidate_metrics = {}

    models = registry.get(
        "models",
        []
    )

    if isinstance(
        models,
        list
    ):

        for model in models:

            if not isinstance(
                model,
                dict
            ):
                continue

            version = model.get(
                "version"
            )

            metrics = model.get(
                "metrics",
                {}
            )

            if not isinstance(
                metrics,
                dict
            ):

                metrics = {}

            if (
                production_version is not None
                and str(version)
                == str(production_version)
            ):

                old_metrics = metrics

            if (
                candidate_version is not None
                and str(version)
                == str(candidate_version)
            ):

                candidate_metrics = metrics

    # --------------------------------------------------------
    # Extract common metrics from comparison CSV
    # if registry does not contain them.
    # --------------------------------------------------------

    if comparison_records:

        latest_comparison = (
            comparison_records[-1]
        )

        if not old_metrics:

            old_metrics = {
                key: value
                for key, value
                in latest_comparison.items()
                if (
                    "old" in key.lower()
                    or "production" in key.lower()
                )
            }

        if not candidate_metrics:

            candidate_metrics = {
                key: value
                for key, value
                in latest_comparison.items()
                if (
                    "candidate" in key.lower()
                    or "new" in key.lower()
                )
            }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if decision in [
        "ACCEPTED",
        "PROMOTED",
        "APPROVED"
    ]:

        normalized_decision = "PROMOTED"

    elif decision in [
        "REJECTED",
        "DECLINED"
    ]:

        normalized_decision = "REJECTED"

    elif comparison_records or promotion_records:

        normalized_decision = "EVALUATED"

    else:

        normalized_decision = "NOT_AVAILABLE"

    if decision_reason is None:

        if normalized_decision == "REJECTED":

            decision_reason = (
                "Candidate did not satisfy the required "
                "performance criteria. Production model retained."
            )

        elif normalized_decision == "PROMOTED":

            decision_reason = (
                "Candidate satisfied the required performance "
                "criteria and was promoted."
            )

    return {

        "available":
            bool(
                comparison_records
                or promotion_records
                or metadata
            ),

        "status":
            normalized_decision,

        "decision":
            normalized_decision,

        "decision_reason":
            decision_reason,

        "current_production_version":
            production_version,

        "candidate_version":
            candidate_version,

        "comparison":
            comparison_records,

        "promotion":
            promotion_records,

        "metadata":
            metadata,

        "old_metrics":
            clean_record(
                old_metrics
            ),

        "candidate_metrics":
            clean_record(
                candidate_metrics
            )
    }


# ============================================================
# AUTOMATIC PIPELINE STATUS
# ============================================================

def get_pipeline_status():
    """
    Normalize automatic_retraining_orchestrator state.

    Dashboard meanings:

        NOT_TRIGGERED
        TRIGGERED
        ONGOING
        COMPLETED
        FAILED

    Stage meanings:

        CHECKPOINT
        ADWIN
        DRIFT DETECTED
        PYSPARK RETRAINING
        MODEL EVALUATION
        MODEL LIFECYCLE
        COMPLETE
        NO DRIFT
    """

    status = safe_read_json(
        PIPELINE_STATUS_FILE
    )

    if not isinstance(
        status,
        dict
    ):

        status = {}

    automatic_state = safe_read_json(
        AUTOMATIC_STATE_FILE
    )

    if not isinstance(
        automatic_state,
        dict
    ):

        automatic_state = {}

    drift = get_drift()

    retraining = get_retraining()

    evaluation = get_model_evaluation()

    registry = safe_read_json(
        REGISTRY_PATH
    )

    if not isinstance(
        registry,
        dict
    ):

        registry = {}

    # --------------------------------------------------------
    # Raw values
    # --------------------------------------------------------

    raw_status = str(
        first_value(
            status,
            "status",
            default="WAITING"
        )
    ).upper()

    stage = str(
        first_value(
            status,
            "stage",
            default="MONITORING"
        )
    ).upper()

    message = first_value(
        status,
        "message",
        "current_operation",
        "stage_message",
        default="System monitoring historical traffic."
    )

    # --------------------------------------------------------
    # Latest history record
    # --------------------------------------------------------

    latest_record = None

    history_df = load_history()

    if not history_df.empty:

        latest_record = len(
            history_df
        )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = first_value(
        status,
        "checkpoint",
        "last_checkpoint"
    )

    checkpoint = safe_int(
        checkpoint
    )

    if checkpoint is None:

        last_processed = first_value(
            automatic_state,
            "last_processed_checkpoint"
        )

        checkpoint = safe_int(
            last_processed
        )

    # --------------------------------------------------------
    # Latest record from history
    # --------------------------------------------------------

    latest_history_record = (
        latest_record
        if latest_record is not None
        else 0
    )

    # --------------------------------------------------------
    # Last completed checkpoint
    # --------------------------------------------------------

    if checkpoint is None:

        last_completed_checkpoint = (
            (
                latest_history_record
                // CHECKPOINT_SIZE
            )
            * CHECKPOINT_SIZE
        )

    else:

        # If checkpoint is ahead of the current history,
        # don't present it as completed.
        if (
            latest_history_record
            >= checkpoint
        ):

            last_completed_checkpoint = checkpoint

        else:

            last_completed_checkpoint = (
                (
                    latest_history_record
                    // CHECKPOINT_SIZE
                )
                * CHECKPOINT_SIZE
            )

    # --------------------------------------------------------
    # Next checkpoint
    # --------------------------------------------------------

    next_checkpoint = (
        (
            last_completed_checkpoint
            + CHECKPOINT_SIZE
        )
        if last_completed_checkpoint is not None
        else CHECKPOINT_SIZE
    )

    # If no checkpoint has been completed yet.
    if latest_history_record < CHECKPOINT_SIZE:

        last_completed_checkpoint = 0
        next_checkpoint = CHECKPOINT_SIZE

    # --------------------------------------------------------
    # Current new drift count
    # --------------------------------------------------------

    new_drift_count = first_value(
        status,
        "new_drift_count",
        "new_drift_events"
    )

    if new_drift_count is None:

        new_drift_count = first_value(
            status,
            "drift_count"
        )

    new_drift_count = (
        safe_int(
            new_drift_count
        )
        if new_drift_count is not None
        else 0
    )

    # --------------------------------------------------------
    # Historical drift count
    # --------------------------------------------------------

    total_drifts = safe_int(
        drift.get(
            "total_drifts"
        )
    )

    if total_drifts is None:
        total_drifts = 0

    # --------------------------------------------------------
    # Retraining flag
    # --------------------------------------------------------

    lock_active = file_exists(
        RETRAINING_LOCK_FILE
    )

    retraining_status = str(
        retraining.get(
            "current_status",
            "NOT_TRIGGERED"
        )
    ).upper()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation_status = "WAITING"

    if evaluation.get(
        "status"
    ) == "PROMOTED":

        evaluation_status = "COMPLETED"

    elif evaluation.get(
        "status"
    ) == "REJECTED":

        evaluation_status = "COMPLETED"

    elif "EVALUATION" in stage:

        evaluation_status = "RUNNING"

    elif evaluation.get(
        "available"
    ):

        evaluation_status = "COMPLETED"

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    last_decision = str(
        registry.get(
            "last_decision",
            ""
        )
    ).upper()

    if last_decision in [
        "ACCEPTED",
        "PROMOTED"
    ]:

        lifecycle_status = "PROMOTED"

    elif last_decision in [
        "REJECTED",
        "DECLINED"
    ]:

        lifecycle_status = "REJECTED"

    elif "LIFECYCLE" in stage:

        lifecycle_status = "RUNNING"

    else:

        lifecycle_status = "WAITING"

    # --------------------------------------------------------
    # Overall retraining state
    # --------------------------------------------------------

    if lock_active:

        normalized_retraining_status = "ONGOING"

    elif (
        "PYSPARK" in stage
        and "FAILED" not in stage
        and raw_status in [
            "RETRAINING",
            "RUNNING"
        ]
    ):

        normalized_retraining_status = "ONGOING"

    elif stage == "DRIFT DETECTED":

        normalized_retraining_status = "TRIGGERED"

    elif "FAILED" in stage:

        normalized_retraining_status = "FAILED"

    elif (
        "PYSPARK" in stage
        or evaluation_status == "COMPLETED"
        or lifecycle_status in [
            "PROMOTED",
            "REJECTED"
        ]
    ):

        normalized_retraining_status = "COMPLETED"

    elif stage == "NO DRIFT":

        normalized_retraining_status = "NOT_TRIGGERED"

    else:

        normalized_retraining_status = (
            retraining_status
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = first_value(
        status,
        "progress_percentage",
        "progress",
        "training_progress"
    )

    progress = safe_float(
        progress
    )

    if progress is None:

        if normalized_retraining_status == "NOT_TRIGGERED":

            progress = 0

        elif normalized_retraining_status == "TRIGGERED":

            progress = 5

        elif normalized_retraining_status == "ONGOING":

            if "PYSPARK" in stage:

                progress = 45

            elif "EVALUATION" in stage:

                progress = 70

            elif "LIFECYCLE" in stage:

                progress = 90

            else:

                progress = 20

        elif normalized_retraining_status == "COMPLETED":

            progress = 100

        else:

            progress = 0

    progress = max(
        0,
        min(
            100,
            progress
        )
    )

    # --------------------------------------------------------
    # Pipeline overall status
    # --------------------------------------------------------

    if "FAILED" in stage:

        normalized_pipeline_status = "FAILED"

    elif normalized_retraining_status == "ONGOING":

        normalized_pipeline_status = "RETRAINING"

    elif normalized_retraining_status == "TRIGGERED":

        normalized_pipeline_status = "TRIGGERED"

    elif normalized_retraining_status == "COMPLETED":

        normalized_pipeline_status = "COMPLETED"

    else:

        normalized_pipeline_status = (
            raw_status
        )

    # --------------------------------------------------------
    # Display stage
    # --------------------------------------------------------

    if stage:

        display_stage = stage

    else:

        display_stage = "MONITORING"

    # --------------------------------------------------------
    # Training times
    # --------------------------------------------------------

    training_times = first_value(
        status,
        "training_times",
        default={}
    )

    if not isinstance(
        training_times,
        dict
    ):

        training_times = {}

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "available":
            bool(status),

        "status":
            normalized_pipeline_status,

        "pipeline_status":
            normalized_pipeline_status,

        "current_status":
            normalized_pipeline_status,

        "stage":
            display_stage,

        "overall_stage":
            display_stage,

        "message":
            message,

        "timestamp":
            first_value(
                status,
                "timestamp",
                "last_updated"
            ),

        "last_updated":
            first_value(
                status,
                "last_updated",
                "timestamp"
            ),

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        "checkpoint":
            last_completed_checkpoint,

        "last_checkpoint":
            last_completed_checkpoint,

        "next_checkpoint":
            next_checkpoint,

        "latest_record":
            latest_history_record,

        "records_processed":
            latest_history_record,

        # ----------------------------------------------------
        # Drift
        # ----------------------------------------------------

        "drift_count":
            new_drift_count,

        "new_drift_count":
            new_drift_count,

        "total_drifts":
            total_drifts,

        "historical_drift_count":
            total_drifts,

        "drift_status":
            (
                "DETECTED"
                if new_drift_count > 0
                else "WAITING"
            ),

        "adwin_status":
            (
                "DRIFT DETECTED"
                if new_drift_count > 0
                else "WAITING"
            ),

        # ----------------------------------------------------
        # Retraining
        # ----------------------------------------------------

        "retraining":
            normalized_retraining_status
            != "NOT_TRIGGERED",

        "retraining_status":
            normalized_retraining_status,

        "pyspark_status":
            normalized_retraining_status,

        "training_status":
            normalized_retraining_status,

        "training_records":
            retraining.get(
                "training_records"
            ),

        "retraining_records":
            retraining.get(
                "training_records"
            ),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        "evaluation_status":
            evaluation_status,

        "evaluation":
            evaluation_status,

        # ----------------------------------------------------
        # Lifecycle
        # ----------------------------------------------------

        "lifecycle_status":
            lifecycle_status,

        "lifecycle":
            lifecycle_status,

        "lifecycle_decision":
            last_decision,

        "lifecycle_version":
            registry.get(
                "current_production_version"
            ),

        "production_version":
            registry.get(
                "current_production_version"
            ),

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        "progress":
            progress,

        "progress_percentage":
            progress,

        "training_progress":
            progress,

        # ----------------------------------------------------
        # Training times
        # ----------------------------------------------------

        "training_times":
            clean_record(
                training_times
            ),

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        "final_result":
            first_value(
                status,
                "final_result",
                default=(
                    "PROMOTED"
                    if lifecycle_status == "PROMOTED"
                    else (
                        "REJECTED"
                        if lifecycle_status == "REJECTED"
                        else (
                            "NO RETRAINING"
                            if normalized_retraining_status
                            == "NOT_TRIGGERED"
                            else "PENDING"
                        )
                    )
                )
            ),

        "display_result":
            (
                "NO DRIFT → RETRAINING NOT REQUIRED"
                if normalized_retraining_status
                == "NOT_TRIGGERED"
                else (
                    "CANDIDATE REJECTED"
                    if lifecycle_status == "REJECTED"
                    else (
                        "MODEL PROMOTED"
                        if lifecycle_status == "PROMOTED"
                        else display_stage
                    )
                )
            ),

        "new_drift_events":
            drift.get(
                "records",
                []
            ),

        "lock_active":
            lock_active
    }


# ============================================================
# SELF-CORRECTION STATE
# ============================================================

def get_self_correction():
    """
    Return self-correction/orchestrator state.
    """

    state = safe_read_json(
        SELF_CORRECTION_FILE
    )

    source = "self_correction_state.json"

    if not state:

        state = safe_read_json(
            AUTOMATIC_STATE_FILE
        )

        source = "automatic_retraining_state.json"

    if not isinstance(
        state,
        dict
    ):

        state = {}

    if not state:

        return {

            "available":
                False,

            "source":
                None

        }

    return {

        "available":
            True,

        "source":
            source,

        **clean_record(
            state
        )
    }


# ============================================================
# PIPELINE LOG
# ============================================================

def get_pipeline_log(
    limit=PIPELINE_LOG_LIMIT
):
    """
    Read latest automatic retraining pipeline log lines.
    """

    if not file_exists(
        PIPELINE_LOG_FILE
    ):

        return {

            "available":
                False,

            "records":
                []
        }

    try:

        with open(
            PIPELINE_LOG_FILE,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:

            lines = file.readlines()

    except Exception:

        return {

            "available":
                False,

            "records":
                []
        }

    lines = lines[
        -limit:
    ]

    records = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        timestamp = None
        message = line

        # Format:
        # [2026-08-30 17:00:00] message

        if line.startswith("["):

            closing = line.find(
                "]"
            )

            if closing != -1:

                timestamp = line[
                    1:closing
                ]

                message = line[
                    closing + 1:
                ].strip()

        records.append({

            "timestamp":
                timestamp,

            "message":
                message

        })

    return {

        "available":
            True,

        "records":
            records,

        "count":
            len(records)

    }


# ============================================================
# VERSION MANAGEMENT SUMMARY
# ============================================================

def get_dashboard_version_summary():
    """
    Compatibility helper.
    """

    return get_version_management()


# ============================================================
# COMPLETE DASHBOARD DATA
# ============================================================

def get_dashboard_data():
    """
    Return one complete normalized dashboard object.
    """

    system = get_system_summary()

    anomalies = get_anomalies(
        limit=ANOMALY_LIMIT
    )

    drift = get_drift()

    retraining = get_retraining()

    retraining_history = (
        get_retraining_history()
    )

    models = get_models()

    model_metadata = (
        get_model_metadata()
    )

    top_model = get_top_model()

    version_management = (
        get_version_management()
    )

    pipeline = (
        get_pipeline_status()
    )

    self_correction = (
        get_self_correction()
    )

    pipeline_log = (
        get_pipeline_log()
    )

    evaluation = (
        get_model_evaluation()
    )

    return {

        # ====================================================
        # SYSTEM
        # ====================================================

        "system":
            system,

        # ====================================================
        # ANOMALIES
        # ====================================================

        "anomalies":
            anomalies,

        # ====================================================
        # DRIFT
        # ====================================================

        "drift":
            drift,

        # ====================================================
        # RETRAINING
        # ====================================================

        "retraining":
            retraining,

        "retraining_history":
            retraining_history,

        # ====================================================
        # MODELS
        # ====================================================

        "models":
            models,

        "model_metadata":
            model_metadata,

        "top_model":
            top_model,

        "version_management":
            version_management,

        # ====================================================
        # AUTOMATIC PIPELINE
        # ====================================================

        "pipeline":
            pipeline,

        "self_correction":
            self_correction,

        "pipeline_log":
            pipeline_log,

        # ====================================================
        # EVALUATION
        # ====================================================

        "evaluation":
            evaluation,

        # ====================================================
        # DASHBOARD SUMMARY
        # ====================================================

        "summary": {

            "total_logs":
                system.get(
                    "total_logs"
                ),

            "processed":
                system.get(
                    "logs_processed"
                ),

            "normal":
                system.get(
                    "normal_logs"
                ),

            "anomalies":
                system.get(
                    "anomalies_detected"
                ),

            "total_drifts":
                drift.get(
                    "total_drifts",
                    0
                ),

            "new_drift_count":
                drift.get(
                    "new_drift_count",
                    0
                ),

            "latest_record":
                pipeline.get(
                    "latest_record"
                ),

            "last_checkpoint":
                pipeline.get(
                    "checkpoint"
                ),

            "next_checkpoint":
                pipeline.get(
                    "next_checkpoint"
                ),

            "retraining_status":
                pipeline.get(
                    "retraining_status"
                ),

            "production_version":
                models.get(
                    "current_production_version"
                ),

            "candidate_version":
                models.get(
                    "latest_candidate_version"
                ),

            "last_decision":
                models.get(
                    "last_decision"
                )

        },

        # ====================================================
        # UPDATE TIMESTAMP
        # ====================================================

        "timestamp":
            time.time()
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "DASHBOARD DATA MODULE TEST"
    )
    print("=" * 80)

    data = get_dashboard_data()

    print("\nSYSTEM")
    print(
        json.dumps(
            data["system"],
            indent=4,
            default=str
        )
    )

    print("\nDRIFT")
    print(
        json.dumps(
            data["drift"],
            indent=4,
            default=str
        )
    )

    print("\nPIPELINE")
    print(
        json.dumps(
            data["pipeline"],
            indent=4,
            default=str
        )
    )

    print("\nRETRAINING")
    print(
        json.dumps(
            data["retraining"],
            indent=4,
            default=str
        )
    )

    print("\nTOP MODEL")
    print(
        json.dumps(
            data["top_model"],
            indent=4,
            default=str
        )
    )

    print("\nMODEL METADATA")
    print(
        json.dumps(
            data["model_metadata"],
            indent=4,
            default=str
        )
    )

    print("\nVERSION MANAGEMENT")
    print(
        json.dumps(
            data["version_management"],
            indent=4,
            default=str
        )
    )

    print("\nEVALUATION")
    print(
        json.dumps(
            data["evaluation"],
            indent=4,
            default=str
        )
    )

    print("\nPIPELINE LOG")
    print(
        json.dumps(
            data["pipeline_log"],
            indent=4,
            default=str
        )
    )

    print("\nSUMMARY")
    print(
        json.dumps(
            data["summary"],
            indent=4,
            default=str
        )
    )

    print()
    print(
        "DASHBOARD DATA MODULE WORKING."
    )

    print("=" * 80)
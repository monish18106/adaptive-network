# ============================================================
# DASHBOARD DATA MODULE
# Real-Time Network Intrusion Detection
# ============================================================
#
# This module is the SINGLE data source for the dashboard.
#
# Important rules:
# 1. System counters come from Parquet history.
# 2. Recent anomalies come from Parquet history, newest first.
# 3. ADWIN data comes from ADWIN output files.
# 4. Retraining data comes from the NEWEST real retraining artifact.
# 5. Candidate training records are NEVER taken from the
#    evaluation sample size (for example, 6000).
# 6. Candidate training metadata is taken from the actual
#    PySpark/Phase-11 output.
# 7. Model registry is the source of truth for versions/status.
# 8. Production metadata is read from production/vX.
# 9. The API returns one normalized object so JS does not have
#    to guess which endpoint/schema to use.
# ============================================================

import os
import glob
import json
import math
import time

import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

REGISTRY_PATH = os.path.join(
    SAVED_MODELS_DIR,
    "model_registry.json"
)

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

PHASE11_REPORT = os.path.join(
    PROCESSED_DIR,
    "phase11_retraining_report.csv"
)

PYSPARK_REPORT = os.path.join(
    PROCESSED_DIR,
    "pyspark_retraining_report.csv"
)

ADWIN_RESULTS = os.path.join(
    PROCESSED_DIR,
    "adwin_drift_results.csv"
)

ADWIN_LATEST = os.path.join(
    PROCESSED_DIR,
    "adwin_latest_result.json"
)

PIPELINE_STATUS = os.path.join(
    PROCESSED_DIR,
    "automatic_pipeline_status.json"
)

SELF_CORRECTION = os.path.join(
    PROCESSED_DIR,
    "self_correction_state.json"
)

AUTOMATIC_STATE = os.path.join(
    PROCESSED_DIR,
    "automatic_retraining_state.json"
)

PIPELINE_LOG = os.path.join(
    PROCESSED_DIR,
    "retraining_pipeline.log"
)

CHECKPOINT_SIZE = 10_000
ANOMALY_LIMIT = 50
DRIFT_LIMIT = 100
PIPELINE_LOG_LIMIT = 150


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_read_json(path):
    if not path or not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
            return value if value is not None else {}
    except Exception:
        return {}


def safe_read_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if hasattr(value, "tolist") and not isinstance(
        value, (str, bytes, dict)
    ):
        try:
            value = value.tolist()
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def clean_record(record):
    if not isinstance(record, dict):
        return record
    return {
        key: clean_value(value)
        for key, value in record.items()
    }


def safe_float(value):
    if value is None:
        return None

    try:
        if isinstance(value, str) and not value.strip():
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def safe_int(value):
    if value is None:
        return None

    try:
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except Exception:
        return None


def first_value(source, *keys, default=None):
    if not isinstance(source, dict):
        return default

    for key in keys:
        if key in source:
            value = source.get(key)
            if value is not None:
                return value

    return default


def file_mtime(path):
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0
    except Exception:
        return 0


# ============================================================
# HISTORY
# ============================================================

def load_history():
    if not os.path.exists(HISTORY_DIR):
        return pd.DataFrame()

    files = glob.glob(
        os.path.join(
            HISTORY_DIR,
            "**",
            "*.parquet"
        ),
        recursive=True
    )

    frames = []

    for path in sorted(files):
        try:
            frame = pd.read_parquet(path)
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


def sort_history(df):
    if df.empty:
        return df

    result = df.copy()

    if "record_index" in result.columns:
        result["_record_sort"] = pd.to_numeric(
            result["record_index"],
            errors="coerce"
        )
        result = result.sort_values(
            "_record_sort",
            ascending=True,
            na_position="last"
        )
        result = result.drop(
            columns=["_record_sort"],
            errors="ignore"
        )
        return result

    if "timestamp" in result.columns:
        result["_time_sort"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce"
        )
        result = result.sort_values(
            "_time_sort",
            ascending=True,
            na_position="last"
        )
        result = result.drop(
            columns=["_time_sort"],
            errors="ignore"
        )

    return result


# ============================================================
# SYSTEM
# ============================================================

def get_system_summary():
    df = sort_history(load_history())

    total = int(len(df))
    normal = 0
    anomalies = 0

    if not df.empty:
        if "status" in df.columns:
            status = (
                df["status"]
                .astype(str)
                .str.upper()
                .str.strip()
            )
            normal = int((status == "NORMAL").sum())
            anomalies = int((status == "ANOMALY").sum())

        elif "prediction" in df.columns:
            prediction = (
                df["prediction"]
                .astype(str)
                .str.upper()
                .str.strip()
            )
            anomalies = int((prediction == "ANOMALY").sum())
            normal = max(total - anomalies, 0)

    registry = safe_read_json(REGISTRY_PATH)
    pipeline = safe_read_json(PIPELINE_STATUS)

    return {
        "total_logs": total,
        "logs_processed": total,
        "normal_logs": normal,
        "anomalies_detected": anomalies,
        "current_production_version": first_value(
            registry,
            "current_production_version"
        ),
        "latest_candidate_version": first_value(
            registry,
            "latest_candidate_version"
        ),
        "last_model_decision": first_value(
            registry,
            "last_decision"
        ),
        "system_status": "RUNNING",
        "pipeline_status": first_value(
            pipeline,
            "status",
            default="WAITING"
        ),
        "pipeline_stage": first_value(
            pipeline,
            "stage",
            default="MONITORING"
        ),
        "timestamp": time.time()
    }


# ============================================================
# ANOMALIES
# ============================================================

def get_anomalies(limit=ANOMALY_LIMIT):
    df = load_history()

    if df.empty or "status" not in df.columns:
        return []

    work = df.copy()

    status = (
        work["status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    work = work[status == "ANOMALY"].copy()

    if work.empty:
        return []

    if "record_index" in work.columns:
        work["_sort_record"] = pd.to_numeric(
            work["record_index"],
            errors="coerce"
        )
        work = work.sort_values(
            "_sort_record",
            ascending=False,
            na_position="last"
        )
    elif "timestamp" in work.columns:
        work["_sort_time"] = pd.to_datetime(
            work["timestamp"],
            errors="coerce"
        )
        work = work.sort_values(
            "_sort_time",
            ascending=False,
            na_position="last"
        )

    work = work.head(limit)

    records = []

    for _, row in work.iterrows():
        features = row.get("features", [])

        if hasattr(features, "tolist"):
            try:
                features = features.tolist()
            except Exception:
                features = []

        record = {
            "record_index": clean_value(
                row.get("record_index")
            ),
            "timestamp": clean_value(
                row.get("timestamp")
            ),
            "status": "ANOMALY",
            "if_score": safe_float(
                row.get("if_score")
            ),
            "ae_score": safe_float(
                row.get("ae_score")
            ),
            "final_score": safe_float(
                row.get("final_score")
            ),
            "threshold": safe_float(
                row.get("threshold")
            ),
            "features": features
        }

        for field in [
            "source_ip",
            "destination_ip",
            "src_ip",
            "dst_ip",
            "protocol",
            "src_port",
            "dst_port",
            "label",
            "prediction",
            "attack_type"
        ]:
            if field in row.index:
                record[field] = clean_value(
                    row.get(field)
                )

        records.append(clean_record(record))

    return records


# ============================================================
# ADWIN
# ============================================================

def get_drift():
    df = safe_read_csv(ADWIN_RESULTS)
    latest_result = safe_read_json(ADWIN_LATEST)

    if df.empty:
        return {
            "available": bool(latest_result),
            "total_drifts": 0,
            "new_drift_count": safe_int(
                first_value(
                    latest_result,
                    "new_drift_count",
                    "drift_count",
                    default=0
                )
            ) or 0,
            "latest_drift": None,
            "latest_result": clean_record(
                latest_result
            ),
            "records": []
        }

    drift_column = None

    for candidate in [
        "drift_detected",
        "drift",
        "is_drift",
        "change_detected"
    ]:
        if candidate in df.columns:
            drift_column = candidate
            break

    if drift_column is None:
        for column in df.columns:
            if "drift" in str(column).lower():
                drift_column = column
                break

    if drift_column is None:
        return {
            "available": True,
            "total_drifts": 0,
            "new_drift_count": 0,
            "latest_drift": None,
            "latest_result": clean_record(latest_result),
            "records": []
        }

    values = (
        df[drift_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    drift_df = df[
        values.isin([
            "true",
            "1",
            "yes",
            "detected"
        ])
    ].copy()

    if "record_index" in drift_df.columns:
        drift_df["_record_sort"] = pd.to_numeric(
            drift_df["record_index"],
            errors="coerce"
        )
        drift_df = drift_df.sort_values(
            "_record_sort",
            ascending=True,
            na_position="last"
        )
    elif "timestamp" in drift_df.columns:
        drift_df["_time_sort"] = pd.to_datetime(
            drift_df["timestamp"],
            errors="coerce"
        )
        drift_df = drift_df.sort_values(
            "_time_sort",
            ascending=True,
            na_position="last"
        )

    records = [
        clean_record(row)
        for row in drift_df.tail(DRIFT_LIMIT).to_dict(
            orient="records"
        )
    ]

    latest = records[-1] if records else None

    pipeline = safe_read_json(PIPELINE_STATUS)

    new_drift_count = first_value(
        pipeline,
        "new_drift_count",
        "new_drift_events"
    )

    if new_drift_count is None:
        new_drift_count = first_value(
            latest_result,
            "new_drift_count",
            default=0
        )

    return {
        "available": True,
        "total_drifts": int(len(drift_df)),
        "new_drift_count": safe_int(
            new_drift_count
        ) or 0,
        "latest_drift": latest,
        "latest_result": clean_record(
            latest_result
        ),
        "records": records
    }


# ============================================================
# RETRAINING ARTIFACT SELECTION
# ============================================================

def choose_latest_retraining_report():
    candidates = []

    for path in [
        PYSPARK_REPORT,
        PHASE11_REPORT
    ]:
        if os.path.exists(path):
            candidates.append(path)

    if not candidates:
        return None

    return max(
        candidates,
        key=file_mtime
    )


def find_candidate_metadata():
    candidates = []

    # The root candidate metadata is written by the
    # normal-traffic retraining stage and must be preferred
    # over evaluation metadata.
    for path in [
        os.path.join(
            SAVED_MODELS_DIR,
            "candidate_training_metadata.json"
        ),
        os.path.join(
            SAVED_MODELS_DIR,
            "candidate_ae_metadata.json"
        ),
        os.path.join(
            SAVED_MODELS_DIR,
            "candidate_ae_metadata.pkl"
        )
    ]:
        if os.path.exists(path):
            candidates.append(path)

    candidate_version = safe_read_json(
        REGISTRY_PATH
    ).get("latest_candidate_version")

    if candidate_version is not None:
        version_dir = os.path.join(
            CANDIDATES_DIR,
            f"v{candidate_version}"
        )

        for name in [
            "candidate_training_metadata.json",
            "model_metadata.json",
            "candidate_metadata.json"
        ]:
            path = os.path.join(version_dir, name)
            if os.path.exists(path):
                candidates.append(path)

    if not candidates:
        return None

    # Prefer explicit candidate_training_metadata.json.
    explicit = [
        p for p in candidates
        if os.path.basename(p).lower()
        == "candidate_training_metadata.json"
    ]

    if explicit:
        return max(explicit, key=file_mtime)

    return max(candidates, key=file_mtime)


def load_metadata_file(path):
    if not path:
        return {}

    if path.lower().endswith(".json"):
        return safe_read_json(path)

    # PKL is only a compatibility fallback.
    # We deliberately do not make dashboard operation depend
    # on joblib being installed.
    try:
        import joblib
        value = joblib.load(path)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


# ============================================================
# RETRAINING DATA NORMALIZATION
# ============================================================

def normalize_training_row(row):
    source = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    model_name = str(
        first_value(
            source,
            "model",
            "model_name",
            "name",
            default="MODEL"
        )
    )

    return {
        "model": model_name,
        "model_type": first_value(
            source,
            "model_type",
            "type",
            default="candidate"
        ),
        "status": first_value(
            source,
            "status",
            default="COMPLETED"
        ),
        "training_records": safe_int(
            first_value(
                source,
                "training_records",
                "records",
                "record_count"
            )
        ),
        "features": safe_int(
            first_value(
                source,
                "features",
                "feature_count",
                "input_dim"
            )
        ),
        "training_time_seconds": safe_float(
            first_value(
                source,
                "training_time_seconds",
                "training_time"
            )
        ),
        "estimators": safe_int(
            first_value(
                source,
                "estimators",
                "n_estimators",
                "if_estimators"
            )
        ),
        "epochs": safe_int(
            first_value(
                source,
                "epochs",
                "ae_epochs"
            )
        ),
        "batch_size": safe_int(
            first_value(
                source,
                "batch_size",
                "ae_batch_size"
            )
        ),
        "learning_rate": safe_float(
            first_value(
                source,
                "learning_rate",
                "ae_learning_rate"
            )
        ),
        "ae_threshold": safe_float(
            first_value(
                source,
                "ae_threshold",
                "threshold"
            )
        ),
        "best_training_loss": safe_float(
            first_value(
                source,
                "best_training_loss",
                "best_loss",
                "ae_best_loss"
            )
        ),
        "device": first_value(
            source,
            "device"
        ),
        "created_at": first_value(
            source,
            "created_at",
            "timestamp"
        )
    }


def get_retraining():
    report_path = choose_latest_retraining_report()

    df = safe_read_csv(report_path)

    metadata_path = find_candidate_metadata()
    metadata = load_metadata_file(metadata_path)

    records = []

    if not df.empty:
        for _, row in df.iterrows():
            records.append(
                normalize_training_row(row)
            )

    # If the report has one row per model, this is the actual
    # PySpark training result. Do NOT replace it with the
    # evaluation sample size.
    if records:
        latest_record = records[-1]

    else:
        latest_record = None

    if not latest_record:
        latest_record = {
            "model": "candidate",
            "model_type": "candidate",
            "status": "UNKNOWN"
        }

    # Metadata JSON can fill missing fields only.
    nested_if = metadata.get(
        "isolation_forest",
        {}
    )
    nested_ae = metadata.get(
        "autoencoder",
        {}
    )

    if not isinstance(nested_if, dict):
        nested_if = {}

    if not isinstance(nested_ae, dict):
        nested_ae = {}

    # Apply metadata fallback to every relevant row.
    for record in records:
        name = str(record.get("model", "")).lower()

        if "isolation" in name:
            record["estimators"] = (
                record["estimators"]
                if record["estimators"] is not None
                else safe_int(
                    first_value(
                        nested_if,
                        "estimators",
                        "n_estimators"
                    )
                )
            )

        if "autoencoder" in name or name == "ae":
            record["epochs"] = (
                record["epochs"]
                if record["epochs"] is not None
                else safe_int(
                    first_value(
                        nested_ae,
                        "epochs"
                    )
                )
            )

            record["batch_size"] = (
                record["batch_size"]
                if record["batch_size"] is not None
                else safe_int(
                    first_value(
                        nested_ae,
                        "batch_size"
                    )
                )
            )

            record["learning_rate"] = (
                record["learning_rate"]
                if record["learning_rate"] is not None
                else safe_float(
                    first_value(
                        nested_ae,
                        "learning_rate"
                    )
                )
            )

            record["ae_threshold"] = (
                record["ae_threshold"]
                if record["ae_threshold"] is not None
                else safe_float(
                    first_value(
                        nested_ae,
                        "ae_threshold",
                        "threshold"
                    )
                )
            )

            record["best_training_loss"] = (
                record["best_training_loss"]
                if record["best_training_loss"] is not None
                else safe_float(
                    first_value(
                        nested_ae,
                        "best_training_loss",
                        "best_loss"
                    )
                )
            )

    history = [
        clean_record(record)
        for record in records
    ]

    # Normal records available in the actual history.
    history_df = load_history()
    normal_records = 0

    if not history_df.empty and "status" in history_df.columns:
        normal_records = int(
            (
                history_df["status"]
                .astype(str)
                .str.upper()
                .str.strip()
                == "NORMAL"
            ).sum()
        )

    return {
        "available": bool(records or metadata),
        "source": os.path.basename(report_path)
            if report_path else None,
        "source_path": report_path,
        "metadata_source": metadata_path,
        "normal_records_available": normal_records,
        "records": history,
        "latest": clean_record(
            latest_record
        ) if latest_record else None,
        "metadata": clean_record(
            metadata
        ) if isinstance(metadata, dict) else {}
    }


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_models():
    registry = safe_read_json(REGISTRY_PATH)

    raw_models = registry.get("models", [])

    if not isinstance(raw_models, list):
        raw_models = []

    models = []

    for model in raw_models:
        if not isinstance(model, dict):
            continue

        item = dict(model)

        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        item["metrics"] = clean_record(metrics)

        if "metric_changes" in item:
            changes = item.get("metric_changes", {})
            item["metric_changes"] = (
                clean_record(changes)
                if isinstance(changes, dict)
                else {}
            )

        models.append(clean_record(item))

    return {
        "current_production_version": registry.get(
            "current_production_version"
        ),
        "latest_candidate_version": registry.get(
            "latest_candidate_version"
        ),
        "previous_production_version": registry.get(
            "previous_production_version"
        ),
        "last_decision": registry.get(
            "last_decision"
        ),
        "last_updated": registry.get(
            "last_updated"
        ),
        "models": models
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def get_model_evaluation():
    comparison = safe_read_csv(PHASE12_COMPARISON)
    promotion = safe_read_csv(PHASE12_PROMOTION)
    metadata = safe_read_json(PHASE12_METADATA)

    comparison_records = [
        clean_record(row)
        for row in comparison.to_dict(
            orient="records"
        )
    ] if not comparison.empty else []

    promotion_records = [
        clean_record(row)
        for row in promotion.to_dict(
            orient="records"
        )
    ] if not promotion.empty else []

    # Normalize common column names so the UI can render
    # different versions of the evaluation script.
    normalized_comparison = []

    for row in comparison_records:
        normalized_comparison.append({
            **row,
            "model": first_value(
                row,
                "model",
                "model_name",
                "name"
            ),
            "version": first_value(
                row,
                "version"
            ),
            "f1": safe_float(
                first_value(row, "f1", "F1")
            ),
            "accuracy": safe_float(
                first_value(row, "accuracy", "Accuracy")
            ),
            "detection_rate": safe_float(
                first_value(
                    row,
                    "detection_rate",
                    "detection"
                )
            ),
            "false_alarm_rate": safe_float(
                first_value(
                    row,
                    "false_alarm_rate",
                    "far"
                )
            )
        })

    return {
        "available": bool(
            normalized_comparison
            or promotion_records
            or metadata
        ),
        "comparison": normalized_comparison,
        "promotion": promotion_records,
        "metadata": clean_record(
            metadata
        ) if isinstance(metadata, dict) else {}
    }


# ============================================================
# PRODUCTION METADATA
# ============================================================

def find_production_metadata(version):
    if version is None:
        return None

    candidates = [
        os.path.join(
            PRODUCTION_DIR,
            f"v{version}",
            "model_metadata.json"
        ),
        os.path.join(
            PRODUCTION_DIR,
            f"v{version}",
            "metadata.json"
        ),
        os.path.join(
            PRODUCTION_DIR,
            f"version_{version}",
            "model_metadata.json"
        )
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def build_version_metadata(model_record, version, status):
    if not isinstance(model_record, dict):
        model_record = {}

    metrics = model_record.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "version": version,
        "status": status,
        "model_name": first_value(
            model_record,
            "model_name",
            "model",
            "name",
            default="Isolation Forest + Autoencoder"
        ),
        "training_records": safe_int(
            first_value(
                model_record,
                "training_records",
                "records"
            )
        ),
        "features": safe_int(
            first_value(
                model_record,
                "features",
                "feature_count",
                "input_dim"
            )
        ),
        "if_estimators": safe_int(
            first_value(
                model_record,
                "if_estimators",
                "estimators",
                "n_estimators"
            )
        ),
        "if_training_time_seconds": safe_float(
            first_value(
                model_record,
                "if_training_time_seconds",
                "if_training_time"
            )
        ),
        "ae_epochs": safe_int(
            first_value(
                model_record,
                "ae_epochs",
                "epochs"
            )
        ),
        "ae_batch_size": safe_int(
            first_value(
                model_record,
                "ae_batch_size",
                "batch_size"
            )
        ),
        "ae_learning_rate": safe_float(
            first_value(
                model_record,
                "ae_learning_rate",
                "learning_rate"
            )
        ),
        "ae_training_time_seconds": safe_float(
            first_value(
                model_record,
                "ae_training_time_seconds",
                "ae_training_time"
            )
        ),
        "ae_best_loss": safe_float(
            first_value(
                model_record,
                "ae_best_loss",
                "best_training_loss",
                "best_loss"
            )
        ),
        "ae_threshold": safe_float(
            first_value(
                model_record,
                "ae_threshold",
                "threshold"
            )
        ),
        "device": first_value(
            model_record,
            "device"
        ),
        "created_at": first_value(
            model_record,
            "created_at",
            "timestamp"
        ),
        "metrics": clean_record(metrics),
        "path": first_value(
            model_record,
            "path"
        ),
        "reason": first_value(
            model_record,
            "reason",
            "promotion_reason"
        )
    }


# ============================================================
# MODEL METADATA
# ============================================================

def get_model_metadata():
    registry = safe_read_json(REGISTRY_PATH)
    models = registry.get("models", [])

    if not isinstance(models, list):
        models = []

    production_version = registry.get(
        "current_production_version"
    )

    candidate_version = registry.get(
        "latest_candidate_version"
    )

    # --------------------------------------------------------
    # Production metadata
    # --------------------------------------------------------

    production_record = None

    for model in models:
        if not isinstance(model, dict):
            continue

        if (
            production_version is not None
            and str(model.get("version"))
            == str(production_version)
        ):
            production_record = model
            break

    production_metadata_path = (
        find_production_metadata(
            production_version
        )
    )

    production_file_metadata = (
        safe_read_json(
            production_metadata_path
        )
        if production_metadata_path
        else {}
    )

    if not isinstance(
        production_file_metadata,
        dict
    ):
        production_file_metadata = {}

    production = build_version_metadata(
        production_record or {},
        production_version,
        "PRODUCTION"
    )

    # File metadata is authoritative for detailed
    # training parameters when present.
    production.update({
        "file_metadata": clean_record(
            production_file_metadata
        ),
        "metadata_source": production_metadata_path
    })

    for key in [
        "training_records",
        "features",
        "if_estimators",
        "if_training_time_seconds",
        "ae_epochs",
        "ae_batch_size",
        "ae_learning_rate",
        "ae_training_time_seconds",
        "ae_best_loss",
        "ae_threshold",
        "device",
        "created_at"
    ]:
        if production.get(key) is None:
            production[key] = first_value(
                production_file_metadata,
                key
            )

    # --------------------------------------------------------
    # Candidate metadata
    # --------------------------------------------------------

    retraining = get_retraining()
    training_records = retraining.get(
        "records",
        []
    )

    candidate_meta_path = retraining.get(
        "metadata_source"
    )

    candidate_file_metadata = retraining.get(
        "metadata",
        {}
    )

    if not isinstance(
        candidate_file_metadata,
        dict
    ):
        candidate_file_metadata = {}

    candidate_record = None

    for model in models:
        if not isinstance(model, dict):
            continue

        if (
            candidate_version is not None
            and str(model.get("version"))
            == str(candidate_version)
        ):
            candidate_record = model
            break

    candidate = build_version_metadata(
        candidate_record or {},
        candidate_version,
        "CANDIDATE"
    )

    # Actual retraining report overrides registry placeholders.
    for row in training_records:
        name = str(row.get("model", "")).lower()

        if (
            "isolation" in name
            and row.get("training_records") is not None
        ):
            candidate["training_records"] = row.get(
                "training_records"
            )
            candidate["features"] = (
                row.get("features")
                if row.get("features") is not None
                else candidate["features"]
            )
            candidate["if_estimators"] = (
                row.get("estimators")
                if row.get("estimators") is not None
                else candidate["if_estimators"]
            )
            candidate["if_training_time_seconds"] = (
                row.get("training_time_seconds")
                if row.get("training_time_seconds") is not None
                else candidate["if_training_time_seconds"]
            )

        if "autoencoder" in name or name == "ae":
            candidate["training_records"] = (
                row.get("training_records")
                if row.get("training_records") is not None
                else candidate["training_records"]
            )
            candidate["features"] = (
                row.get("features")
                if row.get("features") is not None
                else candidate["features"]
            )
            candidate["ae_epochs"] = (
                row.get("epochs")
                if row.get("epochs") is not None
                else candidate["ae_epochs"]
            )
            candidate["ae_batch_size"] = (
                row.get("batch_size")
                if row.get("batch_size") is not None
                else candidate["ae_batch_size"]
            )
            candidate["ae_learning_rate"] = (
                row.get("learning_rate")
                if row.get("learning_rate") is not None
                else candidate["ae_learning_rate"]
            )
            candidate["ae_training_time_seconds"] = (
                row.get("training_time_seconds")
                if row.get("training_time_seconds") is not None
                else candidate["ae_training_time_seconds"]
            )
            candidate["ae_best_loss"] = (
                row.get("best_training_loss")
                if row.get("best_training_loss") is not None
                else candidate["ae_best_loss"]
            )
            candidate["ae_threshold"] = (
                row.get("ae_threshold")
                if row.get("ae_threshold") is not None
                else candidate["ae_threshold"]
            )
            candidate["device"] = (
                row.get("device")
                if row.get("device") is not None
                else candidate["device"]
            )

    # Metadata JSON fills remaining gaps.
    nested_if = candidate_file_metadata.get(
        "isolation_forest",
        {}
    )
    nested_ae = candidate_file_metadata.get(
        "autoencoder",
        {}
    )

    if not isinstance(nested_if, dict):
        nested_if = {}

    if not isinstance(nested_ae, dict):
        nested_ae = {}

    candidate["training_records"] = (
        candidate["training_records"]
        if candidate["training_records"] is not None
        else safe_int(
            first_value(
                candidate_file_metadata,
                "training_records",
                "records"
            )
        )
    )

    candidate["features"] = (
        candidate["features"]
        if candidate["features"] is not None
        else safe_int(
            first_value(
                candidate_file_metadata,
                "features",
                "feature_count",
                "input_dim"
            )
        )
    )

    candidate["if_estimators"] = (
        candidate["if_estimators"]
        if candidate["if_estimators"] is not None
        else safe_int(
            first_value(
                nested_if,
                "estimators",
                "n_estimators"
            )
        )
    )

    candidate["ae_epochs"] = (
        candidate["ae_epochs"]
        if candidate["ae_epochs"] is not None
        else safe_int(
            first_value(
                nested_ae,
                "epochs"
            )
        )
    )

    candidate["ae_batch_size"] = (
        candidate["ae_batch_size"]
        if candidate["ae_batch_size"] is not None
        else safe_int(
            first_value(
                nested_ae,
                "batch_size"
            )
        )
    )

    candidate["ae_learning_rate"] = (
        candidate["ae_learning_rate"]
        if candidate["ae_learning_rate"] is not None
        else safe_float(
            first_value(
                nested_ae,
                "learning_rate"
            )
        )
    )

    candidate["ae_training_time_seconds"] = (
        candidate["ae_training_time_seconds"]
        if candidate["ae_training_time_seconds"] is not None
        else safe_float(
            first_value(
                nested_ae,
                "training_time_seconds",
                "training_time"
            )
        )
    )

    candidate["ae_best_loss"] = (
        candidate["ae_best_loss"]
        if candidate["ae_best_loss"] is not None
        else safe_float(
            first_value(
                nested_ae,
                "best_training_loss",
                "best_loss"
            )
        )
    )

    candidate["ae_threshold"] = (
        candidate["ae_threshold"]
        if candidate["ae_threshold"] is not None
        else safe_float(
            first_value(
                nested_ae,
                "ae_threshold",
                "threshold"
            )
        )
    )

    candidate["device"] = (
        candidate["device"]
        if candidate["device"] is not None
        else first_value(
            nested_ae,
            "device",
            default=first_value(
                candidate_file_metadata,
                "device"
            )
        )
    )

    candidate["file_metadata"] = clean_record(
        candidate_file_metadata
    )

    candidate["metadata_source"] = candidate_meta_path

    return {
        "available": bool(
            production_record
            or production_file_metadata
            or candidate_record
            or candidate_file_metadata
            or training_records
        ),
        "production": clean_record(
            production
        ),
        "candidate": clean_record(
            candidate
        ),
        # Kept for compatibility with the current JS.
        "models": [
            clean_record(production),
            clean_record(candidate)
        ],
        "score_configuration": get_score_configuration(),
        "timestamp": time.time(),
        "last_updated": time.time()
    }


# ============================================================
# SCORE CONFIGURATION
# ============================================================

def get_score_configuration():
    metadata = safe_read_json(PHASE12_METADATA)

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "if_weight": safe_float(
            first_value(
                metadata,
                "if_weight",
                "isolation_forest_weight"
            )
        ),
        "ae_weight": safe_float(
            first_value(
                metadata,
                "ae_weight",
                "autoencoder_weight"
            )
        ),
        "fusion_threshold": safe_float(
            first_value(
                metadata,
                "fusion_threshold",
                "threshold"
            )
        ),
        "f1_tolerance": safe_float(
            first_value(
                metadata,
                "f1_tolerance"
            )
        ),
        "false_alarm_tolerance": safe_float(
            first_value(
                metadata,
                "false_alarm_tolerance",
                "far_tolerance"
            )
        ),
        "evaluation_records": safe_int(
            first_value(
                metadata,
                "evaluation_records",
                "records_evaluated"
            )
        )
    }


# ============================================================
# TOP MODEL
# ============================================================

def get_top_model():
    evaluation = get_model_evaluation()
    comparison = evaluation.get(
        "comparison",
        []
    )

    valid = []

    for row in comparison:
        f1 = safe_float(row.get("f1"))
        if f1 is not None:
            valid.append((f1, row))

    if not valid:
        return {
            "available": False,
            "model": None,
            "selection_reason": "No evaluated model is available."
        }

    # Highest F1 is the primary model-selection criterion.
    valid.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best = valid[0][1]

    return {
        "available": True,
        "model": clean_record(best),
        "selection_reason": (
            "Highest F1 among evaluated model versions."
        )
    }


# ============================================================
# PIPELINE STATUS
# ============================================================

def get_pipeline_status():
    data = safe_read_json(PIPELINE_STATUS)

    if not isinstance(data, dict):
        data = {}

    registry = safe_read_json(REGISTRY_PATH)
    retraining = get_retraining()
    evaluation = get_model_evaluation()

    result = dict(data)

    # Never invent current record values.
    # Prefer explicit pipeline values; otherwise derive from
    # actual history.
    history = load_history()
    history_count = int(len(history))

    result["latest_record"] = safe_int(
        first_value(
            result,
            "latest_record",
            "current_record",
            "record_count"
        )
    )

    if result["latest_record"] is None:
        result["latest_record"] = history_count

    result["checkpoint"] = safe_int(
        first_value(
            result,
            "checkpoint",
            "last_checkpoint"
        )
    )

    if result["checkpoint"] is None:
        completed_checkpoint = (
            history_count // CHECKPOINT_SIZE
        ) * CHECKPOINT_SIZE
        result["checkpoint"] = (
            completed_checkpoint
            if completed_checkpoint > 0
            else None
        )

    result["next_checkpoint"] = (
        safe_int(
            first_value(
                result,
                "next_checkpoint"
            )
        )
    )

    if result["next_checkpoint"] is None:
        result["next_checkpoint"] = (
            (
                history_count // CHECKPOINT_SIZE
            ) + 1
        ) * CHECKPOINT_SIZE
    elif history_count >= result["next_checkpoint"]:
        result["next_checkpoint"] = (
            (
                history_count // CHECKPOINT_SIZE
            ) + 1
        ) * CHECKPOINT_SIZE

    drift = get_drift()

    result["drift_count"] = safe_int(
        first_value(
            result,
            "new_drift_count",
            "drift_count",
            default=drift.get(
                "new_drift_count",
                0
            )
        )
    ) or 0

    # Retraining status must reflect the actual state file
    # when present, not a stale hard-coded UI value.
    retraining_state = str(
        first_value(
            result,
            "retraining_status",
            "retraining",
            default=""
        )
    ).upper()

    if isinstance(
        first_value(result, "retraining"),
        bool
    ):
        retraining_flag = bool(
            result.get("retraining")
        )
    else:
        retraining_flag = retraining_state in [
            "RUNNING",
            "ONGOING",
            "IN_PROGRESS",
            "COMPLETED",
            "TRUE"
        ]

    if retraining.get("available"):
        retraining_flag = True

    result["retraining"] = retraining_flag

    result["production_version"] = registry.get(
        "current_production_version"
    )

    result["candidate_version"] = registry.get(
        "latest_candidate_version"
    )

    result["evaluation"] = first_value(
        result,
        "evaluation",
        default=(
            "COMPLETED"
            if evaluation.get("available")
            else "NOT_STARTED"
        )
    )

    result["lifecycle"] = first_value(
        result,
        "lifecycle",
        default=(
            "COMPLETED"
            if registry.get("last_decision")
            else "NOT_STARTED"
        )
    )

    result["final_result"] = first_value(
        result,
        "final_result",
        "decision",
        default=registry.get(
            "last_decision",
            "NONE"
        )
    )

    # Human-readable status.
    final = str(
        result["final_result"]
    ).upper()

    if final in ["PROMOTED", "ACCEPTED"]:
        result["display_result"] = "NEW MODEL PROMOTED"
    elif final == "REJECTED":
        result["display_result"] = "CANDIDATE REJECTED"
    elif str(result["evaluation"]).upper() == "COMPLETED":
        result["display_result"] = "MODEL EVALUATION COMPLETED"
    elif retraining.get("available"):
        result["display_result"] = "RETRAINING COMPLETED"
    else:
        result["display_result"] = first_value(
            result,
            "stage",
            default="WAITING"
        )

    result["training_metadata"] = (
        retraining.get("latest")
    )

    result["timestamp"] = first_value(
        result,
        "timestamp",
        "last_updated",
        default=time.time()
    )

    return clean_record(result)


# ============================================================
# SELF CORRECTION
# ============================================================

def get_self_correction():
    state = safe_read_json(SELF_CORRECTION)

    if not state:
        state = safe_read_json(AUTOMATIC_STATE)

    if not isinstance(state, dict) or not state:
        return {
            "available": False
        }

    return {
        "available": True,
        **clean_record(state)
    }


# ============================================================
# PIPELINE LOG
# ============================================================

def get_pipeline_log(limit=PIPELINE_LOG_LIMIT):
    if not os.path.exists(PIPELINE_LOG):
        return {
            "available": False,
            "records": []
        }

    try:
        with open(
            PIPELINE_LOG,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            lines = f.readlines()[-limit:]
    except Exception:
        return {
            "available": False,
            "records": []
        }

    records = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        timestamp = None
        message = line

        if line.startswith("["):
            close = line.find("]")
            if close != -1:
                timestamp = line[1:close]
                message = line[close + 1:].strip()

        records.append({
            "timestamp": timestamp,
            "message": message
        })

    return {
        "available": True,
        "records": records,
        "count": len(records)
    }


# ============================================================
# VERSION MANAGEMENT
# ============================================================

def get_version_management():
    registry = get_models()

    versions = []

    for model in registry.get("models", []):
        versions.append(clean_record(model))

    return {
        "current_production_version": registry.get(
            "current_production_version"
        ),
        "latest_candidate_version": registry.get(
            "latest_candidate_version"
        ),
        "previous_production_version": registry.get(
            "previous_production_version"
        ),
        "last_decision": registry.get(
            "last_decision"
        ),
        "last_updated": registry.get(
            "last_updated"
        ),
        "versions": versions,
        # Compatibility key expected by the current JS.
        "models": versions
    }


# ============================================================
# COMPLETE DASHBOARD DATA
# ============================================================

def get_dashboard_data():
    system = get_system_summary()
    anomalies = get_anomalies()
    drift = get_drift()
    retraining = get_retraining()
    models = get_models()
    model_metadata = get_model_metadata()
    top_model = get_top_model()
    pipeline = get_pipeline_status()
    self_correction = get_self_correction()
    pipeline_log = get_pipeline_log()
    evaluation = get_model_evaluation()
    version_management = get_version_management()

    return {
        "system": system,
        "anomalies": anomalies,
        "drift": drift,
        "retraining": retraining,
        "retraining_history": {
            "available": retraining.get("available", False),
            "records": retraining.get("records", []),
            "count": len(
                retraining.get("records", [])
            )
        },
        "models": models,
        "model_metadata": model_metadata,
        "top_model": top_model,
        "version_management": version_management,
        "pipeline": pipeline,
        "self_correction": self_correction,
        "pipeline_log": pipeline_log,
        "evaluation": evaluation,
        "summary": {
            "total_logs": system.get("total_logs"),
            "processed": system.get("logs_processed"),
            "normal": system.get("normal_logs"),
            "anomalies": system.get("anomalies_detected"),
            "total_drifts": drift.get("total_drifts", 0),
            "new_drift_count": drift.get(
                "new_drift_count",
                0
            ),
            "latest_record": pipeline.get(
                "latest_record"
            ),
            "last_checkpoint": pipeline.get(
                "checkpoint"
            ),
            "next_checkpoint": pipeline.get(
                "next_checkpoint"
            ),
            "retraining_status": (
                "COMPLETED"
                if retraining.get("available")
                else "NOT_TRIGGERED"
            ),
            "production_version": models.get(
                "current_production_version"
            ),
            "candidate_version": models.get(
                "latest_candidate_version"
            ),
            "last_decision": models.get(
                "last_decision"
            )
        },
        "timestamp": time.time()
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DASHBOARD DATA MODULE TEST")
    print("=" * 80)

    data = get_dashboard_data()

    print(
        json.dumps(
            data,
            indent=4,
            default=str
        )
    )

    print("=" * 80)
    print("DASHBOARD DATA MODULE WORKING.")
    print("=" * 80)

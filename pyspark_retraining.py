

# ============================================================
# PHASE 11 - NORMAL-TRAFFIC PYSPARK RETRAINING
# ============================================================
#
# Purpose:
#   1. Load ALL historical Parquet data
#   2. Select NORMAL traffic only
#   3. Validate the original 59 features
#   4. Create Spark DataFrame
#   5. Retrain Isolation Forest
#   6. Retrain Autoencoder
#   7. Calculate candidate AE threshold
#   8. Save candidate models separately
#   9. Save detailed training metadata
#  10. Publish live pipeline status for dashboard
#
# IMPORTANT:
#   Production models are NEVER modified here.
#
# Flow:
#
# ADWIN
#    ↓
# automatic_retraining_orchestrator.py
#    ↓
# pyspark_retraining.py
#    ↓
# candidate Isolation Forest + Autoencoder
#    ↓
# Phase 12 evaluation
#    ↓
# Phase 14 promotion / rejection
#
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
from sklearn.ensemble import IsolationForest
PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
import glob
import time
import pickle
import warnings
import json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# BASE PATH
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

PIPELINE_STATUS_PATH = os.path.join(
    PROCESSED_DIR,
    "automatic_pipeline_status.json"
)


# ============================================================
# PYSPARK PYTHON CONFIGURATION
# ============================================================

PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"


# ============================================================
# CANDIDATE MODEL OUTPUTS
# ============================================================

IF_OUTPUT = os.path.join(
    MODEL_DIR,
    "candidate_isolation_forest.pkl"
)

AE_OUTPUT = os.path.join(
    MODEL_DIR,
    "candidate_autoencoder.pth"
)

AE_THRESHOLD_OUTPUT = os.path.join(
    MODEL_DIR,
    "candidate_ae_threshold.pkl"
)

AE_METADATA_OUTPUT = os.path.join(
    MODEL_DIR,
    "candidate_ae_metadata.pkl"
)

CANDIDATE_METADATA_JSON = os.path.join(
    MODEL_DIR,
    "candidate_training_metadata.json"
)

REPORT_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "phase11_retraining_report.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURES = 59

IF_ESTIMATORS = 200

IF_CONTAMINATION = "auto"

IF_RANDOM_STATE = 42

AE_EPOCHS = 20

AE_BATCH_SIZE = 256

AE_LEARNING_RATE = 0.001

AE_THRESHOLD_PERCENTILE = 95

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# LIVE PIPELINE STATUS
# ============================================================

def save_pipeline_status(
    stage,
    message,
    progress=0,
    status="RUNNING",
    **extra
):

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    payload = {

        "pipeline": "SELF_CORRECTION",

        "stage": stage,

        "status": status,

        "progress_percent": progress,

        "message": message,

        "timestamp": time.time(),

        "updated_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    }

    payload.update(
        extra
    )

    try:

        temporary_path = (
            PIPELINE_STATUS_PATH
            + ".tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                indent=4
            )

        os.replace(
            temporary_path,
            PIPELINE_STATUS_PATH
        )

    except Exception as e:

        print(
            f"WARNING: Could not update "
            f"pipeline status: {e}"
        )


# ============================================================
# START
# ============================================================

print("=" * 80)

print(
    "TRAFFIC PYSPARK RETRAINING"
)

print("=" * 80)


save_pipeline_status(
    stage="PYSPARK_STARTED",
    message="PySpark retraining process started.",
    progress=0,
    status="RUNNING"
)


# ============================================================
# PYTHON INFORMATION
# ============================================================

print()

print(
    f"Python executable : {sys.executable}"
)

print(
    f"Python version    : "
    f"{sys.version.split()[0]}"
)


# ============================================================
# PYTORCH
# ============================================================

print(
    "\nLoading PyTorch..."
)

print(
    f"PyTorch version   : "
    f"{torch.__version__}"
)

CUDA_AVAILABLE = (
    torch.cuda.is_available()
)

print(
    f"CUDA available    : "
    f"{CUDA_AVAILABLE}"
)

DEVICE = torch.device(
    "cuda"
    if CUDA_AVAILABLE
    else "cpu"
)

print(
    f"Training device   : "
    f"{DEVICE}"
)


# ============================================================
# PYSPARK IMPORT
# ============================================================

print(
    "\nLoading PySpark..."
)

try:

    from pyspark.sql import SparkSession

    print(
        "PySpark imported successfully."
    )

except Exception as e:

    print(
        f"ERROR: PySpark import failed: {e}"
    )

    save_pipeline_status(
        stage="PYSPARK_FAILED",
        message=(
            "PySpark import failed."
        ),
        progress=0,
        status="FAILED",
        error=str(e)
    )

    sys.exit(1)


# ============================================================
# START SPARK
# ============================================================

print(
    "\nStarting Spark..."
)

spark = None

try:

    spark = (
        SparkSession
        .builder
        .appName(
            "NetworkIntrusionCandidateRetraining"
        )
        .master("local[*]")
        .config(
            "spark.driver.bindAddress",
            "127.0.0.1"
        )
        .config(
            "spark.driver.host",
            "127.0.0.1"
        )
        .config(
            "spark.sql.shuffle.partitions",
            "4"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print(
        "Spark started successfully."
    )

    print(
        f"Spark version: "
        f"{spark.version}"
    )

except Exception as e:

    print(
        f"ERROR: Spark startup failed: {e}"
    )

    save_pipeline_status(
        stage="PYSPARK_FAILED",
        message="Spark startup failed.",
        progress=0,
        status="FAILED",
        error=str(e)
    )

    sys.exit(1)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

try:

    print(
        "\n" + "=" * 80
    )

    print(
        "LOADING HISTORICAL DATA"
    )

    print(
        "=" * 80
    )

    save_pipeline_status(
        stage="LOADING_HISTORY",
        message=(
            "Loading all historical Parquet records."
        ),
        progress=10,
        status="RUNNING"
    )


    parquet_files = sorted(
        glob.glob(
            os.path.join(
                HISTORY_DIR,
                "*.parquet"
            )
        )
    )


    if not parquet_files:

        raise FileNotFoundError(
            f"No Parquet files found in "
            f"{HISTORY_DIR}"
        )


    print(
        f"Parquet files found: "
        f"{len(parquet_files)}"
    )


    frames = []


    for file in parquet_files:

        try:

            df = pd.read_parquet(
                file
            )

            if not df.empty:

                frames.append(
                    df
                )

        except Exception as e:

            print(
                f"WARNING: Could not read "
                f"{file}: {e}"
            )


    if not frames:

        raise ValueError(
            "No valid historical data found."
        )


    history = pd.concat(
        frames,
        ignore_index=True
    )


    print(
        f"Historical records loaded: "
        f"{len(history)}"
    )


    save_pipeline_status(
        stage="HISTORY_LOADED",
        message=(
            "Historical Parquet data loaded successfully."
        ),
        progress=20,
        status="RUNNING",
        parquet_files=len(parquet_files),
        historical_records=len(history)
    )


    # ========================================================
    # STATUS DISTRIBUTION
    # ========================================================

    if "status" not in history.columns:

        raise ValueError(
            "status column not found."
        )


    history["status"] = (
        history["status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    print(
        "\nHistorical status distribution:"
    )

    print(
        history["status"]
        .value_counts()
        .to_string()
    )


    normal_history = history[
        history["status"] == "NORMAL"
    ].copy()


    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTING NORMAL HISTORICAL TRAFFIC"
    )

    print(
        "=" * 80
    )


    print(
        f"Normal records selected: "
        f"{len(normal_history)}"
    )


    if normal_history.empty:

        raise ValueError(
            "No NORMAL historical records available."
        )


    save_pipeline_status(
        stage="NORMAL_TRAFFIC_SELECTED",
        message=(
            "NORMAL traffic selected for candidate training."
        ),
        progress=25,
        status="RUNNING",
        normal_training_records=len(
            normal_history
        ),
        total_historical_records=len(
            history
        ),
        anomaly_records=int(
            (
                history["status"]
                == "ANOMALY"
            ).sum()
        )
    )


    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATING FEATURE VECTORS"
    )

    print(
        "=" * 80
    )


    if "features" not in normal_history.columns:

        raise ValueError(
            "'features' column not found."
        )


    feature_vectors = []

    invalid_records = 0


    for features in normal_history[
        "features"
    ]:

        try:

            if hasattr(
                features,
                "tolist"
            ):

                features = (
                    features.tolist()
                )


            vector = np.asarray(
                features,
                dtype=np.float32
            )


            if (
                vector.ndim != 1
                or len(vector)
                != EXPECTED_FEATURES
                or not np.all(
                    np.isfinite(vector)
                )
            ):

                invalid_records += 1

                continue


            feature_vectors.append(
                vector
            )


        except Exception:

            invalid_records += 1


    print(
        f"Valid feature records  : "
        f"{len(feature_vectors)}"
    )

    print(
        f"Invalid feature records: "
        f"{invalid_records}"
    )


    if not feature_vectors:

        raise ValueError(
            "No valid feature vectors."
        )


    X = np.asarray(
        feature_vectors,
        dtype=np.float32
    )


    print(
        f"Training matrix shape: "
        f"{X.shape}"
    )


    if X.shape[1] != EXPECTED_FEATURES:

        raise ValueError(
            f"Feature count mismatch. "
            f"Expected {EXPECTED_FEATURES}, "
            f"received {X.shape[1]}."
        )


    save_pipeline_status(
        stage="TRAINING_DATA_READY",
        message=(
            "Training feature matrix prepared successfully."
        ),
        progress=30,
        status="RUNNING",
        training_records=len(X),
        feature_count=X.shape[1],
        invalid_records=invalid_records
    )


    # ========================================================
    # CREATE SPARK DATAFRAME
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CREATING SPARK DATAFRAME"
    )

    print(
        "=" * 80
    )


    spark_columns = [
        f"feature_{i}"
        for i in range(
            EXPECTED_FEATURES
        )
    ]


    spark_df = spark.createDataFrame(
        [
            tuple(
                float(value)
                for value in row
            )
            for row in X
        ],
        schema=spark_columns
    )


    spark_record_count = (
        spark_df.count()
    )


    print(
        f"Spark records: "
        f"{spark_record_count}"
    )

    print(
        f"Spark columns: "
        f"{len(spark_df.columns)}"
    )


    if (
        len(spark_df.columns)
        != EXPECTED_FEATURES
    ):

        raise ValueError(
            "Spark feature count verification failed."
        )


    print(
        "Spark DataFrame verification: PASSED"
    )


    save_pipeline_status(
        stage="SPARK_DATAFRAME_READY",
        message=(
            "Spark DataFrame verified successfully."
        ),
        progress=35,
        status="RUNNING",
        spark_records=int(
            spark_record_count
        ),
        feature_count=EXPECTED_FEATURES
    )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RETRAINING ISOLATION FOREST"
    )

    print(
        "=" * 80
    )


    print(
        f"Training records : {len(X)}"
    )

    print(
        f"Features         : "
        f"{X.shape[1]}"
    )

    print(
        f"Estimators       : "
        f"{IF_ESTIMATORS}"
    )


    save_pipeline_status(
        stage="TRAINING_ISOLATION_FOREST",
        message=(
            "Isolation Forest training is in progress."
        ),
        progress=40,
        status="RUNNING",
        model="Isolation Forest",
        training_records=len(X),
        feature_count=EXPECTED_FEATURES,
        estimators=IF_ESTIMATORS
    )


    if_start = time.perf_counter()


    isolation_forest = IsolationForest(

        n_estimators=IF_ESTIMATORS,

        contamination=IF_CONTAMINATION,

        random_state=IF_RANDOM_STATE,

        n_jobs=-1
    )


    print(
        "Training Isolation Forest..."
    )


    isolation_forest.fit(
        X
    )


    if_training_time = (
        time.perf_counter()
        - if_start
    )


    print(
        "Isolation Forest training complete."
    )

    print(
        f"Training time: "
        f"{if_training_time:.3f} seconds"
    )


    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    joblib.dump(
        isolation_forest,
        IF_OUTPUT
    )


    print(
        "Candidate Isolation Forest saved:"
    )

    print(
        IF_OUTPUT
    )


    save_pipeline_status(
        stage="ISOLATION_FOREST_COMPLETED",
        message=(
            "Isolation Forest candidate training completed."
        ),
        progress=50,
        status="RUNNING",
        model="Isolation Forest",
        training_records=len(X),
        feature_count=EXPECTED_FEATURES,
        estimators=IF_ESTIMATORS,
        training_time_seconds=round(
            if_training_time,
            3
        )
    )


    # ========================================================
    # AUTOENCODER
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RETRAINING AUTOENCODER"
    )

    print(
        "=" * 80
    )


    class Autoencoder(
        nn.Module
    ):

        def __init__(
            self,
            input_dim
        ):

            super().__init__()


            self.encoder = nn.Sequential(

                nn.Linear(
                    input_dim,
                    64
                ),

                nn.ReLU(),

                nn.Linear(
                    64,
                    32
                ),

                nn.ReLU(),

                nn.Linear(
                    32,
                    16
                )
            )


            self.decoder = nn.Sequential(

                nn.Linear(
                    16,
                    32
                ),

                nn.ReLU(),

                nn.Linear(
                    32,
                    64
                ),

                nn.ReLU(),

                nn.Linear(
                    64,
                    input_dim
                )
            )


        def forward(
            self,
            x
        ):

            encoded = (
                self.encoder(x)
            )

            decoded = (
                self.decoder(encoded)
            )

            return decoded


    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )


    dataset = TensorDataset(
        X_tensor
    )


    loader = DataLoader(

        dataset,

        batch_size=AE_BATCH_SIZE,

        shuffle=True
    )


    model = Autoencoder(
        EXPECTED_FEATURES
    ).to(
        DEVICE
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=AE_LEARNING_RATE
    )


    print(
        f"Training records : {len(X)}"
    )

    print(
        f"Features         : "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"Epochs           : "
        f"{AE_EPOCHS}"
    )

    print(
        f"Batch size       : "
        f"{AE_BATCH_SIZE}"
    )

    print(
        f"Learning rate    : "
        f"{AE_LEARNING_RATE}"
    )

    print(
        f"Device           : "
        f"{DEVICE}"
    )


    save_pipeline_status(
        stage="TRAINING_AUTOENCODER",
        message=(
            "Autoencoder training is in progress."
        ),
        progress=55,
        status="RUNNING",
        model="Autoencoder",
        training_records=len(X),
        feature_count=EXPECTED_FEATURES,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        device=str(DEVICE)
    )


    # ========================================================
    # TRAIN AUTOENCODER
    # ========================================================

    ae_start = time.perf_counter()


    best_loss = float(
        "inf"
    )


    for epoch in range(
        AE_EPOCHS
    ):

        model.train()

        epoch_loss = 0.0

        sample_count = 0


        for batch in loader:

            batch_x = batch[0].to(
                DEVICE
            )


            optimizer.zero_grad()


            reconstructed = model(
                batch_x
            )


            loss = criterion(
                reconstructed,
                batch_x
            )


            loss.backward()

            optimizer.step()


            current_batch_size = (
                batch_x.size(0)
            )


            epoch_loss += (
                loss.item()
                * current_batch_size
            )


            sample_count += (
                current_batch_size
            )


        average_loss = (
            epoch_loss
            / sample_count
        )


        if average_loss < best_loss:

            best_loss = (
                average_loss
            )


        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{AE_EPOCHS}"
            f" - loss: "
            f"{average_loss:.6f}"
        )


        # Progress from 55% to 70%
        epoch_progress = (
            55
            + int(
                (
                    (epoch + 1)
                    / AE_EPOCHS
                )
                * 15
            )
        )


        save_pipeline_status(
            stage="TRAINING_AUTOENCODER",
            message=(
                f"Autoencoder epoch "
                f"{epoch + 1}/{AE_EPOCHS}"
            ),
            progress=epoch_progress,
            status="RUNNING",
            model="Autoencoder",
            epoch=epoch + 1,
            total_epochs=AE_EPOCHS,
            current_loss=float(
                average_loss
            ),
            best_training_loss=float(
                best_loss
            ),
            training_records=len(X),
            feature_count=EXPECTED_FEATURES
        )


    ae_training_time = (
        time.perf_counter()
        - ae_start
    )


    print(
        "\nAutoencoder training complete."
    )

    print(
        f"Training time: "
        f"{ae_training_time:.3f} seconds"
    )

    print(
        f"Best training loss: "
        f"{best_loss:.6f}"
    )


    # ========================================================
    # AE THRESHOLD
    # ========================================================

    print(
        "\nCalculating Autoencoder threshold..."
    )


    save_pipeline_status(
        stage="CALCULATING_AE_THRESHOLD",
        message=(
            "Calculating Autoencoder reconstruction threshold."
        ),
        progress=72,
        status="RUNNING",
        model="Autoencoder"
    )


    model.eval()


    errors = []


    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            AE_BATCH_SIZE
        ):

            batch_x = (
                X_tensor[
                    start:
                    start + AE_BATCH_SIZE
                ]
                .to(DEVICE)
            )


            reconstructed = model(
                batch_x
            )


            batch_errors = torch.mean(

                (
                    reconstructed
                    - batch_x
                ) ** 2,

                dim=1
            )


            errors.extend(
                batch_errors
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    errors = np.asarray(
        errors,
        dtype=np.float64
    )


    new_threshold = float(
        np.percentile(
            errors,
            AE_THRESHOLD_PERCENTILE
        )
    )


    print(
        f"AE threshold percentile: "
        f"{AE_THRESHOLD_PERCENTILE}"
    )

    print(
        f"New AE threshold       : "
        f"{new_threshold:.12f}"
    )


    save_pipeline_status(
        stage="AE_THRESHOLD_COMPLETED",
        message=(
            "Autoencoder threshold calculated."
        ),
        progress=78,
        status="RUNNING",
        model="Autoencoder",
        ae_threshold=float(
            new_threshold
        ),
        threshold_percentile=(
            AE_THRESHOLD_PERCENTILE
        ),
        best_training_loss=float(
            best_loss
        )
    )


    # ========================================================
    # SAVE AUTOENCODER
    # ========================================================

    print(
        "\nSaving candidate Autoencoder..."
    )


    torch.save(
        model.state_dict(),
        AE_OUTPUT
    )


    joblib.dump(
        new_threshold,
        AE_THRESHOLD_OUTPUT
    )


    print(
        "Candidate Autoencoder saved:"
    )

    print(
        AE_OUTPUT
    )


    print(
        "Candidate AE threshold saved:"
    )

    print(
        AE_THRESHOLD_OUTPUT
    )


    # ========================================================
    # AE METADATA
    # ========================================================

    metadata = {

        "input_dim":
            EXPECTED_FEATURES,

        "architecture": [

            EXPECTED_FEATURES,

            64,

            32,

            16,

            32,

            64,

            EXPECTED_FEATURES
        ],

        "epochs":
            AE_EPOCHS,

        "batch_size":
            AE_BATCH_SIZE,

        "learning_rate":
            AE_LEARNING_RATE,

        "training_records":
            int(len(X)),

        "training_data":
            "NORMAL historical traffic",

        "feature_count":
            EXPECTED_FEATURES,

        "threshold_percentile":
            AE_THRESHOLD_PERCENTILE,

        "ae_threshold":
            float(new_threshold),

        "best_training_loss":
            float(best_loss),

        "training_time_seconds":
            float(ae_training_time),

        "device":
            str(DEVICE),

        "created_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "model_type":
            "candidate",

        "training_reason":
            "ADWIN/self-correction candidate retraining",

        "production_modified":
            False
    }


    joblib.dump(
        metadata,
        AE_METADATA_OUTPUT
    )


    print(
        "Candidate AE metadata saved:"
    )

    print(
        AE_METADATA_OUTPUT
    )


    # ========================================================
    # CANDIDATE MODEL METADATA
    # ========================================================

    candidate_metadata = {

        "model_type":
            "candidate",

        "training_records":
            int(len(X)),

        "feature_count":
            EXPECTED_FEATURES,

        "training_data":
            "NORMAL historical traffic",

        "total_historical_records":
            int(len(history)),

        "normal_records":
            int(len(normal_history)),

        "invalid_feature_records":
            int(invalid_records),

        "spark_records":
            int(spark_record_count),

        "isolation_forest": {

            "estimators":
                IF_ESTIMATORS,

            "contamination":
                IF_CONTAMINATION,

            "random_state":
                IF_RANDOM_STATE,

            "training_time_seconds":
                round(
                    if_training_time,
                    3
                )
        },

        "autoencoder": {

            "epochs":
                AE_EPOCHS,

            "batch_size":
                AE_BATCH_SIZE,

            "learning_rate":
                AE_LEARNING_RATE,

            "training_time_seconds":
                round(
                    ae_training_time,
                    3
                ),

            "best_training_loss":
                float(best_loss),

            "ae_threshold":
                float(new_threshold),

            "threshold_percentile":
                AE_THRESHOLD_PERCENTILE,

            "device":
                str(DEVICE)
        },

        "training_reason":
            "ADWIN drift triggered self-correction retraining",

        "production_modified":
            False,

        "created_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }


    with open(
        CANDIDATE_METADATA_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            candidate_metadata,
            f,
            indent=4
        )


    print(
        "Candidate training metadata saved:"
    )

    print(
        CANDIDATE_METADATA_JSON
    )


    # ========================================================
    # RETRAINING REPORT
    # ========================================================

    report = pd.DataFrame([

        {

            "model":
                "Isolation Forest",

            "model_type":
                "candidate",

            "training_records":
                len(X),

            "features":
                EXPECTED_FEATURES,

            "estimators":
                IF_ESTIMATORS,

            "epochs":
                None,

            "training_time_seconds":
                round(
                    if_training_time,
                    3
                ),

            "status":
                "CANDIDATE_READY"
        },

        {

            "model":
                "Autoencoder",

            "model_type":
                "candidate",

            "training_records":
                len(X),

            "features":
                EXPECTED_FEATURES,

            "estimators":
                None,

            "epochs":
                AE_EPOCHS,

            "training_time_seconds":
                round(
                    ae_training_time,
                    3
                ),

            "best_training_loss":
                float(best_loss),

            "ae_threshold":
                float(new_threshold),

            "status":
                "CANDIDATE_READY"
        }
    ])


    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )


    report.to_csv(
        REPORT_OUTPUT,
        index=False
    )


    # ========================================================
    # COMPLETED
    # ========================================================

    save_pipeline_status(
        stage="PYSPARK_COMPLETED",
        message=(
            "PySpark retraining completed. "
            "Candidate models are ready for evaluation."
        ),
        progress=100,
        status="COMPLETED",
        model="Candidate Models",
        training_records=len(X),
        feature_count=EXPECTED_FEATURES,
        isolation_forest_training_time=round(
            if_training_time,
            3
        ),
        autoencoder_training_time=round(
            ae_training_time,
            3
        ),
        ae_threshold=float(
            new_threshold
        ),
        best_training_loss=float(
            best_loss
        ),
        candidate_isolation_forest=IF_OUTPUT,
        candidate_autoencoder=AE_OUTPUT,
        candidate_ae_threshold=AE_THRESHOLD_OUTPUT,
        candidate_metadata=CANDIDATE_METADATA_JSON,
        report=REPORT_OUTPUT
    )


    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RETRAINING COMPLETE"
    )

    print(
        "=" * 80
    )


    print(
        f"Historical records     : "
        f"{len(history)}"
    )

    print(
        f"Normal training records: "
        f"{len(X)}"
    )

    print(
        f"Feature count           : "
        f"{X.shape[1]}"
    )

    print(
        f"Isolation Forest time   : "
        f"{if_training_time:.3f}s"
    )

    print(
        f"Autoencoder time        : "
        f"{ae_training_time:.3f}s"
    )

    print(
        f"New AE threshold        : "
        f"{new_threshold:.6f}"
    )

    print(
        f"Best AE loss            : "
        f"{best_loss:.6f}"
    )


    print(
        "\nCandidate models:"
    )

    print(
        f"  IF        : {IF_OUTPUT}"
    )

    print(
        f"  AE        : {AE_OUTPUT}"
    )

    print(
        f"  Threshold : "
        f"{AE_THRESHOLD_OUTPUT}"
    )

    print(
        f"  AE Meta   : "
        f"{AE_METADATA_OUTPUT}"
    )

    print(
        f"  Metadata  : "
        f"{CANDIDATE_METADATA_JSON}"
    )

    print(
        f"\nReport: "
        f"{REPORT_OUTPUT}"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Production models were NOT modified."
    )

    print(
        "Candidate models are ready for Phase 12 evaluation."
    )

    print(
        "Comparing OLD vs NEW."
    )

    print(
        "Yet to decide promotion or rejection."
    )


    print(
        "=" * 80
    )


except Exception as e:

    # ========================================================
    # FAILURE STATUS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RETRAINING FAILED"
    )

    print(
        "=" * 80
    )

    print(
        f"Error: {e}"
    )

    traceback.print_exc()


    save_pipeline_status(
        stage="PYSPARK_FAILED",
        message=(
            "PySpark retraining failed."
        ),
        progress=0,
        status="FAILED",
        error=str(e)
    )


    raise


finally:

    # ========================================================
    # STOP SPARK
    # ========================================================

    if spark is not None:

        try:

            spark.stop()

            print(
                "\nSpark stopped."
            )

        except Exception as e:

            print(
                f"WARNING: Spark stop failed: "
                f"{e}"
            )


print(
    "RETRAINING finished."
)

print(
    "=" * 80
)
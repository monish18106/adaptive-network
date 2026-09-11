import torch
import torch.nn as nn

import os
import sys
import glob
import time
import shutil
import json
import joblib
import numpy as np
import pandas as pd



# ============================================================
# PHASE 12 - MODEL EVALUATION AND PROMOTION
# ============================================================

print("=" * 90)
print("PHASE 12 - OLD VS CANDIDATE MODEL EVALUATION")
print("=" * 90)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

IF_WEIGHT = 0.2
AE_WEIGHT = 0.8

# Minimum acceptable performance
# Candidate must not significantly degrade F1.
F1_TOLERANCE = 0.02

# Candidate must not increase false alarms by more than this.
FAR_TOLERANCE = 0.02


# ============================================================
# MODEL FILES
# ============================================================

OLD_IF_PATH = os.path.join(
    MODEL_DIR,
    "isolation_forest.pkl"
)

OLD_AE_PATH = os.path.join(
    MODEL_DIR,
    "autoencoder.pth"
)

OLD_THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "ae_threshold.pkl"
)

OLD_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "ae_metadata.pkl"
)


CANDIDATE_IF_PATH = os.path.join(
    MODEL_DIR,
    "candidate_isolation_forest.pkl"
)

CANDIDATE_AE_PATH = os.path.join(
    MODEL_DIR,
    "candidate_autoencoder.pth"
)

CANDIDATE_THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "candidate_ae_threshold.pkl"
)

CANDIDATE_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "candidate_ae_metadata.pkl"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    OLD_IF_PATH,
    OLD_AE_PATH,
    OLD_THRESHOLD_PATH,
    OLD_METADATA_PATH,
    CANDIDATE_IF_PATH,
    CANDIDATE_AE_PATH,
    CANDIDATE_THRESHOLD_PATH,
    CANDIDATE_METADATA_PATH,
]

print("\nChecking model files...")

for path in required_files:

    if not os.path.exists(path):
        print(f"ERROR: Missing file:")
        print(path)
        sys.exit(1)

    print(f"OK: {os.path.basename(path)}")


# ============================================================
# AUTOENCODER ARCHITECTURE
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING HISTORICAL EVALUATION DATA")
print("=" * 90)

parquet_files = sorted(
    glob.glob(
        os.path.join(
            HISTORY_DIR,
            "*.parquet"
        )
    )
)

print(f"Parquet files found: {len(parquet_files)}")

if not parquet_files:
    print("ERROR: No Parquet history found.")
    sys.exit(1)


frames = []

for filepath in parquet_files:

    try:

        df = pd.read_parquet(filepath)

        if "features" in df.columns:
            frames.append(df)

    except Exception as e:

        print(
            f"WARNING: Could not read {filepath}: {e}"
        )


if not frames:

    print("ERROR: No Parquet files contain features.")
    sys.exit(1)


history = pd.concat(
    frames,
    ignore_index=True
)

print(
    f"Historical records loaded: {len(history)}"
)


# ============================================================
# VALIDATE FEATURES
# ============================================================

print("\nValidating feature vectors...")

valid_features = []
valid_status = []

for _, row in history.iterrows():

    features = row["features"]

    if not isinstance(
        features,
        (list, tuple, np.ndarray)
    ):
        continue

    if len(features) != 59:
        continue

    try:

        vector = np.asarray(
            features,
            dtype=np.float32
        )

        if not np.all(
            np.isfinite(vector)
        ):
            continue

        valid_features.append(vector)
        valid_status.append(
            str(row.get("status", "UNKNOWN"))
        )

    except Exception:
        continue


if not valid_features:

    print("ERROR: No valid feature vectors.")
    sys.exit(1)


X = np.asarray(
    valid_features,
    dtype=np.float32
)

statuses = np.asarray(
    valid_status
)

print(
    f"Valid records: {len(X)}"
)

print(
    f"Feature count: {X.shape[1]}"
)

print(
    f"Evaluation matrix: {X.shape}"
)


# ============================================================
# AVAILABLE HISTORICAL STATUS
# ============================================================

normal_mask = statuses == "NORMAL"
anomaly_mask = statuses == "ANOMALY"

print("\nHistorical status distribution:")

print(
    pd.Series(statuses).value_counts()
)


# ============================================================
# LOAD OLD MODELS
# ============================================================

print("\n" + "=" * 90)
print("LOADING OLD PRODUCTION MODELS")
print("=" * 90)

old_if = joblib.load(
    OLD_IF_PATH
)

old_metadata = joblib.load(
    OLD_METADATA_PATH
)

old_threshold = float(
    joblib.load(
        OLD_THRESHOLD_PATH
    )
)

old_input_dim = int(
    old_metadata["input_dim"]
)

print(
    f"Old AE input dimension: {old_input_dim}"
)

print(
    f"Old AE threshold: {old_threshold}"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Evaluation device: {device}"
)


old_ae = Autoencoder(
    old_input_dim
).to(device)

old_ae.load_state_dict(
    torch.load(
        OLD_AE_PATH,
        map_location=device
    )
)

old_ae.eval()


# ============================================================
# LOAD CANDIDATE MODELS
# ============================================================

print("\n" + "=" * 90)
print("LOADING CANDIDATE MODELS")
print("=" * 90)

candidate_if = joblib.load(
    CANDIDATE_IF_PATH
)

candidate_metadata = joblib.load(
    CANDIDATE_METADATA_PATH
)

candidate_threshold = float(
    joblib.load(
        CANDIDATE_THRESHOLD_PATH
    )
)

candidate_input_dim = int(
    candidate_metadata["input_dim"]
)

print(
    f"Candidate AE input dimension: "
    f"{candidate_input_dim}"
)

print(
    f"Candidate AE threshold: "
    f"{candidate_threshold}"
)


candidate_ae = Autoencoder(
    candidate_input_dim
).to(device)

candidate_ae.load_state_dict(
    torch.load(
        CANDIDATE_AE_PATH,
        map_location=device
    )
)

candidate_ae.eval()


# ============================================================
# FEATURE DIMENSION CHECK
# ============================================================

if old_input_dim != X.shape[1]:

    print(
        "\nERROR: Old model feature dimension "
        "does not match historical data."
    )

    sys.exit(1)


if candidate_input_dim != X.shape[1]:

    print(
        "\nERROR: Candidate model feature dimension "
        "does not match historical data."
    )

    sys.exit(1)


# ============================================================
# SCORE FUNCTION
# ============================================================

def calculate_scores(
    if_model,
    ae_model,
    ae_threshold
):

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    if_scores = -if_model.score_samples(X)

    if_scores = np.asarray(
        if_scores,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Autoencoder
    # --------------------------------------------------------

    ae_errors = []

    batch_size = 512

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X)
            )

            batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            reconstructed = ae_model(
                batch
            )

            errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            ae_errors.extend(
                errors.cpu().numpy()
            )

    ae_errors = np.asarray(
        ae_errors,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Normalize AE score
    # --------------------------------------------------------

    ae_scores = np.minimum(
        ae_errors / ae_threshold,
        1.0
    )

    # --------------------------------------------------------
    # Normalize IF score
    #
    # The existing live system uses IF score directly.
    # --------------------------------------------------------

    if_scores = np.clip(
        if_scores,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    fusion_scores = (
        IF_WEIGHT * if_scores
        +
        AE_WEIGHT * ae_scores
    )

    # --------------------------------------------------------
    # Use the locked production threshold.
    #
    # The fusion threshold is 0.85.
    # --------------------------------------------------------

    predictions = (
        fusion_scores >= 0.85
    )

    return (
        if_scores,
        ae_errors,
        ae_scores,
        fusion_scores,
        predictions
    )


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(predictions):

    actual = anomaly_mask

    predicted = predictions

    true_normal = np.sum(
        (~actual) & (~predicted)
    )

    false_alarm = np.sum(
        (~actual) & predicted
    )

    missed_attack = np.sum(
        actual & (~predicted)
    )

    true_attack = np.sum(
        actual & predicted
    )

    total = len(actual)

    accuracy = (
        true_normal + true_attack
    ) / total

    detection_rate = (
        true_attack /
        max(
            true_attack + missed_attack,
            1
        )
    )

    false_alarm_rate = (
        false_alarm /
        max(
            false_alarm + true_normal,
            1
        )
    )

    precision = (
        true_attack /
        max(
            true_attack + false_alarm,
            1
        )
    )

    f1 = (
        2 * precision * detection_rate /
        max(
            precision + detection_rate,
            1e-12
        )
    )

    return {
        "accuracy": float(accuracy),
        "true_normal": int(true_normal),
        "false_alarm": int(false_alarm),
        "missed_attack": int(missed_attack),
        "true_attack": int(true_attack),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "precision": float(precision),
        "f1": float(f1)
    }


# ============================================================
# EVALUATE OLD MODEL
# ============================================================

print("\n" + "=" * 90)
print("EVALUATING OLD PRODUCTION MODEL")
print("=" * 90)

start_time = time.time()

(
    old_if_scores,
    old_ae_errors,
    old_ae_scores,
    old_fusion_scores,
    old_predictions
) = calculate_scores(
    old_if,
    old_ae,
    old_threshold
)

old_metrics = calculate_metrics(
    old_predictions
)

old_eval_time = time.time() - start_time

print(
    f"Evaluation time: {old_eval_time:.2f} seconds"
)

print("\nOLD MODEL RESULTS")

for key, value in old_metrics.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# EVALUATE CANDIDATE MODEL
# ============================================================

print("\n" + "=" * 90)
print("EVALUATING CANDIDATE MODEL")
print("=" * 90)

start_time = time.time()

(
    candidate_if_scores,
    candidate_ae_errors,
    candidate_ae_scores,
    candidate_fusion_scores,
    candidate_predictions
) = calculate_scores(
    candidate_if,
    candidate_ae,
    candidate_threshold
)

candidate_metrics = calculate_metrics(
    candidate_predictions
)

candidate_eval_time = (
    time.time() - start_time
)

print(
    f"Evaluation time: "
    f"{candidate_eval_time:.2f} seconds"
)

print("\nCANDIDATE MODEL RESULTS")

for key, value in candidate_metrics.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# COMPARE OLD VS NEW
# ============================================================

print("\n" + "=" * 90)
print("OLD VS CANDIDATE COMPARISON")
print("=" * 90)

f1_change = (
    candidate_metrics["f1"]
    -
    old_metrics["f1"]
)

far_change = (
    candidate_metrics["false_alarm_rate"]
    -
    old_metrics["false_alarm_rate"]
)

accuracy_change = (
    candidate_metrics["accuracy"]
    -
    old_metrics["accuracy"]
)

detection_change = (
    candidate_metrics["detection_rate"]
    -
    old_metrics["detection_rate"]
)

precision_change = (
    candidate_metrics["precision"]
    -
    old_metrics["precision"]
)


print(
    f"Accuracy change      : {accuracy_change:+.4f}"
)

print(
    f"Detection change     : {detection_change:+.4f}"
)

print(
    f"Precision change     : {precision_change:+.4f}"
)

print(
    f"F1 change            : {f1_change:+.4f}"
)

print(
    f"False Alarm change   : {far_change:+.4f}"
)


# ============================================================
# PROMOTION DECISION
# ============================================================

print("\n" + "=" * 90)
print("MODEL PROMOTION DECISION")
print("=" * 90)


# Candidate is acceptable when:
#
# 1. F1 does not degrade beyond tolerance
# 2. False alarm rate does not increase beyond tolerance
#
# This prevents blindly replacing the old model.

f1_ok = (
    candidate_metrics["f1"]
    >=
    old_metrics["f1"]
    -
    F1_TOLERANCE
)

far_ok = (
    candidate_metrics["false_alarm_rate"]
    <=
    old_metrics["false_alarm_rate"]
    +
    FAR_TOLERANCE
)


# Strong improvement is preferred.
#
# If candidate improves F1 OR improves detection while
# maintaining acceptable false alarms, it is promoted.

strong_improvement = (
    candidate_metrics["f1"]
    >
    old_metrics["f1"]
    and
    far_ok
)


safe_maintenance = (
    f1_ok
    and
    far_ok
)


if strong_improvement:

    decision = "ACCEPTED"

    reason = (
        "Candidate improved F1 while maintaining "
        "an acceptable false alarm rate."
    )

elif safe_maintenance:

    decision = "ACCEPTED"

    reason = (
        "Candidate safely maintains performance "
        "within the configured tolerance."
    )

else:

    decision = "REJECTED"

    reason = (
        "Candidate did not satisfy the required "
        "performance criteria. Production model retained."
    )


print(
    f"\nDecision: {decision}"
)

print(
    f"Reason  : {reason}"
)


# ============================================================
# VERSION INFORMATION
# ============================================================

version_file = os.path.join(
    MODEL_DIR,
    "model_version.json"
)

current_version = 1

if os.path.exists(version_file):

    try:

        with open(
            version_file,
            "r",
            encoding="utf-8"
        ) as f:

            version_data = json.load(f)

            current_version = int(
                version_data.get(
                    "production_version",
                    1
                )
            )

    except Exception:

        current_version = 1


if decision == "ACCEPTED":

    new_version = current_version + 1

else:

    new_version = current_version


# ============================================================
# ARCHIVE DIRECTORY
# ============================================================

ARCHIVE_DIR = os.path.join(
    MODEL_DIR,
    "archive"
)

CANDIDATE_ARCHIVE_DIR = os.path.join(
    MODEL_DIR,
    "candidates",
    f"version_{new_version}"
)

os.makedirs(
    ARCHIVE_DIR,
    exist_ok=True
)

os.makedirs(
    CANDIDATE_ARCHIVE_DIR,
    exist_ok=True
)


# ============================================================
# PROMOTION
# ============================================================

if decision == "ACCEPTED":

    print("\n" + "=" * 90)
    print("PROMOTING CANDIDATE MODEL")
    print("=" * 90)

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    # --------------------------------------------------------
    # Backup current production models
    # --------------------------------------------------------

    backup_dir = os.path.join(
        ARCHIVE_DIR,
        f"version_{current_version}_{timestamp}"
    )

    os.makedirs(
        backup_dir,
        exist_ok=True
    )

    shutil.copy2(
        OLD_IF_PATH,
        os.path.join(
            backup_dir,
            "isolation_forest.pkl"
        )
    )

    shutil.copy2(
        OLD_AE_PATH,
        os.path.join(
            backup_dir,
            "autoencoder.pth"
        )
    )

    shutil.copy2(
        OLD_THRESHOLD_PATH,
        os.path.join(
            backup_dir,
            "ae_threshold.pkl"
        )
    )

    shutil.copy2(
        OLD_METADATA_PATH,
        os.path.join(
            backup_dir,
            "ae_metadata.pkl"
        )

    )

    print(
        f"Old production model archived:"
    )

    print(
        backup_dir
    )

    # --------------------------------------------------------
    # Replace production with candidate
    # --------------------------------------------------------

    shutil.copy2(
        CANDIDATE_IF_PATH,
        OLD_IF_PATH
    )

    shutil.copy2(
        CANDIDATE_AE_PATH,
        OLD_AE_PATH
    )

    shutil.copy2(
        CANDIDATE_THRESHOLD_PATH,
        OLD_THRESHOLD_PATH
    )

    shutil.copy2(
        CANDIDATE_METADATA_PATH,
        OLD_METADATA_PATH
    )

    print(
        "Candidate promoted to production."
    )

else:

    print("\nProduction models were NOT changed.")

    # --------------------------------------------------------
    # Preserve rejected candidate
    # --------------------------------------------------------

    shutil.copy2(
        CANDIDATE_IF_PATH,
        os.path.join(
            CANDIDATE_ARCHIVE_DIR,
            "candidate_isolation_forest.pkl"
        )
    )

    shutil.copy2(
        CANDIDATE_AE_PATH,
        os.path.join(
            CANDIDATE_ARCHIVE_DIR,
            "candidate_autoencoder.pth"
        )
    )

    shutil.copy2(
        CANDIDATE_THRESHOLD_PATH,
        os.path.join(
            CANDIDATE_ARCHIVE_DIR,
            "candidate_ae_threshold.pkl"
        )
    )

    shutil.copy2(
        CANDIDATE_METADATA_PATH,
        os.path.join(
            CANDIDATE_ARCHIVE_DIR,
            "candidate_ae_metadata.pkl"
        )
    )

    print(
        "Rejected candidate preserved at:"
    )

    print(
        CANDIDATE_ARCHIVE_DIR
    )


# ============================================================
# SAVE MODEL METADATA
# ============================================================

metadata = {

    "production_version": new_version,

    "previous_production_version":
        current_version,

    "decision": decision,

    "decision_reason": reason,

    "timestamp": time.time(),

    "evaluation_records": int(len(X)),

    "feature_count": int(X.shape[1]),

    "normal_records": int(
        np.sum(normal_mask)
    ),

    "anomaly_records": int(
        np.sum(anomaly_mask)
    ),

    "if_weight": IF_WEIGHT,

    "ae_weight": AE_WEIGHT,

    "fusion_threshold": 0.85,

    "old_ae_threshold": old_threshold,

    "candidate_ae_threshold":
        candidate_threshold,

    "old_metrics": old_metrics,

    "candidate_metrics":
        candidate_metrics,

    "metric_changes": {

        "accuracy":
            accuracy_change,

        "detection_rate":
            detection_change,

        "precision":
            precision_change,

        "f1":
            f1_change,

        "false_alarm_rate":
            far_change
    },

    "f1_tolerance":
        F1_TOLERANCE,

    "false_alarm_tolerance":
        FAR_TOLERANCE,

    "trigger":
        "PHASE_11_CANDIDATE_EVALUATION",

    "status":
        "PRODUCTION"
        if decision == "ACCEPTED"
        else "REJECTED"
}


metadata_path = os.path.join(
    PROCESSED_DIR,
    "phase12_model_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "OLD_PRODUCTION",
        **old_metrics
    },
    {
        "model": "CANDIDATE",
        **candidate_metrics
    }
])

comparison_path = os.path.join(
    PROCESSED_DIR,
    "phase12_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE DECISION REPORT
# ============================================================

report = pd.DataFrame([
    {
        "production_version_before":
            current_version,

        "production_version_after":
            new_version,

        "decision":
            decision,

        "reason":
            reason,

        "evaluation_records":
            len(X),

        "old_f1":
            old_metrics["f1"],

        "candidate_f1":
            candidate_metrics["f1"],

        "old_accuracy":
            old_metrics["accuracy"],

        "candidate_accuracy":
            candidate_metrics["accuracy"],

        "old_detection_rate":
            old_metrics["detection_rate"],

        "candidate_detection_rate":
            candidate_metrics["detection_rate"],

        "old_false_alarm_rate":
            old_metrics["false_alarm_rate"],

        "candidate_false_alarm_rate":
            candidate_metrics["false_alarm_rate"],

        "old_precision":
            old_metrics["precision"],

        "candidate_precision":
            candidate_metrics["precision"],

        "if_weight":
            IF_WEIGHT,

        "ae_weight":
            AE_WEIGHT,

        "fusion_threshold":
            0.85,

        "old_ae_threshold":
            old_threshold,

        "candidate_ae_threshold":
            candidate_threshold,

        "timestamp":
            time.time()
    }
])

report_path = os.path.join(
    PROCESSED_DIR,
    "phase12_promotion_report.csv"
)

report.to_csv(
    report_path,
    index=False
)


# ============================================================
# SAVE VERSION FILE
# ============================================================

version_data = {

    "production_version":
        new_version,

    "last_decision":
        decision,

    "last_update":
        time.time(),

    "if_weight":
        IF_WEIGHT,

    "ae_weight":
        AE_WEIGHT,

    "fusion_threshold":
        0.85
}

with open(
    version_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        version_data,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("PHASE 12 COMPLETE")
print("=" * 90)

print(
    f"Decision              : {decision}"
)

print(
    f"Production version    : {new_version}"
)

print(
    f"Evaluation records    : {len(X)}"
)

print(
    f"Old F1                : "
    f"{old_metrics['f1']:.4f}"
)

print(
    f"Candidate F1          : "
    f"{candidate_metrics['f1']:.4f}"
)

print(
    f"Old False Alarm Rate  : "
    f"{old_metrics['false_alarm_rate']:.4f}"
)

print(
    f"Candidate FAR         : "
    f"{candidate_metrics['false_alarm_rate']:.4f}"
)

print(
    f"\nComparison report:"
)

print(
    comparison_path
)

print(
    f"\nPromotion report:"
)

print(
    report_path
)

print(
    f"\nMetadata:"
)

print(
    metadata_path
)

if decision == "ACCEPTED":

    print(
        "\nNEW CANDIDATE MODEL IS NOW PRODUCTION."
    )

    print(
        "Previous production model is archived."
    )

else:

    print(
        "\nOLD PRODUCTION MODEL REMAINS ACTIVE."
    )

    print(
        "Rejected candidate model was preserved."
    )

print(
    "\nNo rejected model was deleted."
)

print(
    "=" * 90
)
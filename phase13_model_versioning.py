import os
import json
import shutil
import time
from pathlib import Path


# ============================================================
# PHASE 13 - MODEL VERSIONING & REGISTRY
# ============================================================

print("=" * 90)
print("MODEL VERSIONING & MODEL REGISTRY")
print("=" * 90)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path("models/saved_models")

PRODUCTION_DIR = BASE_DIR / "production"
CANDIDATES_DIR = BASE_DIR / "candidates"
ARCHIVE_DIR = BASE_DIR / "archive"

PRODUCTION_V1_DIR = PRODUCTION_DIR / "v1"
CANDIDATE_V2_DIR = CANDIDATES_DIR / "v2"

REGISTRY_FILE = BASE_DIR / "model_registry.json"

PHASE12_METADATA = Path(
    "data/processed/phase12_model_metadata.json"
)

PHASE12_COMPARISON = Path(
    "data/processed/phase12_model_comparison.csv"
)

PHASE12_PROMOTION = Path(
    "data/processed/phase12_promotion_report.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    PRODUCTION_DIR,
    CANDIDATES_DIR,
    ARCHIVE_DIR,
    PRODUCTION_V1_DIR,
    CANDIDATE_V2_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER
# ============================================================

def copy_file(source, destination):
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileNotFoundError(
            f"Required file not found: {source}"
        )

    shutil.copy2(source, destination)

    print(
        f"Copied:\n"
        f"  {source}\n"
        f"  -> {destination}"
    )


# ============================================================
# PRODUCTION MODEL FILES
# ============================================================

production_files = {
    "isolation_forest.pkl": BASE_DIR / "isolation_forest.pkl",
    "autoencoder.pth": BASE_DIR / "autoencoder.pth",
    "ae_threshold.pkl": BASE_DIR / "ae_threshold.pkl",
    "ae_metadata.pkl": BASE_DIR / "ae_metadata.pkl",
}


print("\n" + "=" * 90)
print("ORGANIZING PRODUCTION MODEL")
print("=" * 90)

for filename, source in production_files.items():

    destination = PRODUCTION_V1_DIR / filename

    copy_file(source, destination)


# ============================================================
# LOAD PHASE 12 METADATA
# ============================================================

phase12_metadata = {}

if PHASE12_METADATA.exists():

    print("\nLoading Phase 12 metadata...")

    with open(
        PHASE12_METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        phase12_metadata = json.load(f)

    print("Phase 12 metadata loaded.")

else:

    print(
        "WARNING: Phase 12 metadata file not found."
    )


# ============================================================
# FIND CURRENT CANDIDATE
# ============================================================

print("\n" + "=" * 90)
print("LOCATING CANDIDATE MODEL")
print("=" * 90)


# Phase 12 already preserved the rejected candidate
# under candidates/version_1.

OLD_CANDIDATE_DIR = CANDIDATES_DIR / "version_1"


candidate_sources = {
    "isolation_forest.pkl":
        OLD_CANDIDATE_DIR / "candidate_isolation_forest.pkl",

    "autoencoder.pth":
        OLD_CANDIDATE_DIR / "candidate_autoencoder.pth",

    "ae_threshold.pkl":
        OLD_CANDIDATE_DIR / "candidate_ae_threshold.pkl",

    "ae_metadata.pkl":
        OLD_CANDIDATE_DIR / "candidate_ae_metadata.pkl",
}


# If version_1 directory does not contain the candidate,
# fall back to the root candidate files.

if not all(
    path.exists()
    for path in candidate_sources.values()
):

    print(
        "Candidate version_1 directory is incomplete."
    )

    print(
        "Using root candidate model files instead."
    )

    candidate_sources = {
        "isolation_forest.pkl":
            BASE_DIR / "candidate_isolation_forest.pkl",

        "autoencoder.pth":
            BASE_DIR / "candidate_autoencoder.pth",

        "ae_threshold.pkl":
            BASE_DIR / "candidate_ae_threshold.pkl",

        "ae_metadata.pkl":
            BASE_DIR / "candidate_ae_metadata.pkl",
    }


# ============================================================
# COPY CANDIDATE TO VERSION 2
# ============================================================

print("\nCreating Candidate Version 2...")

for filename, source in candidate_sources.items():

    destination = CANDIDATE_V2_DIR / filename

    copy_file(source, destination)


# ============================================================
# READ DECISION
# ============================================================

decision = phase12_metadata.get(
    "decision",
    "UNKNOWN"
)

decision_reason = phase12_metadata.get(
    "decision_reason",
    "No decision reason available."
)


candidate_metrics = phase12_metadata.get(
    "candidate_metrics",
    {}
)

old_metrics = phase12_metadata.get(
    "old_metrics",
    {}
)

metric_changes = phase12_metadata.get(
    "metric_changes",
    {}
)


# ============================================================
# CANDIDATE METADATA
# ============================================================

candidate_metadata = {
    "version": 2,

    "model_type": "candidate",

    "status": decision,

    "training_reason": (
        phase12_metadata.get(
            "trigger",
            "ADWIN_DRIFT"
        )
    ),

    "created_at": time.time(),

    "feature_count": phase12_metadata.get(
        "feature_count",
        59
    ),

    "evaluation_records": phase12_metadata.get(
        "evaluation_records",
        0
    ),

    "normal_records": phase12_metadata.get(
        "normal_records",
        0
    ),

    "anomaly_records": phase12_metadata.get(
        "anomaly_records",
        0
    ),

    "if_weight": phase12_metadata.get(
        "if_weight",
        0.2
    ),

    "ae_weight": phase12_metadata.get(
        "ae_weight",
        0.8
    ),

    "fusion_threshold": phase12_metadata.get(
        "fusion_threshold",
        0.85
    ),

    "old_ae_threshold": phase12_metadata.get(
        "old_ae_threshold"
    ),

    "candidate_ae_threshold": phase12_metadata.get(
        "candidate_ae_threshold"
    ),

    "old_metrics": old_metrics,

    "candidate_metrics": candidate_metrics,

    "metric_changes": metric_changes,

    "decision": decision,

    "decision_reason": decision_reason,

    "phase12_metadata": str(
        PHASE12_METADATA
    ),

    "phase12_comparison": str(
        PHASE12_COMPARISON
    ),

    "phase12_promotion_report": str(
        PHASE12_PROMOTION
    ),
}


candidate_metadata_file = (
    CANDIDATE_V2_DIR / "model_metadata.json"
)


with open(
    candidate_metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        candidate_metadata,
        f,
        indent=4
    )


print(
    f"\nCandidate metadata saved:\n"
    f"{candidate_metadata_file}"
)


# ============================================================
# PRODUCTION METADATA
# ============================================================

production_metadata = {
    "version": 1,

    "model_type": "production",

    "status": "ACTIVE",

    "created_at": time.time(),

    "source": "Original production model",

    "feature_count": phase12_metadata.get(
        "feature_count",
        59
    ),

    "if_weight": phase12_metadata.get(
        "if_weight",
        0.2
    ),

    "ae_weight": phase12_metadata.get(
        "ae_weight",
        0.8
    ),

    "fusion_threshold": phase12_metadata.get(
        "fusion_threshold",
        0.85
    ),

    "ae_threshold": phase12_metadata.get(
        "old_ae_threshold"
    ),
}


production_metadata_file = (
    PRODUCTION_V1_DIR / "model_metadata.json"
)


with open(
    production_metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        production_metadata,
        f,
        indent=4
    )


print(
    f"Production metadata saved:\n"
    f"{production_metadata_file}"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

registry = {
    "current_production_version": 1,

    "latest_candidate_version": 2,

    "last_decision": decision,

    "last_updated": time.time(),

    "models": [
        {
            "version": 1,
            "model_type": "production",
            "status": "ACTIVE",
            "path": str(
                PRODUCTION_V1_DIR
            ),
            "reason": "Original production model",
            "metrics": old_metrics
        },

        {
            "version": 2,
            "model_type": "candidate",
            "status": decision,
            "path": str(
                CANDIDATE_V2_DIR
            ),
            "reason": decision_reason,
            "metrics": candidate_metrics,
            "metric_changes": metric_changes
        }
    ]
}


# ============================================================
# SAVE REGISTRY
# ============================================================

with open(
    REGISTRY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        registry,
        f,
        indent=4
    )


print(
    f"\nModel registry saved:\n"
    f"{REGISTRY_FILE}"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 90)
print("FINAL VERIFICATION")
print("=" * 90)


required_paths = [
    PRODUCTION_V1_DIR / "isolation_forest.pkl",
    PRODUCTION_V1_DIR / "autoencoder.pth",
    PRODUCTION_V1_DIR / "ae_threshold.pkl",
    PRODUCTION_V1_DIR / "ae_metadata.pkl",
    PRODUCTION_V1_DIR / "model_metadata.json",

    CANDIDATE_V2_DIR / "isolation_forest.pkl",
    CANDIDATE_V2_DIR / "autoencoder.pth",
    CANDIDATE_V2_DIR / "ae_threshold.pkl",
    CANDIDATE_V2_DIR / "ae_metadata.pkl",
    CANDIDATE_V2_DIR / "model_metadata.json",

    REGISTRY_FILE,
]


all_ok = True


for path in required_paths:

    if path.exists():

        print(
            f"OK     {path}"
        )

    else:

        print(
            f"MISSING {path}"
        )

        all_ok = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 90)

if all_ok:

    print("COMPLETED!!!")
    print()
    print("Production model : v1 ACTIVE")
    print(
        f"Candidate model  : v2 {decision}"
    )
    print()
    print(
        "No existing model files were deleted."
    )
    print(
        "No production model was replaced."
    )
    print(
        "Model registry created successfully."
    )

else:

    print("INCOMPLETE")
    print(
        "One or more required files are missing."
    )

print("=" * 90)
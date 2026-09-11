import os
import json
import time
import shutil


# ============================================================
# PHASE 13 - MODEL VERSIONING & METADATA
# ============================================================

BASE_DIR = "models/saved_models"

CANDIDATE_DIR = os.path.join(
    BASE_DIR,
    "candidates"
)

ARCHIVE_DIR = os.path.join(
    BASE_DIR,
    "archive"
)

REGISTRY_FILE = os.path.join(
    BASE_DIR,
    "model_registry.json"
)

PHASE12_METADATA = (
    "data/processed/phase12_model_metadata.json"
)

PHASE12_REPORT = (
    "data/processed/phase12_promotion_report.csv"
)

MODEL_VERSION_FILE = os.path.join(
    BASE_DIR,
    "model_version.json"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CANDIDATE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)


print("=" * 80)
print("PHASE 13 - MODEL VERSIONING & METADATA")
print("=" * 80)


# ============================================================
# LOAD CURRENT PRODUCTION VERSION
# ============================================================

production_version = 1

if os.path.exists(MODEL_VERSION_FILE):

    try:

        with open(
            MODEL_VERSION_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            version_data = json.load(f)

        production_version = int(
            version_data.get(
                "production_version",
                1
            )
        )

    except Exception as e:

        print(
            f"WARNING: Could not read model_version.json: {e}"
        )


print(
    f"Current production version: v{production_version}"
)


# ============================================================
# LOAD PHASE 12 METADATA
# ============================================================

phase12_metadata = {}

if os.path.exists(PHASE12_METADATA):

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
# DETERMINE CANDIDATE VERSION
# ============================================================

candidate_versions = []

for name in os.listdir(CANDIDATE_DIR):

    if name.startswith("version_"):

        try:

            number = int(
                name.replace(
                    "version_",
                    ""
                )
            )

            candidate_versions.append(number)

        except ValueError:

            pass


if candidate_versions:

    candidate_version = max(
        candidate_versions
    )

else:

    candidate_version = 1


candidate_path = os.path.join(
    CANDIDATE_DIR,
    f"version_{candidate_version}"
)


print(
    f"Candidate version: v{candidate_version}"
)

print(
    f"Candidate path: {candidate_path}"
)


# ============================================================
# COLLECT CANDIDATE FILES
# ============================================================

candidate_files = []

if os.path.exists(candidate_path):

    for root, dirs, files in os.walk(
        candidate_path
    ):

        for filename in files:

            relative_path = os.path.relpath(
                os.path.join(root, filename),
                candidate_path
            )

            candidate_files.append(
                relative_path
            )


print(
    f"Candidate files preserved: {len(candidate_files)}"
)


# ============================================================
# BUILD VERSION METADATA
# ============================================================

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


candidate_metadata = {

    "version": candidate_version,

    "model_type": [
        "Isolation Forest",
        "Autoencoder"
    ],

    "status": phase12_metadata.get(
        "status",
        "UNKNOWN"
    ),

    "decision": phase12_metadata.get(
        "decision",
        "UNKNOWN"
    ),

    "decision_reason": phase12_metadata.get(
        "decision_reason",
        ""
    ),

    "created_at": phase12_metadata.get(
        "timestamp",
        time.time()
    ),

    "training_reason": phase12_metadata.get(
        "trigger",
        "UNKNOWN"
    ),

    "evaluation_records": phase12_metadata.get(
        "evaluation_records",
        0
    ),

    "feature_count": phase12_metadata.get(
        "feature_count",
        59
    ),

    "normal_records": phase12_metadata.get(
        "normal_records",
        0
    ),

    "anomaly_records": phase12_metadata.get(
        "anomaly_records",
        0
    ),

    "configuration": {

        "if_weight": phase12_metadata.get(
            "if_weight"
        ),

        "ae_weight": phase12_metadata.get(
            "ae_weight"
        ),

        "fusion_threshold": phase12_metadata.get(
            "fusion_threshold"
        ),

        "old_ae_threshold": phase12_metadata.get(
            "old_ae_threshold"
        ),

        "candidate_ae_threshold": phase12_metadata.get(
            "candidate_ae_threshold"
        )
    },

    "candidate_metrics": candidate_metrics,

    "production_metrics": old_metrics,

    "metric_changes": metric_changes,

    "candidate_files": candidate_files,

    "candidate_path": candidate_path,

    "preserved": True,

    "production_version": production_version
}


# ============================================================
# LOAD EXISTING REGISTRY
# ============================================================

registry = {

    "current_production_version":
        production_version,

    "versions": []
}


if os.path.exists(REGISTRY_FILE):

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            existing_registry = json.load(f)

        if isinstance(
            existing_registry,
            dict
        ):

            registry = existing_registry

    except Exception as e:

        print(
            f"WARNING: Could not load registry: {e}"
        )


# ============================================================
# ENSURE VERSION LIST EXISTS
# ============================================================

if "versions" not in registry:

    registry["versions"] = []


# ============================================================
# ADD ORIGINAL PRODUCTION MODEL
# ============================================================

production_exists = any(

    item.get("version") == production_version
    and item.get("status") == "PRODUCTION"

    for item in registry["versions"]
)


if not production_exists:

    original_metadata = {

        "version": production_version,

        "model_type": [
            "Isolation Forest",
            "Autoencoder"
        ],

        "status": "PRODUCTION",

        "created_at": None,

        "training_reason":
            "ORIGINAL_PRODUCTION_MODEL",

        "production": True,

        "candidate": False,

        "preserved": True,

        "model_files": [

            "models/saved_models/"
            "isolation_forest.pkl",

            "models/saved_models/"
            "autoencoder.pth",

            "models/saved_models/"
            "ae_threshold.pkl",

            "models/saved_models/"
            "ae_metadata.pkl"
        ]
    }

    registry["versions"].append(
        original_metadata
    )


# ============================================================
# REMOVE DUPLICATE CANDIDATE REGISTRY ENTRY
# ============================================================

registry["versions"] = [

    item

    for item in registry["versions"]

    if not (

        item.get("version") == candidate_version
        and item.get("candidate") is True

    )

]


# ============================================================
# ADD CANDIDATE VERSION
# ============================================================

candidate_registry_entry = {

    **candidate_metadata,

    "candidate": True,

    "production": False
}


registry["versions"].append(
    candidate_registry_entry
)


# ============================================================
# UPDATE CURRENT PRODUCTION
# ============================================================

registry[
    "current_production_version"
] = production_version


registry[
    "last_updated"
] = time.time()


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


# ============================================================
# SAVE CANDIDATE METADATA INSIDE VERSION DIRECTORY
# ============================================================

if os.path.exists(candidate_path):

    candidate_metadata_file = os.path.join(
        candidate_path,
        "metadata.json"
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
        f"Candidate metadata saved:"
    )

    print(
        candidate_metadata_file
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("PHASE 13 VERSIONING COMPLETE")
print("=" * 80)

print(
    f"Production version : v{production_version}"
)

print(
    f"Candidate version  : v{candidate_version}"
)

print(
    "Candidate status   : "
    f"{candidate_metadata['status']}"
)

print(
    "Decision            : "
    f"{candidate_metadata['decision']}"
)

print(
    "Candidate preserved : YES"
)

print()
print(
    "Model registry:"
)

print(
    REGISTRY_FILE
)

print()
print(
    "Production model was NOT modified."
)

print(
    "Rejected candidate was NOT deleted."
)

print(
    "Phase 14 will implement automatic "
    "promotion and rollback."
)

print("=" * 80)
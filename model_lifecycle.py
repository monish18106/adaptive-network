import os
import json
import shutil
import time


# ============================================================
# PHASE 14 - AUTOMATIC MODEL PROMOTION & ROLLBACK
# ============================================================
#
# Pipeline:
#
# ADWIN
#   ↓
# PySpark Retraining
#   ↓
# Model Evaluation
#   ↓
# Model Lifecycle
#
# Responsibilities:
#   1. Read Phase 12 evaluation decision.
#   2. Verify candidate model.
#   3. Promote candidate if ACCEPTED.
#   4. Preserve current production if REJECTED.
#   5. Archive previous production when promoting.
#   6. Update model registry.
#   7. Write lifecycle status for dashboard.
#
# Production model is NEVER replaced directly.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

PRODUCTION_DIR = os.path.join(
    MODEL_DIR,
    "production"
)

CANDIDATES_DIR = os.path.join(
    MODEL_DIR,
    "candidates"
)

ARCHIVE_DIR = os.path.join(
    MODEL_DIR,
    "archive"
)

REGISTRY_FILE = os.path.join(
    MODEL_DIR,
    "model_registry.json"
)

PHASE12_METADATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "phase12_model_metadata.json"
)

PIPELINE_STATUS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "automatic_pipeline_status.json"
)


# ============================================================
# REQUIRED MODEL FILES
# ============================================================

MODEL_FILES = [
    "isolation_forest.pkl",
    "autoencoder.pth",
    "ae_threshold.pkl",
    "ae_metadata.pkl",
    "model_metadata.json",
]


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
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


def ensure_directories():

    os.makedirs(
        PRODUCTION_DIR,
        exist_ok=True
    )

    os.makedirs(
        CANDIDATES_DIR,
        exist_ok=True
    )

    os.makedirs(
        ARCHIVE_DIR,
        exist_ok=True
    )


def safe_value(value):

    """
    Convert values into JSON-safe values.
    """

    if value is None:
        return None

    try:

        if hasattr(
            value,
            "item"
        ):

            return value.item()

    except Exception:
        pass

    return value


# ============================================================
# DASHBOARD PIPELINE STATUS
# ============================================================

def update_pipeline_status(
    status=None,
    stage=None,
    message=None,
    lifecycle=None,
    decision=None,
    production_version=None,
    candidate_version=None,
    reason=None,
    evaluation=None
):

    """
    Update automatic_pipeline_status.json.

    This file is consumed by dashboard_data.py.
    """

    existing = {}

    if os.path.exists(
        PIPELINE_STATUS_FILE
    ):

        try:

            existing = load_json(
                PIPELINE_STATUS_FILE
            )

        except Exception:

            existing = {}

    if status is not None:

        existing["status"] = status

    if stage is not None:

        existing["stage"] = stage

    if message is not None:

        existing["message"] = message

    if lifecycle is not None:

        existing["lifecycle"] = lifecycle

    if decision is not None:

        existing["decision"] = decision

    if production_version is not None:

        existing[
            "production_version"
        ] = production_version

    if candidate_version is not None:

        existing[
            "candidate_version"
        ] = candidate_version

    if reason is not None:

        existing["reason"] = reason

    if evaluation is not None:

        existing[
            "evaluation"
        ] = evaluation

    existing[
        "last_updated"
    ] = time.time()

    save_json(
        PIPELINE_STATUS_FILE,
        existing
    )


# ============================================================
# FIND PRODUCTION VERSION
# ============================================================

def find_active_production():

    versions = []

    if not os.path.exists(
        PRODUCTION_DIR
    ):

        return None

    for name in os.listdir(
        PRODUCTION_DIR
    ):

        if not name.startswith("v"):
            continue

        path = os.path.join(
            PRODUCTION_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        try:

            version = int(
                name[1:]
            )

            versions.append(
                (
                    version,
                    path
                )
            )

        except ValueError:

            continue

    if not versions:

        return None

    versions.sort(
        reverse=True
    )

    return versions[0]


# ============================================================
# FIND LATEST CANDIDATE
# ============================================================

def find_latest_candidate():

    versions = []

    if not os.path.exists(
        CANDIDATES_DIR
    ):

        return None

    for name in os.listdir(
        CANDIDATES_DIR
    ):

        if not name.startswith("v"):
            continue

        path = os.path.join(
            CANDIDATES_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        try:

            version = int(
                name[1:]
            )

            versions.append(
                (
                    version,
                    path
                )
            )

        except ValueError:

            continue

    if not versions:

        return None

    versions.sort(
        reverse=True
    )

    return versions[0]


# ============================================================
# REGISTRY
# ============================================================

def create_or_load_registry():

    if os.path.exists(
        REGISTRY_FILE
    ):

        try:

            registry = load_json(
                REGISTRY_FILE
            )

            if "models" not in registry:

                registry["models"] = []

            return registry

        except Exception:

            pass

    return {
        "current_production_version": None,
        "previous_production_version": None,
        "latest_candidate_version": None,
        "last_decision": None,
        "last_updated": time.time(),
        "models": []
    }


# ============================================================
# UPDATE EXISTING MODEL STATUS
# ============================================================

def update_registry_model_status(
    registry,
    version,
    status
):

    for model in registry.get(
        "models",
        []
    ):

        if model.get(
            "version"
        ) == version:

            model["status"] = status

            model[
                "status_updated_at"
            ] = time.time()


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(
    registry,
    version,
    model_type,
    status,
    metadata=None,
    metrics=None,
    reason=None
):

    entry = {

        "version":
            version,

        "model_type":
            model_type,

        "status":
            status,

        "timestamp":
            time.time()
    }

    if metadata:

        entry[
            "metadata"
        ] = metadata

    if metrics:

        entry[
            "metrics"
        ] = metrics

    if reason:

        entry[
            "reason"
        ] = reason

    registry[
        "models"
    ].append(
        entry
    )


# ============================================================
# START
# ============================================================

print(
    "=" * 90
)

print(
    "PHASE 14 - AUTOMATIC MODEL PROMOTION & ROLLBACK"
)

print(
    "=" * 90
)


ensure_directories()


# ============================================================
# INITIAL PIPELINE STATUS
# ============================================================

update_pipeline_status(

    status="RUNNING",

    stage="LIFECYCLE",

    message="Model lifecycle decision started.",

    lifecycle="RUNNING",

    decision="PENDING"
)


# ============================================================
# LOAD REGISTRY
# ============================================================

registry = create_or_load_registry()


# ============================================================
# LOAD PHASE 12
# ============================================================

if not os.path.exists(
    PHASE12_METADATA
):

    message = (
        "Phase 12 metadata not found. "
        "Model lifecycle cannot continue."
    )

    update_pipeline_status(

        status="ERROR",

        stage="LIFECYCLE",

        message=message,

        lifecycle="ERROR",

        decision="ERROR",

        reason=message
    )

    print()
    print("ERROR:", message)
    print(PHASE12_METADATA)

    raise SystemExit(1)


phase12 = load_json(
    PHASE12_METADATA
)


decision = str(
    phase12.get(
        "decision",
        "UNKNOWN"
    )
).upper()


decision_reason = phase12.get(
    "decision_reason",
    "No decision reason provided."
)


print()
print(
    "Phase 12 decision:",
    decision
)


# ============================================================
# FIND CURRENT PRODUCTION
# ============================================================

production = find_active_production()


if production:

    current_version, current_dir = production

    print(
        "Current production:",
        f"v{current_version}"
    )

else:

    current_version = None
    current_dir = None

    print(
        "Current production: NONE"
    )


# ============================================================
# FIND CANDIDATE
# ============================================================

candidate = find_latest_candidate()


if candidate:

    candidate_version, candidate_dir = candidate

    print(
        "Latest candidate:",
        f"v{candidate_version}"
    )

else:

    candidate_version = None
    candidate_dir = None

    print(
        "Latest candidate: NONE"
    )


# ============================================================
# COMMON EVALUATION INFORMATION
# ============================================================

candidate_metrics = phase12.get(
    "candidate_metrics",
    {}
)

old_metrics = phase12.get(
    "old_metrics",
    {}
)

metric_changes = phase12.get(
    "metric_changes",
    {}
)


evaluation_summary = {

    "evaluation_records":
        safe_value(
            phase12.get(
                "evaluation_records"
            )
        ),

    "old_metrics":
        old_metrics,

    "candidate_metrics":
        candidate_metrics,

    "metric_changes":
        metric_changes,

    "evaluation_time":
        safe_value(
            phase12.get(
                "evaluation_time"
            )
        )
}


# ============================================================
# REJECTION
# ============================================================

if decision != "ACCEPTED":

    print()
    print(
        "=" * 90
    )

    print(
        "PROMOTION DECISION"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Decision: REJECTED"
    )

    print(
        "Reason:",
        decision_reason
    )

    print(
        "Production model will remain unchanged."
    )

    if candidate_version:

        print(
            f"Candidate v{candidate_version} "
            "remains preserved."
        )

    registry[
        "current_production_version"
    ] = current_version

    registry[
        "latest_candidate_version"
    ] = candidate_version

    registry[
        "last_decision"
    ] = "REJECTED"

    registry[
        "last_updated"
    ] = time.time()

    if current_version is None:

        registry[
            "previous_production_version"
        ] = None

    save_json(
        REGISTRY_FILE,
        registry
    )


    # --------------------------------------------------------
    # DASHBOARD STATUS
    # --------------------------------------------------------

    update_pipeline_status(

        status="COMPLETED",

        stage="LIFECYCLE",

        message=(
            "Candidate rejected. "
            "Production model retained."
        ),

        lifecycle="REJECTED",

        decision="REJECTED",

        production_version=
            current_version,

        candidate_version=
            candidate_version,

        reason=
            decision_reason,

        evaluation=
            evaluation_summary
    )


    print()
    print(
        "Registry updated."
    )

    print()
    print(
        "Phase 14 completed safely."
    )

    print(
        "=" * 90
    )

    raise SystemExit(0)


# ============================================================
# ACCEPTED CANDIDATE
# ============================================================

if candidate_dir is None:

    message = (
        "Phase 12 accepted the candidate, "
        "but candidate model directory was not found."
    )

    update_pipeline_status(

        status="ERROR",

        stage="LIFECYCLE",

        message=message,

        lifecycle="ERROR",

        decision="ACCEPTED",

        production_version=
            current_version,

        candidate_version=
            None,

        reason=message,

        evaluation=
            evaluation_summary
    )

    print()
    print(
        "ERROR:",
        message
    )

    raise SystemExit(1)


# ============================================================
# VERIFY CANDIDATE
# ============================================================

print()
print(
    "=" * 90
)

print(
    "VERIFYING CANDIDATE MODEL"
)

print(
    "=" * 90
)


update_pipeline_status(

    status="RUNNING",

    stage="LIFECYCLE",

    message="Verifying candidate model files.",

    lifecycle="VERIFYING",

    decision="ACCEPTED",

    production_version=
        current_version,

    candidate_version=
        candidate_version,

    evaluation=
        evaluation_summary
)


for filename in MODEL_FILES:

    path = os.path.join(
        candidate_dir,
        filename
    )

    if not os.path.exists(path):

        message = (
            f"Required candidate file missing: "
            f"{filename}"
        )

        update_pipeline_status(

            status="ERROR",

            stage="LIFECYCLE",

            message=message,

            lifecycle="ERROR",

            decision="ACCEPTED",

            production_version=
                current_version,

            candidate_version=
                candidate_version,

            reason=message,

            evaluation=
                evaluation_summary
        )

        print(
            "MISSING:",
            filename
        )

        print(
            "Promotion cancelled."
        )

        raise SystemExit(1)

    print(
        "OK:",
        filename
    )


# ============================================================
# DETERMINE NEW PRODUCTION VERSION
# ============================================================

if current_version is None:

    new_version = 1

else:

    new_version = (
        current_version + 1
    )


new_production_dir = os.path.join(

    PRODUCTION_DIR,

    f"v{new_version}"
)


print()
print(
    "New production version:",
    f"v{new_version}"
)


# ============================================================
# ARCHIVE CURRENT PRODUCTION
# ============================================================

if current_dir:

    print()
    print(
        "=" * 90
    )

    print(
        "ARCHIVING CURRENT PRODUCTION"
    )

    print(
        "=" * 90
    )

    archive_version_dir = os.path.join(

        ARCHIVE_DIR,

        f"v{current_version}"
    )


    if os.path.exists(
        archive_version_dir
    ):

        archive_version_dir = os.path.join(

            ARCHIVE_DIR,

            f"v{current_version}_"
            f"{int(time.time())}"
        )


    shutil.copytree(

        current_dir,

        archive_version_dir
    )


    print(
        f"Production v{current_version} archived."
    )


# ============================================================
# PROMOTION
# ============================================================

print()
print(
    "=" * 90
)

print(
    "PROMOTING CANDIDATE"
)

print(
    "=" * 90
)


update_pipeline_status(

    status="RUNNING",

    stage="LIFECYCLE",

    message=(
        f"Promoting candidate v{candidate_version} "
        f"to production v{new_version}."
    ),

    lifecycle="PROMOTING",

    decision="ACCEPTED",

    production_version=
        current_version,

    candidate_version=
        candidate_version,

    evaluation=
        evaluation_summary
)


os.makedirs(
    new_production_dir,
    exist_ok=True
)


for filename in MODEL_FILES:

    source = os.path.join(
        candidate_dir,
        filename
    )

    destination = os.path.join(
        new_production_dir,
        filename
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"Copied: {filename}"
    )


# ============================================================
# PRODUCTION METADATA
# ============================================================

production_metadata = {

    "version":
        new_version,

    "status":
        "ACTIVE",

    "previous_production_version":
        current_version,

    "candidate_version":
        candidate_version,

    "promotion_decision":
        "ACCEPTED",

    "promotion_reason":
        decision_reason,

    "promoted_at":
        time.time(),

    "evaluation_records":
        phase12.get(
            "evaluation_records"
        ),

    "old_metrics":
        old_metrics,

    "candidate_metrics":
        candidate_metrics,

    "metric_changes":
        metric_changes,

    "if_weight":
        phase12.get(
            "if_weight"
        ),

    "ae_weight":
        phase12.get(
            "ae_weight"
        ),

    "fusion_threshold":
        phase12.get(
            "fusion_threshold"
        ),

    "old_ae_threshold":
        phase12.get(
            "old_ae_threshold"
        ),

    "candidate_ae_threshold":
        phase12.get(
            "candidate_ae_threshold"
        ),

    "trigger":
        phase12.get(
            "trigger"
        )
}


save_json(

    os.path.join(

        new_production_dir,

        "model_metadata.json"
    ),

    production_metadata
)


# ============================================================
# UPDATE REGISTRY
# ============================================================

# Mark previous production as archived.

if current_version is not None:

    update_registry_model_status(

        registry,

        current_version,

        "ARCHIVED"
    )


# Mark candidate as promoted.

update_registry_model_status(

    registry,

    candidate_version,

    "PROMOTED"
)


# Register new production version.

register_model(

    registry,

    new_version,

    "Isolation Forest + Autoencoder",

    "ACTIVE",

    metadata=production_metadata,

    metrics=candidate_metrics,

    reason=decision_reason
)


registry[
    "current_production_version"
] = new_version


registry[
    "previous_production_version"
] = current_version


registry[
    "latest_candidate_version"
] = candidate_version


registry[
    "last_decision"
] = "ACCEPTED"


registry[
    "last_updated"
] = time.time()


save_json(
    REGISTRY_FILE,
    registry
)


# ============================================================
# FINAL DASHBOARD STATUS
# ============================================================

update_pipeline_status(

    status="COMPLETED",

    stage="LIFECYCLE",

    message=(
        f"Candidate v{candidate_version} "
        f"promoted to production v{new_version}."
    ),

    lifecycle="PROMOTED",

    decision="ACCEPTED",

    production_version=
        new_version,

    candidate_version=
        candidate_version,

    reason=
        decision_reason,

    evaluation=
        evaluation_summary
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print(
    "=" * 90
)

print(
    "MODEL PROMOTION SUCCESSFUL"
)

print(
    "=" * 90
)

print()
print(
    f"Previous production : "
    f"v{current_version}"
    if current_version is not None
    else
    "Previous production : NONE"
)

print(
    f"Candidate            : "
    f"v{candidate_version}"
)

print(
    f"New production       : "
    f"v{new_version}"
)

print(
    "Decision             : ACCEPTED"
)

print(
    "Status               : ACTIVE"
)

print(
    "Reason               :",
    decision_reason
)

print()
print(
    "Registry updated."
)

print(
    "Pipeline status updated."
)

print()
print(
    "Phase 14 completed successfully."
)

print(
    "=" * 90
)
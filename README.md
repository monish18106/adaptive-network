# 🛡️ Adaptive Network Intrusion Detection System

> A **real-time, self-correcting** network intrusion detection platform that streams live traffic through Apache Kafka, scores it with dual ML models (Isolation Forest + Autoencoder), detects concept drift via ADWIN, automatically retrains with PySpark, and visualizes everything on a Flask dashboard.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Data Pipeline Flow](#data-pipeline-flow)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
  - [Kafka Layer](#1-kafka-layer)
  - [Feature Processing (Flink)](#2-feature-processing-flink-layer)
  - [Inference Pipeline](#3-inference-pipeline)
  - [Parquet Storage](#4-parquet-storage)
  - [Drift Detection (ADWIN)](#5-drift-detection-adwin)
  - [PySpark Retraining](#6-pyspark-retraining-phase-11)
  - [Model Evaluation](#7-model-evaluation-phase-12)
  - [Model Versioning](#8-model-versioning-phase-13)
  - [Model Lifecycle](#9-model-lifecycle--promotionrollback-phase-14)
  - [Self-Correction Controller](#10-self-correction-controller)
  - [Automatic Retraining Orchestrator](#11-automatic-retraining-orchestrator-phase-16)
  - [Dashboard](#12-dashboard-phase-15)
- [Machine Learning Models](#machine-learning-models)
- [Kafka Topics](#kafka-topics)
- [Data Directory Layout](#data-directory-layout)
- [Model Artifacts](#model-artifacts)
- [Notebooks](#notebooks)
- [Configuration Reference](#configuration-reference)
- [Execution Order](#execution-order)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Dataset](#dataset)

---

## System Overview

This system implements a **closed-loop, adaptive** intrusion detection pipeline with the following capabilities:

| Capability | Description |
|---|---|
| **Real-Time Streaming** | Network traffic is streamed record-by-record through Kafka topics |
| **Dual-Model Scoring** | Each record is scored by both an Isolation Forest and an Autoencoder |
| **Weighted Fusion** | Scores are combined using a weighted fusion (20% IF + 80% AE) |
| **Threshold Alerting** | Records exceeding the anomaly threshold (0.85) are flagged |
| **Parquet Archival** | All alert records are persisted in batch Parquet files for historical analysis |
| **Concept Drift Detection** | ADWIN monitors anomaly scores for statistical drift at 10,000-record checkpoints |
| **Automatic Retraining** | When drift is confirmed, PySpark retrains candidate models on all historical normal traffic |
| **Champion/Challenger Evaluation** | Old (production) vs. new (candidate) models are compared on F1, precision, recall, and FAR |
| **Promotion / Rejection / Rollback** | Candidates are promoted only if they meet strict quality gates; production is archived and recoverable |
| **Model Registry & Versioning** | All model versions are tracked in a JSON registry with timestamps and metadata |
| **Live Dashboard** | A Flask web dashboard provides real-time visibility into every stage of the pipeline |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE NETWORK IDS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────┐                   │
│   │  Kafka   │───▶│   Feature    │───▶│   Inference     │                   │
│   │ Producer │    │  Processor   │    │   Worker        │                   │
│   │          │    │  (Flink)     │    │ (IF + AE)       │                   │
│   └──────────┘    └──────────────┘    └────────┬────────┘                   │
│    raw_traffic     processed_features          │ anomaly_scores             │
│                                                ▼                            │
│                                       ┌─────────────────┐                   │
│                                       │  Fusion Worker  │                   │
│                                       │ (0.2·IF + 0.8·AE)│                  │
│                                       └────────┬────────┘                   │
│                                                │ fusion_scores              │
│                                                ▼                            │
│                                       ┌─────────────────┐                   │
│                                       │  Alert Worker   │                   │
│                                       │ (threshold 0.85)│                   │
│                                       └────────┬────────┘                   │
│                                                │ alerts                     │
│                                    ┌───────────┼───────────┐                │
│                                    ▼           ▼           ▼                │
│                            ┌────────────┐ ┌─────────┐ ┌──────────┐          │
│                            │  Parquet   │ │  Alert  │ │ Dashboard│          │
│                            │  Storage   │ │ Consumer│ │  (Flask) │          │
│                            └─────┬──────┘ └─────────┘ └──────────┘          │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐       │
│  │                  SELF-CORRECTION LOOP                            │       │
│  │                                                                   │       │
│  │   Parquet History ──▶ ADWIN (10k checkpoint) ──▶ Drift?          │       │
│  │                                                    │              │       │
│  │                                              YES   │   NO         │       │
│  │                                                ▼   └──▶ Wait     │       │
│  │                                          PySpark Retrain          │       │
│  │                                                ▼                  │       │
│  │                                       Model Evaluation            │       │
│  │                                                ▼                  │       │
│  │                                     Promote / Reject / Rollback   │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Flow

```
CICIDS2017 CSV ──▶ producer.py ──▶ [raw_traffic]
                                        │
                                        ▼
                                  processor.py   (scales features via scaler.pkl)
                                        │
                                        ▼
                                [processed_features]
                                        │
                                        ▼
                              inference_worker.py  (IF score + AE reconstruction error)
                                        │
                                        ▼
                                [anomaly_scores]
                                        │
                                        ▼
                              fusion_worker.py     (weighted: 0.2 × IF + 0.8 × AE)
                                        │
                                        ▼
                                [fusion_scores]
                                        │
                                        ▼
                              alert_worker.py      (threshold ≥ 0.85 → ANOMALY)
                                        │
                                        ▼
                                  [alerts]
                                    │    │
                    ┌───────────────┘    └───────────────┐
                    ▼                                    ▼
            parquet_storage.py                   alert_consumer.py
            (batch=100 → Parquet)                (real-time viewer)
                    │
                    ▼
          data/history/*.parquet
                    │
                    ▼
          ADWIN (δ=0.002, checkpoint=10k)
                    │
              drift detected?
               ╱          ╲
             YES           NO
              │             └──▶ continue monitoring
              ▼
       PySpark Retraining (candidate IF + AE)
              ▼
       Model Evaluation (old vs candidate)
              ▼
       Lifecycle (promote / reject / rollback)
```

---

## Project Structure

```
adaptive-network/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── kafka config.txt                       # Kafka setup & execution commands
├── Final file-by-file checklist.txt       # Development checklist
│
├── kafka/                                 # Kafka producers & consumers
│   ├── producers/
│   │   └── producer.py                    # Streams CICIDS2017 CSV → raw_traffic topic
│   └── consumers/
│       └── consumer_test.py               # Test consumer for raw_traffic
│
├── flink/                                 # Feature processing layer
│   ├── config.py                          # Kafka broker & topic configuration
│   ├── processor.py                       # Scales raw records → processed_features topic
│   ├── feature_processor.py               # Standalone preprocessing function
│   ├── flink_job.py                       # Flink job definition
│   ├── processed_consumer.py              # Test consumer for processed_features
│   ├── test.py                            # Flink test script
│   ├── checkpoints/                       # Flink checkpoints
│   └── jobs/                              # Flink job JARs
│
├── inference/                             # ML inference workers
│   ├── inference_worker.py                # IF + AE scoring → anomaly_scores topic
│   ├── fusion_worker.py                   # Weighted score fusion → fusion_scores topic
│   ├── alert_worker.py                    # Threshold classification → alerts topic
│   ├── fusion_consumer.py                 # Test consumer for fusion_scores
│   ├── score_consumer.py                  # Test consumer for anomaly_scores
│   └── alert_consumer.py                  # Real-time alert display consumer
│
├── parquet_storage.py                     # Kafka alerts → batch Parquet files
├── adwin_drift_detection.py               # ADWIN concept drift detection
├── pyspark_retraining.py                  # Phase 11 – PySpark candidate model training
├── pyspark_check.py                       # Utility to inspect Parquet history data
├── model_evaluation.py                    # Phase 12 – Old vs candidate evaluation
├── model_versioning.py                    # Phase 13 – Version tracking & metadata
├── phase13_model_versioning.py            # Phase 13 – Registry & archive management
├── model_lifecycle.py                     # Phase 14 – Promotion / rejection / rollback
├── self_correction_controller.py          # Automated ADWIN → retrain → evaluate → lifecycle
├── automatic_retraining_orchestrator.py   # Phase 16 – Full automatic pipeline orchestrator
│
├── dashboard/                             # Flask web dashboard (Phase 15)
│   ├── app.py                             # Flask application & API routes
│   ├── dashboard_data.py                  # Main data-source module for all API endpoints
│   ├── dashboard_data1.py                 # Alternate/backup dashboard data module
│   ├── templates/
│   │   └── dashboard.html                 # Dashboard HTML template
│   └── static/
│       ├── dashboard.css                  # Dashboard stylesheet
│       └── dashboard.js                   # Dashboard JavaScript (AJAX polling)
│
├── data/                                  # All data artifacts
│   ├── raw/                               # Original CICIDS2017 CSV files
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   ├── Wednesday-workingHours.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   │   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── processed/                         # Cleaned datasets & pipeline state files
│   │   ├── CICIDS2017_Cleaned.csv         # ~883 MB cleaned dataset
│   │   ├── model_dataset.csv              # ~770 MB model-ready dataset
│   │   ├── adwin_drift_results.csv        # ADWIN drift analysis output
│   │   ├── adwin_state.json               # ADWIN checkpoint state
│   │   ├── automatic_pipeline_status.json  # Current pipeline stage
│   │   ├── automatic_pipeline_history.json # Pipeline run history
│   │   ├── automatic_retraining_state.json # Retraining state
│   │   ├── retraining_status.json         # Retraining status
│   │   ├── self_correction_state.json     # Self-correction checkpoint
│   │   ├── retraining_pipeline.log        # Pipeline log file
│   │   ├── phase11_retraining_report.csv  # PySpark retraining report
│   │   ├── phase12_model_comparison.csv   # Old vs candidate comparison
│   │   ├── phase12_model_metadata.json    # Evaluation metadata
│   │   ├── phase12_promotion_report.csv   # Promotion decision report
│   │   └── pyspark_retraining_report.csv  # PySpark training metrics
│   ├── features/                          # Extracted feature sets
│   └── history/                           # Parquet alert history (~709 files)
│
├── models/                                # ML model artifacts
│   ├── saved_models/                      # Active model storage
│   │   ├── isolation_forest.pkl           # Production Isolation Forest
│   │   ├── autoencoder.pth                # Production Autoencoder (PyTorch)
│   │   ├── ae_threshold.pkl               # Production AE anomaly threshold
│   │   ├── ae_metadata.pkl                # Production AE metadata (input_dim)
│   │   ├── scaler.pkl                     # StandardScaler for feature preprocessing
│   │   ├── candidate_isolation_forest.pkl # Candidate IF (post-retrain)
│   │   ├── candidate_autoencoder.pth      # Candidate AE (post-retrain)
│   │   ├── candidate_ae_threshold.pkl     # Candidate AE threshold
│   │   ├── candidate_ae_metadata.pkl      # Candidate AE metadata
│   │   ├── candidate_training_metadata.json # Candidate training details
│   │   ├── retrained_isolation_forest.pkl # Retrained IF (intermediate)
│   │   ├── retrained_autoencoder.pth      # Retrained AE (intermediate)
│   │   ├── retrained_ae_threshold.pkl     # Retrained AE threshold
│   │   ├── retrained_ae_metadata.pkl      # Retrained AE metadata
│   │   ├── model_registry.json            # Model version registry
│   │   ├── model_version.json             # Current production version number
│   │   ├── production/                    # Promoted production model snapshots
│   │   ├── candidates/                    # Candidate model staging area
│   │   └── archive/                       # Archived previous production versions
│   ├── evaluation/                        # Evaluation artifacts
│   └── training/                          # Training artifacts
│
├── notebooks/                             # Jupyter notebooks for exploration
│   ├── data_understanding.ipynb           # Data exploration & visualization
│   ├── model_data_prep.ipynb              # Feature engineering & dataset prep
│   ├── model_train_data_split.ipynb       # Train/test split logic
│   ├── ae.ipynb                           # Autoencoder training
│   ├── ae_retrain.ipynb                   # Autoencoder retraining experiments
│   ├── autoencoder_evaluation.ipynb       # AE performance evaluation
│   └── streamdata.ipynb                   # Streaming data experiments
│
├── logs/                                  # Runtime log storage
├── retraining/                            # Retraining artifacts
├── storage/                               # Additional storage
├── tests/                                 # Test suite
└── .venv/                                 # Python virtual environment
```

---

## Module Reference

### 1. Kafka Layer

#### `kafka/producers/producer.py`
Reads `data/processed/model_dataset.csv` row-by-row and publishes each record as JSON to the `raw_traffic` Kafka topic. Includes a configurable 10ms sleep for demo pacing and flushes every 10,000 records.

#### `kafka/consumers/consumer_test.py`
A minimal test consumer that subscribes to `raw_traffic` from the earliest offset and prints every message.

---

### 2. Feature Processing (Flink Layer)

#### `flink/processor.py`
The core **feature processor** that acts as the Flink-equivalent processing stage:
1. Consumes from `raw_traffic` topic
2. Removes the `Label` column (ground truth) from each record
3. Applies the pre-fitted `StandardScaler` (loaded from `models/saved_models/scaler.pkl`)
4. Publishes the scaled feature vector to `processed_features` topic

#### `flink/config.py`
Central configuration for Kafka broker address, input/output topics, and scaler path.

#### `flink/feature_processor.py`
Standalone preprocessing function that loads the scaler and provides a `preprocess(record)` function.

---

### 3. Inference Pipeline

#### `inference/inference_worker.py`
The **dual-model scoring engine**:
- Loads the production **Isolation Forest** (`isolation_forest.pkl`)
- Loads the production **Autoencoder** (`autoencoder.pth`) — a 59→64→32→16→32→64→59 architecture
- For each `processed_features` message:
  - Computes the **IF anomaly score** (normalized to 0–1)
  - Computes the **AE reconstruction error** (MSE, normalized against threshold)
  - Publishes both scores to `anomaly_scores` topic

**Autoencoder Architecture:**
```
Encoder: 59 → 64 → 32 → 16  (ReLU activations)
Decoder: 16 → 32 → 64 → 59  (ReLU activations)
```

#### `inference/fusion_worker.py`
Consumes from `anomaly_scores` and computes a **weighted fusion score**:
```
final_score = 0.2 × IF_score + 0.8 × AE_score
```
Publishes results (including original features, individual scores, and final score) to `fusion_scores`.

#### `inference/alert_worker.py`
Consumes from `fusion_scores` and applies a **classification threshold** of **0.85**:
- `final_score ≥ 0.85` → `ANOMALY`
- `final_score < 0.85` → `NORMAL`

Publishes enriched alert records (features, all scores, threshold, status, timestamp) to `alerts`.

#### `inference/alert_consumer.py`, `fusion_consumer.py`, `score_consumer.py`
Real-time consumers for monitoring the respective topics. `alert_consumer.py` provides a formatted display of status, scores, and thresholds.

---

### 4. Parquet Storage

#### `parquet_storage.py`
Consumes from the `alerts` Kafka topic and **batches** records (default batch size: 100) into timestamped Parquet files stored in `data/history/`. File naming convention:
```
alerts_YYYYMMDD_HHMMSS_XXXX.parquet
```
This forms the historical data lake that feeds the drift detection and retraining pipeline.

---

### 5. Drift Detection (ADWIN)

#### `adwin_drift_detection.py`
Implements **ADWIN (ADaptive WINdowing)** concept drift detection from the `river` library:

| Parameter | Value | Description |
|---|---|---|
| `DELTA` | 0.002 | ADWIN sensitivity parameter |
| `CHECKPOINT_SIZE` | 10,000 | Records between ADWIN evaluations |
| `DRIFT_TRIGGER_COUNT` | 50 | Drift events required before triggering retraining |

**Process:**
1. Loads all Parquet history from `data/history/`
2. Feeds anomaly scores into the ADWIN detector at each checkpoint
3. Saves drift results to `data/processed/adwin_drift_results.csv`
4. Persists detector state to `data/processed/adwin_state.json`
5. When `DRIFT_TRIGGER_COUNT` is reached, writes a retraining trigger file

---

### 6. PySpark Retraining (Phase 11)

#### `pyspark_retraining.py`
Distributed retraining using PySpark when drift is confirmed:

1. **Data Loading** — Reads all historical Parquet files from `data/history/`
2. **Normal Traffic Selection** — Filters for `status == "NORMAL"` records only
3. **Feature Validation** — Verifies the original 59 features are present
4. **Spark DataFrame Creation** — Converts to Spark for distributed processing
5. **Isolation Forest Retraining** — Trains a new `IsolationForest` via scikit-learn
6. **Autoencoder Retraining** — Trains a new PyTorch Autoencoder (59→64→32→16→32→64→59)
7. **Threshold Calculation** — Computes new AE anomaly threshold from reconstruction errors
8. **Candidate Saving** — Saves all candidate models with `candidate_` prefix
9. **Metadata Publishing** — Writes training metadata (record count, feature count, duration, timestamps)
10. **Pipeline Status Updates** — Publishes live status for the dashboard

> ⚠️ Production models are **NEVER** modified by this module. Only candidate artifacts are written.

---

### 7. Model Evaluation (Phase 12)

#### `model_evaluation.py`
Compares **old (production)** vs. **new (candidate)** models:

| Configuration | Value |
|---|---|
| IF Weight | 0.2 |
| AE Weight | 0.8 |
| F1 Tolerance | 0.02 (candidate must not drop F1 by more than 2%) |
| FAR Tolerance | 0.02 (candidate must not increase false alarms by more than 2%) |

**Evaluation Metrics:**
- Precision, Recall, F1-Score
- False Alarm Rate (FAR)
- Reconstruction error distribution
- AUC-ROC curves

**Outputs:**
- `data/processed/phase12_model_comparison.csv`
- `data/processed/phase12_model_metadata.json`
- `data/processed/phase12_promotion_report.csv`

**Decision Logic:**
- `ACCEPTED` — Candidate meets all quality gates → eligible for promotion
- `REJECTED` — Candidate fails one or more gates → production model retained

---

### 8. Model Versioning (Phase 13)

#### `model_versioning.py` & `phase13_model_versioning.py`
Manage the **model version registry**:

- Tracks production version number in `model_version.json`
- Maintains a full registry in `model_registry.json` with:
  - Version number
  - Status (`production`, `candidate`, `archived`)
  - Timestamps (created, promoted, archived)
  - Training metadata
  - Evaluation metrics
- Archives previous production versions under `models/saved_models/archive/`
- Stages candidates under `models/saved_models/candidates/`

---

### 9. Model Lifecycle — Promotion/Rollback (Phase 14)

#### `model_lifecycle.py`
Handles the **final lifecycle decision** after evaluation:

**Pipeline Position:**
```
ADWIN → PySpark Retraining → Model Evaluation → Model Lifecycle
```

**Responsibilities:**
1. Read Phase 12 evaluation decision
2. Verify candidate model files exist and are valid
3. If `ACCEPTED`:
   - Archive current production model
   - Copy candidate → production
   - Update model registry
   - Write lifecycle status for dashboard
4. If `REJECTED`:
   - Preserve current production model
   - Log rejection reason
   - Write lifecycle status for dashboard
5. Update `automatic_pipeline_status.json` for dashboard

> 🔒 Production model is **NEVER replaced directly**. Archive-then-copy pattern ensures safe rollback.

---

### 10. Self-Correction Controller

#### `self_correction_controller.py`
The **automated control loop** that ties together all self-correction phases:

```
Parquet reaches 10,000 records
        ↓
     ADWIN drift detection
        ↓
   Drift detected?
     /          \
   NO            YES
    │              │
  Wait          PySpark retraining
                    ↓
               Model evaluation
                    ↓
               Model lifecycle (promote/reject)
                    ↓
              Continue to 20k, 30k, 40k...
```

Persists its state in `data/processed/self_correction_state.json`.

---

### 11. Automatic Retraining Orchestrator (Phase 16)

#### `automatic_retraining_orchestrator.py`
The **master orchestrator** (2,641 lines) that runs the complete autonomous pipeline:

1. Monitors Parquet history continuously
2. At each 10,000-record checkpoint, runs ADWIN
3. If drift is confirmed:
   - Spawns PySpark retraining as a subprocess
   - Runs model evaluation
   - Executes model lifecycle decision
4. Publishes pipeline status at every stage to `automatic_pipeline_status.json`
5. Logs all events to `automatic_pipeline_history.json`
6. Continues monitoring after each cycle completes

**Status Stages:**
`MONITORING` → `ADWIN_RUNNING` → `DRIFT_DETECTED` → `RETRAINING` → `EVALUATING` → `LIFECYCLE_DECISION` → `MONITORING`

---

### 12. Dashboard (Phase 15)

#### `dashboard/app.py`
Flask web application serving the monitoring dashboard at `http://127.0.0.1:5000`.

**API Endpoints:**

| Route | Handler | Description |
|---|---|---|
| `GET /` | `home()` | Renders the main dashboard HTML |
| `GET /api/system` | `system()` | System summary (total records, anomalies, normal) |
| `GET /api/anomalies` | `anomalies()` | Recent anomaly records |
| `GET /api/drift` | `drift()` | ADWIN drift detection results |
| `GET /api/retraining` | `retraining()` | Retraining pipeline status & history |
| `GET /api/models` | `models()` | Model registry & version info |
| `GET /api/evaluation` | `evaluation()` | Phase 12 comparison results |
| `GET /api/self-correction` | `self_correction()` | Self-correction controller state |
| `GET /api/dashboard` | `dashboard()` | Aggregated dashboard data (single endpoint) |

#### `dashboard/dashboard_data.py`
The **single data-source module** for all API endpoints. Key rules:
- System counters are derived from Parquet history
- Recent anomalies are newest-first from Parquet history
- ADWIN data is read from ADWIN output files
- Retraining data comes from the newest retraining artifact
- Model registry is the source of truth for versions/status
- Returns one normalized JSON object so the frontend doesn't need to guess schemas

#### `dashboard/static/dashboard.js` & `dashboard.css`
Frontend implementation (~88KB JS, ~25KB CSS) providing real-time AJAX polling and visualization of all pipeline stages.

#### `dashboard/templates/dashboard.html`
HTML template (~24KB) with sections for:
- System summary cards
- Anomaly detection table (compact + scrollable)
- ADWIN drift detection status
- Automatic self-correction pipeline status
- Top performing model display
- Model metadata & registry
- Retraining history

---

## Machine Learning Models

### Isolation Forest (IF)
| Property | Value |
|---|---|
| Algorithm | `sklearn.ensemble.IsolationForest` |
| Role | Unsupervised anomaly detection on tabular features |
| Score Weight | 20% of final fusion score |
| File | `models/saved_models/isolation_forest.pkl` |

### Autoencoder (AE)
| Property | Value |
|---|---|
| Framework | PyTorch (`torch.nn.Module`) |
| Architecture | 59 → 64 → 32 → **16** → 32 → 64 → 59 |
| Activations | ReLU (all hidden layers) |
| Anomaly Metric | Mean Squared Error (reconstruction error) |
| Score Weight | 80% of final fusion score |
| File | `models/saved_models/autoencoder.pth` |

### Feature Preprocessing
| Property | Value |
|---|---|
| Scaler | `sklearn.preprocessing.StandardScaler` |
| Input Features | 59 network traffic features |
| File | `models/saved_models/scaler.pkl` |

### Fusion Formula
```
final_score = 0.2 × IF_score + 0.8 × AE_score
```

### Alert Threshold
```
final_score ≥ 0.85  →  ANOMALY
final_score <  0.85  →  NORMAL
```

---

## Kafka Topics

| Topic | Producer | Consumer(s) | Description |
|---|---|---|---|
| `raw_traffic` | `producer.py` | `processor.py`, `consumer_test.py` | Raw CSV records as JSON |
| `processed_features` | `processor.py` | `inference_worker.py` | Scaled 59-dim feature vectors |
| `anomaly_scores` | `inference_worker.py` | `fusion_worker.py`, `score_consumer.py` | IF + AE individual scores |
| `fusion_scores` | `fusion_worker.py` | `alert_worker.py`, `fusion_consumer.py` | Weighted combined scores |
| `alerts` | `alert_worker.py` | `parquet_storage.py`, `alert_consumer.py` | Classified ANOMALY/NORMAL records |

---

## Data Directory Layout

```
data/
├── raw/                    # 8 original CICIDS2017 CSV files (~885 MB total)
├── processed/              # Cleaned datasets + pipeline state
│   ├── CICIDS2017_Cleaned.csv   (~883 MB)
│   ├── model_dataset.csv        (~770 MB)
│   ├── adwin_drift_results.csv  (~3.6 MB, drift analysis output)
│   └── *.json                   (pipeline state files)
├── features/               # Extracted feature sets (empty — features in processed/)
└── history/                # ~709 Parquet files (~30KB each, alert history)
```

**Total Raw Data Size:** ~1.65 GB (CSV) + ~22 MB (Parquet history)

---

## Model Artifacts

```
models/saved_models/
├── isolation_forest.pkl              # Production IF model
├── autoencoder.pth                   # Production AE model (~57KB)
├── ae_threshold.pkl                  # Production AE threshold
├── ae_metadata.pkl                   # Production AE metadata (input_dim=59)
├── scaler.pkl                        # StandardScaler (~3.3KB)
├── candidate_isolation_forest.pkl    # Latest candidate IF (~1.4MB)
├── candidate_autoencoder.pth         # Latest candidate AE (~57KB)
├── candidate_ae_threshold.pkl        # Latest candidate threshold
├── candidate_ae_metadata.pkl         # Latest candidate metadata
├── candidate_training_metadata.json  # Training details
├── retrained_*.pkl / .pth            # Intermediate retrained models
├── model_registry.json               # Version registry
├── model_version.json                # Current version number
├── production/                       # Promoted production snapshots
├── candidates/                       # Candidate staging area
└── archive/                          # Archived previous versions
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `data_understanding.ipynb` | Exploratory data analysis on CICIDS2017 — distributions, correlations, class balance |
| `model_data_prep.ipynb` | Feature engineering, cleaning, and dataset preparation |
| `model_train_data_split.ipynb` | Train/test/validation split strategy |
| `ae.ipynb` | Initial Autoencoder training and hyperparameter tuning |
| `ae_retrain.ipynb` | Autoencoder retraining experiments |
| `autoencoder_evaluation.ipynb` | Detailed AE performance evaluation |
| `streamdata.ipynb` | Streaming data pipeline experiments |

---

## Configuration Reference

| Parameter | Value | Location | Description |
|---|---|---|---|
| Kafka Bootstrap Server | `localhost:9092` | All Kafka modules | Kafka broker address |
| ADWIN Delta | `0.002` | `adwin_drift_detection.py` | ADWIN sensitivity |
| ADWIN Checkpoint | `10,000` records | `adwin_drift_detection.py` | Records between ADWIN evaluations |
| Drift Trigger Count | `50` | `adwin_drift_detection.py` | Drift events before retraining |
| IF Weight | `0.2` | `fusion_worker.py`, `model_evaluation.py` | Isolation Forest fusion weight |
| AE Weight | `0.8` | `fusion_worker.py`, `model_evaluation.py` | Autoencoder fusion weight |
| Alert Threshold | `0.85` | `alert_worker.py` | ANOMALY/NORMAL classification |
| F1 Tolerance | `0.02` | `model_evaluation.py` | Max allowable F1 drop |
| FAR Tolerance | `0.02` | `model_evaluation.py` | Max allowable FAR increase |
| Parquet Batch Size | `100` | `parquet_storage.py` | Records per Parquet file |
| Input Features | `59` | All model modules | Network traffic feature count |
| Dashboard Port | `5000` | `dashboard/app.py` | Flask server port |

---

## Execution Order

Start each component in a **separate terminal** in the following order:

```bash
# 1. Start Kafka broker (requires Kafka installation)
cd ~/kafka_2.13-4.2.1
bin/kafka-server-start.sh config/controller.properties
bin/kafka-server-start.sh config/broker.properties

# 2. Start the Kafka producer (streams dataset)
python kafka/producers/producer.py

# 3. Start the feature processor
python flink/processor.py

# 4. Start the inference worker (load models first — takes a moment)
python inference/inference_worker.py

# 5. Start the fusion worker
python inference/fusion_worker.py

# 6. Start the alert worker
python inference/alert_worker.py

# 7. Start the alert consumer (optional — real-time monitoring)
python inference/alert_consumer.py

# 8. Start Parquet storage (persists alerts to disk)
python parquet_storage.py

# 9. Start the automatic retraining orchestrator
python automatic_retraining_orchestrator.py

# 10. Start the Flask dashboard
python -m dashboard.app
```

Then open: **http://127.0.0.1:5000**

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.12.1 | PyTorch — Autoencoder model |
| `scikit-learn` | 1.9.0 | Isolation Forest, StandardScaler |
| `pyspark` | 4.1.2 | Distributed retraining |
| `kafka-python` | 3.0.2 | Kafka producer/consumer clients |
| `pandas` | 3.0.3 | DataFrame operations |
| `numpy` | 2.4.6 | Numerical computation |
| `pyarrow` | 24.0.0 | Parquet file I/O |
| `river` | 0.25.0 | ADWIN drift detection |
| `scipy` | 1.18.0 | Scientific computing |
| `matplotlib` | 3.11.0 | Plotting & visualization |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `joblib` | 1.5.3 | Model serialization |
| `flask` | *(implied)* | Web dashboard framework |
| `watchdog` | 6.0.0 | File system monitoring |
| `tqdm` | 4.68.3 | Progress bars |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Apache Kafka 4.2.1+ (with KRaft mode)
- Java 17+ (for Kafka/PySpark)
- ~4 GB RAM minimum (for PySpark + PyTorch)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/monish18106/adaptive-network.git
cd adaptive-network

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Kafka (see kafka config.txt for detailed steps)

# 5. Create required Kafka topics
kafka-topics.sh --create --topic raw_traffic --bootstrap-server localhost:9092
kafka-topics.sh --create --topic processed_features --bootstrap-server localhost:9092
kafka-topics.sh --create --topic anomaly_scores --bootstrap-server localhost:9092
kafka-topics.sh --create --topic fusion_scores --bootstrap-server localhost:9092
kafka-topics.sh --create --topic alerts --bootstrap-server localhost:9092

# 6. Follow the Execution Order above to start all components
```

---

## Dataset

This system is built on the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity:

| Property | Detail |
|---|---|
| **Name** | CICIDS2017 (Intrusion Detection Evaluation Dataset) |
| **Source** | University of New Brunswick |
| **Format** | 8 CSV files covering a full work week (Mon–Fri) |
| **Size** | ~885 MB raw |
| **Records** | ~2.8 million network flow records |
| **Features** | 78 original features, 59 used after cleaning |
| **Attack Types** | DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet |
| **Benign Label** | `BENIGN` |

---

<p align="center">
  <em>Built with Apache Kafka • PyTorch • PySpark • scikit-learn • River • Flask</em>
</p>

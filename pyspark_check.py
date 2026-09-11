import pandas as pd
import glob
import os


HISTORY_DIR = "data/history"


print("=" * 80)
print("PHASE 11 - PYSPARK RETRAINING DATA CHECK")
print("=" * 80)


# ------------------------------------------------------------
# FIND PARQUET FILES
# ------------------------------------------------------------

files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*.parquet")))

print(f"\nParquet files found : {len(files)}")

if not files:
    raise FileNotFoundError("No Parquet files found.")


# ------------------------------------------------------------
# LOAD ALL HISTORY
# ------------------------------------------------------------

frames = []

for file in files:

    try:
        df = pd.read_parquet(file)

        if len(df) > 0:
            frames.append(df)

    except Exception as e:
        print(f"Skipping {file}: {e}")


if not frames:
    raise ValueError("No readable Parquet data found.")


history = pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

print(f"Total historical records : {len(history)}")

print("\nColumns:")
for col in history.columns:
    print(f" - {col}")


# ------------------------------------------------------------
# DATA TYPES
# ------------------------------------------------------------

print("\nData types:")
print(history.dtypes)


# ------------------------------------------------------------
# STATUS DISTRIBUTION
# ------------------------------------------------------------

if "status" in history.columns:

    print("\nStatus distribution:")
    print(history["status"].value_counts())


# ------------------------------------------------------------
# SCORE STATISTICS
# ------------------------------------------------------------

score_columns = [
    "if_score",
    "ae_score",
    "final_score"
]

available_scores = [
    col for col in score_columns
    if col in history.columns
]

if available_scores:

    print("\nScore statistics:")
    print(
        history[available_scores].describe()
    )


# ------------------------------------------------------------
# CHECK FOR ORIGINAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RETRAINING FEATURE CHECK")
print("=" * 80)

original_feature_count = 59

if len(history.columns) >= original_feature_count:

    print(
        f"Historical data has at least "
        f"{original_feature_count} columns."
    )

else:

    print(
        f"Historical data has only "
        f"{len(history.columns)} columns."
    )

    print(
        "\nWARNING:"
    )

    print(
        "The current Parquet history does NOT contain "
        "the original 59 network features."
    )

    print(
        "Therefore we should NOT retrain the Autoencoder "
        "or Isolation Forest directly from these Parquet files."
    )


# ------------------------------------------------------------
# FINAL DECISION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PHASE 11 CHECK COMPLETE")
print("=" * 80)

print(
    "\nNext step:"
)

print(
    "Identify the correct historical source containing "
    "the original 59 features."
)

print(
    "\nDo NOT modify the trained models yet."
)
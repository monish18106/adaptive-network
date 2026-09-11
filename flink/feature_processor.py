import joblib
import pandas as pd

scaler = joblib.load("../saved_models/scaler.pkl")

def preprocess(record: dict):

    # Remove label
    record.pop("Label", None)

    df = pd.DataFrame([record])

    scaled = scaler.transform(df)

    return scaled[0].tolist()
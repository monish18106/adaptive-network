import joblib

model = joblib.load("../models/saved_models/isolation_forest.pkl")

print(type(model))
print(model.n_features_in_)
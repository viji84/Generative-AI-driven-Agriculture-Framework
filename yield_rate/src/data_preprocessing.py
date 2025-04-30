import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load dataset
df = pd.read_csv("datasets\Yield Rate.csv", dtype={"State Name": str}, low_memory=False)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select features and target variables
common_columns = ["year", "state_name"]
target_columns = [col for col in df.columns if col.endswith("_yield")]
features = df.drop(columns=common_columns + target_columns)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler
os.makedirs("models", exist_ok=True)
pickle.dump(scaler, open("models\scaler.pkl", "wb"))

print("Data preprocessing complete and scaler saved!")

import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("datasets/merged_Data.csv", dtype={"State Name": str}, low_memory=False)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select features and target variables
common_columns = ["year", "state_name"]
target_columns = [col for col in df.columns if col.endswith("_yield")]
features = df.drop(columns=common_columns + target_columns)

# Load scaler and normalize features
scaler = pickle.load(open("models/scaler.pkl", "rb"))
features_scaled = scaler.transform(features)

# Define crops
crops = [col.replace("_yield", "") for col in target_columns]

# Store evaluation results
evaluation_results = {}

for crop, crop_yield in zip(crops, target_columns):
    y_train = df[crop_yield].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, y_train, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, objective="reg:squarederror")
    model.fit(X_train, y_train)
    
    # Save each model separately
    model.save_model(f"models/{crop}_model.bin")
    
    # Model evaluation
    y_pred = model.predict(X_test)
    evaluation_results[crop] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }

# Save evaluation results
evaluation_df = pd.DataFrame(evaluation_results).T
evaluation_df.to_csv("models/evaluation_summary.csv", index=True)

print("✅ Model training complete! Individual models saved as .bin files.")

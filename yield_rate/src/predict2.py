import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer

# Load dataset
df = pd.read_csv("datasets\merged_data.csv", dtype={"State Name": str}, low_memory=False)

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
overall_metrics = {"Precision": [], "Recall": [], "F1-Score": [], "Accuracy": []}

# Bin yield values into categories
n_bins = 3  # Define number of categories (Low, Medium, High)
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

def bin_values(y):
    return discretizer.fit_transform(y.reshape(-1, 1)).astype(int).flatten()

for crop, crop_yield in zip(crops, target_columns):
    y_train = df[crop_yield].fillna(0).values
    y_train_binned = bin_values(y_train)
    
    _, X_test, _, y_test = train_test_split(features_scaled, y_train, test_size=0.2, random_state=42)
    
    # Load the saved model
    model = xgb.XGBRegressor()
    model.load_model(f"models/{crop}_model.bin")
    
    # Model evaluation
    y_pred = model.predict(X_test)
    
    # Compute accuracy (approximation)
    mean_actual = np.mean(y_test)
    accuracy = 1 - (mean_absolute_error(y_test, y_pred) / mean_actual) if mean_actual != 0 else 0
    
    precision = precision_score(bin_values(y_test), bin_values(y_pred), average='weighted', zero_division=0)
    recall = recall_score(bin_values(y_test), bin_values(y_pred), average='weighted', zero_division=0)
    f1 = f1_score(bin_values(y_test), bin_values(y_pred), average='weighted', zero_division=0)
    
    overall_metrics["Precision"].append(precision)
    overall_metrics["Recall"].append(recall)
    overall_metrics["F1-Score"].append(f1)
    overall_metrics["Accuracy"].append(accuracy * 100)  # Convert to percentage
    
    evaluation_results[crop] = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy * 100  # Convert to percentage
    }

# Compute overall classification metrics
overall_precision = np.mean(overall_metrics["Precision"])
overall_recall = np.mean(overall_metrics["Recall"])
overall_f1 = np.mean(overall_metrics["F1-Score"])
overall_accuracy = np.mean(overall_metrics["Accuracy"])

# Convert to DataFrame for tabular display
evaluation_df = pd.DataFrame(evaluation_results).T

# Print overall classification metrics
print("\n🎯 Overall Classification Metrics:\n")
print("Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Accuracy: {:.2f}%\n".format(
    overall_precision, overall_recall, overall_f1, overall_accuracy))

# Print classification report format
print("Classification Report:\n")
print(evaluation_df.to_string(float_format="%.4f"))
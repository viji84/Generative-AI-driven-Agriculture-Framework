import pandas as pd
import numpy as np
import joblib
import h5py
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# Load the dataset with explicit dtype handling
file_path = "datasets\Yield Rate.csv"
df = pd.read_csv(file_path, low_memory=False, dtype=str)

# Convert all numeric columns to proper data types
df = df.apply(pd.to_numeric, errors='coerce')

# Trim column names to remove extra spaces
df.columns = df.columns.str.strip()

# Define target variable
target_column = "RICE YIELD"

# Drop rows with missing target values
df = df.dropna(subset=[target_column])

# Ensure target variable is non-negative
df[target_column] = df[target_column].clip(lower=0)

# Convert target variable into categorical bins
num_classes = 10  # Define number of categories
df[target_column], bins = pd.qcut(df[target_column], q=num_classes, labels=False, retbins=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Splitting the dataset into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the XGBoost classifier
model = XGBClassifier(objective="multi:softmax", eval_metric="mlogloss", num_class=num_classes)
model.fit(X_train, y_train)

# Save the trained model in .h5 format
joblib.dump(model, "xgboost_model.pkl")

# Save the model to a binary buffer and store it in .h5 format
buffer = io.BytesIO()
joblib.dump(model, buffer)
with h5py.File("xgboost_model.h5", "w") as h5f:
    h5f.create_dataset("model", data=np.void(buffer.getvalue()))

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"R² Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print("Classification Report:\n", classification_rep)

# Generate and display classification report as a heatmap
report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()



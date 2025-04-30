Project Title
Generative AI-Driven Framework for Agriculture: Predicting Crop Yields and Adaptive Farming Strategies

📌 Overview
This project aims to build a robust crop yield prediction system using a hybrid dataset comprising real and synthetic data. Synthetic data is generated using Generative Adversarial Networks (GAN) and Variational Autoencoders (VAE). The predictive model is built using XGBoost, and it estimates the yield rate of various crops based on agro-climatic and soil-related parameters.

The project is part of a larger goal to assist farmers and policymakers in making data-driven decisions for climate-resilient agriculture.

🔍 Features
✅ Predicts crop yield using real and synthetic data.

✅ Uses XGBoost Regression for high-accuracy predictions.

✅ Employs GANs and VAEs to augment sparse datasets.

✅ Supports analysis of climate impact on crop productivity.

✅ Interactive interface for farmers and researchers (Web UI or Dashboard).

✅ Crop selection and automatic feature filling for missing inputs.

🧪 Technologies Used
Python (NumPy, Pandas, Scikit-learn)

XGBoost

TensorFlow/Keras (for GAN & VAE)

Flask/Streamlit (for deployment/UI)

VS Code (development environment)

📂 Dataset Details
Real Data Sources:

Crop yield statistics (area, production, irrigated area)

Rainfall (monthly & annual)

Temperature (monthly max/min)

Soil nutrients: Nitrogen (N), Phosphorus (P), Potassium (K)

Fertilizer consumption and irrigation data

Synthetic Data:

Generated using GAN and VAE models to improve data volume and balance

📈 Model Architecture
Data Preprocessing:

Missing value imputation

Normalization & encoding

Feature engineering (e.g., annual rainfall, temperature range)

Data Augmentation:

Train GAN & VAE models on real data

Merge generated synthetic data with real data

Model Training:

XGBoost Regressor trained on combined dataset

Evaluation using RMSE, MAE, R² Score

Deployment (Optional):

Flask or Streamlit app

Optional Power BI dashboard

🚀 Getting Started

Prerequisites
pip install -r requirements.txt

Clone and Run
git clone https://github.com/yourusername/yield_rate_prediction.git
cd yield_rate_prediction
python train_xgboost.py  # for training the model
python app.py            # to run the web app

📊 Evaluation Metrics
Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R² Score


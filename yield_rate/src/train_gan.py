import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

# Load preprocessed data and scaler
df = pd.read_csv("datasets\Yield Rate.csv", dtype={"State Name": str}, low_memory=False)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select features and target variables
common_columns = ["year", "state_name"]
target_columns = [col for col in df.columns if col.endswith("_yield")]
features = df.drop(columns=common_columns + target_columns)

# Load the scaler used for data preprocessing
scaler = pickle.load(open("models\scaler.pkl", "rb"))
features_scaled = scaler.transform(features)  # Scale features using the same scaler as before

# Set parameters
latent_dim = 10  # Latent space dimension
input_dim = features_scaled.shape[1]  # Number of input features

# Generator Model
def build_generator():
    model = tf.keras.Sequential([
        Dense(32, activation="relu", input_dim=latent_dim),
        Dense(64, activation="relu"),
        Dense(input_dim, activation="tanh")  # 'tanh' to match the range of the scaled features
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential([
        Dense(64, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(32),
        LeakyReLU(alpha=0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

# Create GAN
generator = build_generator()
discriminator = build_discriminator()

gan_input = Input(shape=(latent_dim,))
generated_data = generator(gan_input)
discriminator.trainable = False
validity = discriminator(generated_data)

gan = Model(gan_input, validity)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# Train GAN for 20 epochs
epochs = 20  
batch_size = 32

for epoch in range(epochs):
    idx = np.random.randint(0, features_scaled.shape[0], batch_size)
    real_samples = features_scaled[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_samples = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_samples, np.zeros((batch_size, 1)))

    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 5 == 0:  # Reduce output frequency to every 5 epochs
        print(f"Epoch {epoch}: D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# Generate synthetic data using the trained generator
synthetic_data_gan = generator.predict(np.random.normal(0, 1, (len(features_scaled), latent_dim)))

# Inverse scale the synthetic data
synthetic_data_gan_original = scaler.inverse_transform(synthetic_data_gan)

# Save synthetic data
os.makedirs("datasets", exist_ok=True)
np.save("datasets\synthetic_gan.npy", synthetic_data_gan_original)

print("GAN Training Complete and Synthetic Data Saved!")

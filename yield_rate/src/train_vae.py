import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import MeanSquaredError

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

# Define the VAE loss function
def vae_loss(inputs, vae_outputs, z_mean, z_log_var):
    # Reconstruction loss (Mean Squared Error)
    recon_loss = MeanSquaredError()(inputs, vae_outputs)
    
    # KL divergence loss
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    
    # Total VAE loss
    return K.mean(recon_loss + kl_loss)

input_dim = features_scaled.shape[1]  # Number of input features
latent_dim = 10  # Latent space dimension

# Encoder architecture
inputs = Input(shape=(input_dim,))
h = Dense(64, activation="relu")(inputs)
h = Dense(32, activation="relu")(h)
z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_var = Dense(latent_dim, name="z_log_var")(h)

# Sampling function for the latent space
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))  # Standard Normal Distribution
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Latent space
z = Lambda(sampling, output_shape=(latent_dim,), name="z_sampling")([z_mean, z_log_var])

# Decoder architecture
decoder_input = Input(shape=(latent_dim,))
d = Dense(32, activation="relu")(decoder_input)
d = Dense(64, activation="relu")(d)
outputs = Dense(input_dim, activation="sigmoid")(d)  # 'sigmoid' or 'tanh' depending on your data range

# Encoder and Decoder models
encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
decoder = Model(decoder_input, outputs, name="decoder")

# VAE Model
vae_outputs = decoder(z)
vae = Model(inputs, vae_outputs, name="vae_mlp")

# Compute the loss manually
def compute_loss(inputs, vae_outputs, z_mean, z_log_var):
    return vae_loss(inputs, vae_outputs, z_mean, z_log_var)

# Compile the model
vae.compile(optimizer="adam")

# Training the VAE
epochs = 10  # Number of epochs for training
batch_size = 32  # Batch size for training

# Custom training loop
@tf.function  # Optional for faster training with TensorFlow 2.x
def train_step(inputs):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = encoder(inputs)
        vae_outputs = decoder(z)
        loss = compute_loss(inputs, vae_outputs, z_mean, z_log_var)
    grads = tape.gradient(loss, vae.trainable_variables)
    vae.optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for batch in range(0, len(features_scaled), batch_size):
        batch_inputs = features_scaled[batch:batch+batch_size]
        loss = train_step(batch_inputs)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Generate synthetic data
# Sampling random latent vectors
random_latent_vectors = np.random.normal(size=(len(features_scaled), latent_dim))

# Generate synthetic data from the random latent vectors
synthetic_data_vae = decoder.predict(random_latent_vectors)

# Save synthetic data
os.makedirs("datasets", exist_ok=True)
np.save("datasets\synthetic_vae.npy", synthetic_data_vae)

print("VAE Training Complete and Synthetic Data Saved!")

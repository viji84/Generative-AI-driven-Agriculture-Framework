from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import torch
import os
from datasets import Dataset

# Ensure the correct device is selected
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
file_path = r"C:\Users\Beve2\Desktop\yield_rate\datasets\Updated_Farming_Strategies.csv"  # Direct path

df = pd.read_csv(file_path)

# Select input columns
input_columns = ['Nitrogen Level', 'Phosphorous Level', 'Potassium Level',
                 'Temperature (°C)', 'pH Level', 'Crop Selected', 'Yield Rate']

# Combine selected columns into a single input text
df["input_text"] = df[input_columns].astype(str).agg(" | ".join, axis=1)

# Combine "Irrigation Plan" and "Crop Rotation Plan" as target text
df["target_text"] = df["Irrigation Plan"].astype(str) + " [SEP] " + df["Crop Rotation Plan"].astype(str)

# Reduce dataset size (use a smaller fraction for quick training)
df_sampled = df.sample(frac=0.2, random_state=42)  # Use only 20% of data

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_sampled[["input_text", "target_text"]])

# Split dataset into train and validation
split_dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% eval
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Tokenize dataset
def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")  # Reduced max_length
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")  # Increased for two plans

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Training Arguments (Optimized for Speed)
training_args = TrainingArguments(
    output_dir="./t5_farming_strategy_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=0.5,  # Reduced to speed up
    logging_steps=500,  # Reduce logging frequency
    save_strategy="no",  # Avoid slow save checkpoints
    learning_rate=5e-4,  # Increased LR for faster convergence
    weight_decay=0.01,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=4  # Efficient batch simulation
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("models/t5_farming_strategy_model")
tokenizer.save_pretrained("models/t5_farming_strategy_model")

print("✅ Training complete! Model saved.")
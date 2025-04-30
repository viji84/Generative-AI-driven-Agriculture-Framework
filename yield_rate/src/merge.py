import pandas as pd

# Load datasets
df_gan = pd.read_csv("datasets2/synthetic_gan_50yrs.csv")
df_vae = pd.read_csv("datasets2/synthetic_vae_50yrs.csv")

# Ensure both datasets have the same number of rows
min_rows = min(len(df_gan), len(df_vae))

# Trim both datasets to the minimum number of rows
df_gan_trimmed = df_gan.iloc[:min_rows]
df_vae_trimmed = df_vae.iloc[:min_rows]

# Merge the datasets column-wise
df_merged = pd.concat([df_gan_trimmed, df_vae_trimmed], axis=1)

# Save the merged dataset
df_merged.to_csv("datasets2/merged_dataset.csv", index=False)

print(f"Merged dataset saved with {len(df_merged)} rows and {df_merged.shape[1]} columns.")

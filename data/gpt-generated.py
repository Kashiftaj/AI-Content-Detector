from datasets import load_dataset
import pandas as pd
import os

# Make sure raw folder exists
os.makedirs("raw", exist_ok=True)

print("Downloading AI-Text-Detection-Pile (GPT-generated + human)...")

# Load full train split (all parquet parts automatically)
ds = load_dataset("artem9k/ai-text-detection-pile", split="train")

print("Converting to DataFrame...")
df = ds.to_pandas()

# Save it
output_path = "raw/gpt_generated.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"Dataset saved → {output_path}")
print("Rows:", len(df))

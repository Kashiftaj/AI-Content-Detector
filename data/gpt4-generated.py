from datasets import load_dataset
import pandas as pd
import os

os.makedirs("raw", exist_ok=True)

print("Downloading routellm/gpt4_dataset ...")
ds = load_dataset("routellm/gpt4_dataset", split="train")

print("Converting to pandas DataFrame ...")
df = ds.to_pandas()

output_path = "raw/routellm_gpt4_dataset.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print("Saved dataset →", output_path)
print("Total rows:", len(df))

"""
Preprocess APTOS Diabetic Retinopathy Dataset
Convert 5-class (0-4) → Binary (0: No DR, 1: DR Present)
Resize to 224x224, split into train/val/test
"""

import os
import pandas as pd
import random
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
CSV_PATH    = os.path.join(os.path.dirname(SCRIPT_DIR), "DR1", "raw_data", "train.csv")  # Update path to raw data
RAW_IMG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "DR1", "raw_data", "train_images")  # Update path to raw data
OUT_DIR     = os.path.join(SCRIPT_DIR, "processed_data_dr_binary"

IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)

# ================= LOAD CSV =================
df = pd.read_csv(CSV_PATH)
print("Total rows in CSV:", len(df))
print("\nOriginal class distribution (5-class):")
print(df["diagnosis"].value_counts().sort_index())

# ================= CONVERT TO BINARY =================
# 0 → 0 (No DR)
# 1,2,3,4 → 1 (DR Present)
df["binary_label"] = (df["diagnosis"] >= 1).astype(int)

print("\nBinary class distribution:")
print(df["binary_label"].value_counts().sort_index())
print("  0 (No DR):", (df["binary_label"] == 0).sum())
print("  1 (DR)   :", (df["binary_label"] == 1).sum())

# ================= FILTER VALID IMAGES =================
valid_rows = []
for i in range(len(df)):
    img_name = df.iloc[i]["id_code"] + ".png"
    img_path = os.path.join(RAW_IMG_DIR, img_name)
    if os.path.exists(img_path):
        valid_rows.append(i)

df = df.iloc[valid_rows].reset_index(drop=True)
print("\nValid images found:", len(df))

# ================= SHUFFLE & SPLIT =================
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

total = len(df)
train_end = int(0.70 * total)
val_end   = train_end + int(0.15 * total)

split_df = {
    "train": df[:train_end],
    "val":   df[train_end:val_end],
    "test":  df[val_end:]
}

print("\nSplit sizes:")
for split, data in split_df.items():
    print(f"  {split}: {len(data)}")

# ================= PROCESS & SAVE =================
for split, data in split_df.items():
    for _, row in data.iterrows():
        label = str(row["binary_label"])
        img_name = row["id_code"] + ".png"

        src = os.path.join(RAW_IMG_DIR, img_name)
        dst_dir = os.path.join(OUT_DIR, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(dst_dir, img_name))
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# ================= VERIFY =================
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

for split in ["train", "val", "test"]:
    for cls in ["0", "1"]:
        cls_dir = os.path.join(OUT_DIR, split, cls)
        if os.path.exists(cls_dir):
            count = len(os.listdir(cls_dir))
            print(f"  {split}/{cls}: {count} images")

print("\nOutput saved at:", OUT_DIR)

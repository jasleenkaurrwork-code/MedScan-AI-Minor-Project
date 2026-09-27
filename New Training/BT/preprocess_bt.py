"""
Preprocess Brain Tumor MRI Dataset - 4 Classes
- Uses RAW data from Brain Tumor folder (Training + Testing)
- Splits Training into train/val (85/15), Testing stays as test
- Resizes to 224x224
- 4 classes: glioma, meningioma, notumor, pituitary
"""

import os
import random
import shutil
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
RAW_TRAIN = os.path.join(os.path.dirname(SCRIPT_DIR), "Brain Tumor", "Training")  # Update path to raw data
RAW_TEST  = os.path.join(os.path.dirname(SCRIPT_DIR), "Brain Tumor", "Testing")  # Update path to raw data
OUT_DIR   = os.path.join(SCRIPT_DIR, "processed_data_bt"

IMG_SIZE = (224, 224)
SEED = 42
VAL_RATIO = 0.15

random.seed(SEED)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

print("=" * 50)
print("BT - PREPROCESSING (4-Class, from Raw Data)")
print("=" * 50)

# ================= TRAIN + VAL (from Training folder) =================
print("\nSplitting Training data into train/val...")

for cls in CLASSES:
    src_dir = os.path.join(RAW_TRAIN, cls)
    all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    val_count = int(len(all_images) * VAL_RATIO)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    for split, images in [("train", train_images), ("val", val_images)]:
        dst_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for img_name in images:
            src_path = os.path.join(src_dir, img_name)
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(os.path.join(dst_dir, img_name))
            except Exception as e:
                print("  Error: {}: {}".format(img_name, e))

    print("  {}: {} train, {} val".format(cls, len(train_images), len(val_images)))

# ================= TEST (from Testing folder) =================
print("\nProcessing Testing data...")

for cls in CLASSES:
    src_dir = os.path.join(RAW_TEST, cls)
    dst_dir = os.path.join(OUT_DIR, "test", cls)
    os.makedirs(dst_dir, exist_ok=True)

    all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in all_images:
        src_path = os.path.join(src_dir, img_name)
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(dst_dir, img_name))
        except Exception as e:
            print("  Error: {}: {}".format(img_name, e))

    print("  {}: {} test".format(cls, len(all_images)))

# ================= VERIFY =================
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

total = 0
for split in ["train", "val", "test"]:
    print("\n{}:".format(split.upper()))
    for cls in CLASSES:
        cls_dir = os.path.join(OUT_DIR, split, cls)
        count = len(os.listdir(cls_dir))
        print("  {}: {}".format(cls, count))
        total += count

print("\nTotal images:", total)
print("Output:", OUT_DIR)

"""
Preprocess CheXpert Chest X-Ray Dataset
- Uses existing processed_data from CX1 (already split into train/val/test)
- Balances classes by undersampling majority class (Pleural_Effusion)
- Resizes to 224x224
- Saves balanced dataset for efficient training
"""

import os
import random
import shutil
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "CX1", "processed_data")  # Update path to raw data
OUT_DIR = os.path.join(SCRIPT_DIR, "processed_data_cx"

IMG_SIZE = (224, 224)
SEED = 42
MAX_PER_CLASS_TRAIN = 10000  # Cap each class to balance the dataset
MAX_PER_CLASS_VAL = 1400
MAX_PER_CLASS_TEST = 1200

random.seed(SEED)

# ================= PROCESS =================
splits = {
    "train": MAX_PER_CLASS_TRAIN,
    "val": MAX_PER_CLASS_VAL,
    "test": MAX_PER_CLASS_TEST
}

print("=" * 50)
print("CX - PREPROCESSING (Balanced)")
print("=" * 50)

for split, max_per_class in splits.items():
    src_split = os.path.join(SRC_DIR, split)
    classes = sorted(os.listdir(src_split))

    print(f"\n--- {split.upper()} ---")

    for cls in classes:
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(OUT_DIR, split, cls)
        os.makedirs(dst_cls, exist_ok=True)

        all_images = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        original_count = len(all_images)

        # Undersample if needed
        if len(all_images) > max_per_class:
            random.shuffle(all_images)
            all_images = all_images[:max_per_class]

        count = 0
        for img_name in all_images:
            src_path = os.path.join(src_cls, img_name)
            dst_path = os.path.join(dst_cls, img_name)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(dst_path)
                count += 1
            except Exception as e:
                print(f"  Error: {img_name}: {e}")

        print(f"  {cls}: {original_count} -> {count} images")

# ================= VERIFY =================
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

total = 0
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    split_dir = os.path.join(OUT_DIR, split)
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        count = len(os.listdir(cls_dir))
        print(f"  {cls}: {count}")
        total += count

print(f"\nTotal images: {total}")
print("Output:", OUT_DIR)

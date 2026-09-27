"""
Preprocess CheXpert Chest X-Ray Dataset - BINARY
- Normal (No_Finding) vs Abnormal (Pleural_Effusion, Pneumonia, Pneumothorax)
- Balanced classes, resized to 224x224
"""

import os
import random
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "CX1", "processed_data")  # Update path to raw data
OUT_DIR = os.path.join(SCRIPT_DIR, "processed_data_cx_binary"

IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)

NORMAL_CLASSES = ["No_Finding"]
ABNORMAL_CLASSES = ["Pleural_Effusion", "Pneumonia", "Pneumothorax"]

splits_config = {
    "train": 5000,  # per class
    "val": 1400,
    "test": 1200
}

print("=" * 50)
print("CX - BINARY PREPROCESSING")
print("Normal: No_Finding")
print("Abnormal: Pleural_Effusion + Pneumonia + Pneumothorax")
print("=" * 50)

for split, max_per_class in splits_config.items():
    src_split = os.path.join(SRC_DIR, split)

    # Collect Normal images
    normal_images = []
    for cls in NORMAL_CLASSES:
        cls_dir = os.path.join(src_split, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                normal_images.append(os.path.join(cls_dir, f))

    # Collect Abnormal images
    abnormal_images = []
    for cls in ABNORMAL_CLASSES:
        cls_dir = os.path.join(src_split, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                abnormal_images.append(os.path.join(cls_dir, f))

    random.shuffle(normal_images)
    random.shuffle(abnormal_images)

    normal_images = normal_images[:max_per_class]
    abnormal_images = abnormal_images[:max_per_class]

    print("\n--- {} ---".format(split.upper()))
    print("  Normal (0):   {} images".format(len(normal_images)))
    print("  Abnormal (1): {} images".format(len(abnormal_images)))

    for label, images in [("0", normal_images), ("1", abnormal_images)]:
        dst_dir = os.path.join(OUT_DIR, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        for src_path in images:
            fname = os.path.basename(src_path)
            # Add class prefix to avoid name collisions
            dst_path = os.path.join(dst_dir, fname)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(dst_path)
            except Exception as e:
                print("  Error: {}: {}".format(fname, e))

# ================= VERIFY =================
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

total = 0
for split in ["train", "val", "test"]:
    print("\n{}:".format(split.upper()))
    for cls in ["0", "1"]:
        cls_dir = os.path.join(OUT_DIR, split, cls)
        count = len(os.listdir(cls_dir))
        label = "Normal" if cls == "0" else "Abnormal"
        print("  {} ({}): {} images".format(cls, label, count))
        total += count

print("\nTotal images:", total)
print("Output:", OUT_DIR)

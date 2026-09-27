"""
Preprocess Breast Cancer (IDC) Dataset - Binary
- Already binary: 0 (non-IDC) vs 1 (IDC)
- Balances by undersampling majority class
- Resizes to 224x224
"""

import os
import random
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "BC1", "processed_data")  # Update path to raw data
OUT_DIR = os.path.join(SCRIPT_DIR, "processed_data_bc"

IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)

splits_config = {
    "train": 5000,
    "val": 1500,
    "test": 1500
}

print("=" * 50)
print("BC - PREPROCESSING (Binary, Balanced)")
print("0: Non-IDC (Normal)")
print("1: IDC (Cancer)")
print("=" * 50)

for split, max_per_class in splits_config.items():
    src_split = os.path.join(SRC_DIR, split)

    for cls in ["0", "1"]:
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(OUT_DIR, split, cls)
        os.makedirs(dst_cls, exist_ok=True)

        all_images = [f for f in os.listdir(src_cls) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        original_count = len(all_images)

        random.shuffle(all_images)
        all_images = all_images[:max_per_class]

        count = 0
        for img_name in all_images:
            src_path = os.path.join(src_cls, img_name)
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(os.path.join(dst_cls, img_name))
                count += 1
            except Exception as e:
                print("  Error: {}: {}".format(img_name, e))

        label = "Non-IDC" if cls == "0" else "IDC"
        print("  {}/{} ({}): {} -> {}".format(split, cls, label, original_count, count))

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
        label = "Non-IDC" if cls == "0" else "IDC"
        print("  {} ({}): {}".format(cls, label, count))
        total += count

print("\nTotal images:", total)
print("Output:", OUT_DIR)

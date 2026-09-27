"""
Train DenseNet121 for Chest X-Ray (CX) Classification - 4 Classes
- Full fine-tuning (all layers unfrozen)
- Data augmentation + ImageNet normalization
- Weighted CrossEntropyLoss (handles class imbalance)
- ReduceLROnPlateau scheduler
- GPU (CUDA) only
- Saves best model based on validation accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= PATHS =================
DATASET_DIR = os.path.join(SCRIPT_DIR, "processed_data_cx"
SAVE_PATH   = os.path.join(SCRIPT_DIR, "DenseNet121_CX.pth"

train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")
test_dir  = os.path.join(DATASET_DIR, "test")

# ================= CONFIG =================
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
IMG_SIZE = 224

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= DATASETS =================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=val_transform)

NUM_CLASSES = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Num Classes:", NUM_CLASSES)
print("Train:", len(train_dataset))
print("Val:  ", len(val_dataset))
print("Test: ", len(test_dataset))

# ================= CLASS WEIGHTS =================
class_counts = np.zeros(NUM_CLASSES)
for _, label in train_dataset:
    class_counts[label] += 1

print("\nClass distribution in training:")
for i, count in enumerate(class_counts):
    print("  Class {} ({}): {} samples".format(i, train_dataset.classes[i], int(count)))

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class weights:", class_weights)

# ================= DATALOADERS =================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ================= MODEL =================
model = models.densenet121(weights="IMAGENET1K_V1")

# Unfreeze ALL layers for full fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Replace classifier
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

print("\nModel: DenseNet121 (full fine-tuning)")
print("Classifier: Linear({} -> {})".format(in_features, NUM_CLASSES))

# ================= LOSS, OPTIMIZER, SCHEDULER =================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# ================= TRAINING LOOP =================
best_val_acc = 0.0

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    # --- TRAIN ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print("  Batch {}/{} | Loss: {:.4f}".format(
                batch_idx + 1, len(train_loader), loss.item()
            ))

    train_acc = 100.0 * train_correct / train_total
    train_loss = train_loss / train_total

    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total

    # --- LR SCHEDULER ---
    scheduler.step(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]
    print("Epoch {:02d}/{} | Train Loss: {:.4f} | Train Acc: {:.2f}% | Val Acc: {:.2f}% | LR: {:.6f}".format(
        epoch + 1, EPOCHS, train_loss, train_acc, val_acc, current_lr
    ))

    # --- SAVE BEST ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("  >> Best model saved! (Val Acc: {:.2f}%)".format(val_acc))

# ================= TEST =================
print("\n" + "=" * 60)
print("TESTING (Best Model)")
print("=" * 60)

model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100.0 * test_correct / test_total

print("\nBest Validation Accuracy: {:.2f}%".format(best_val_acc))
print("Test Accuracy:            {:.2f}%".format(test_acc))
print("\nModel saved at:", SAVE_PATH)

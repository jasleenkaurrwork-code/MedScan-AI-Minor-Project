"""
MC Dropout Inference for Breast Cancer (IDC) Binary Classification
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 224
BATCH_SIZE = 16
MC_PASSES = 20
NUM_CLASSES = 2

DATASET_DIR = os.path.join(SCRIPT_DIR, "processed_data_bc"
MODEL_PATH  = os.path.join(SCRIPT_DIR, "DenseNet121_BC.pth"
DISEASE = "BC"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNetMC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.densenet121(weights=None)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)


def main():
    print("Running MC Dropout for:", DISEASE)
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "test"), transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    print("Classes:", test_dataset.classes)
    print("Test samples:", len(test_dataset))

    model = DenseNetMC(NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            suffix = key[len("classifier."):]
            new_key = "base.classifier.1.{}".format(suffix)
        else:
            new_key = "base.{}".format(key)
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    print("Model weights loaded successfully!")

    model = model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    print("Dropout enabled, BatchNorm in eval mode")

    all_preds = []
    all_uncertainty = []
    all_labels = []
    all_confidence = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            mc_outputs = []
            for _ in range(MC_PASSES):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                mc_outputs.append(probs.unsqueeze(0))

            mc_outputs = torch.cat(mc_outputs, dim=0)
            mean_probs = mc_outputs.mean(dim=0)
            variance = mc_outputs.var(dim=0).mean(dim=1)

            preds = torch.argmax(mean_probs, dim=1)
            confidence = mean_probs.max(dim=1).values

            all_preds.extend(preds.cpu().numpy())
            all_uncertainty.extend(variance.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())

    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / len(all_labels)
    print("\nMC Dropout Accuracy: {:.2f}%".format(accuracy * 100))

    df = pd.DataFrame({
        "Label": all_labels, "Prediction": all_preds,
        "Uncertainty": all_uncertainty, "Confidence": all_confidence
    })
    csv_path = os.path.join(SCRIPT_DIR, "MC_results_{}.csv".format(DISEASE))
    df.to_csv(csv_path, index=False)
    print("CSV saved at:", csv_path)

    thresholds = np.linspace(0, max(all_uncertainty), 30)
    coverage_list = []
    accuracy_list = []
    for t in thresholds:
        selected = [i for i, u in enumerate(all_uncertainty) if u <= t]
        if len(selected) == 0:
            continue
        sel_correct = sum(1 for i in selected if all_preds[i] == all_labels[i])
        coverage_list.append(len(selected) / len(all_preds))
        accuracy_list.append(sel_correct / len(selected))

    plt.figure(figsize=(8, 5))
    plt.plot(coverage_list, accuracy_list, marker="o", linewidth=2, color="#2196F3")
    plt.xlabel("Coverage", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Coverage vs Accuracy ({})".format(DISEASE), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "coverage_accuracy_{}.png".format(DISEASE))
    plt.savefig(plot_path, dpi=150)
    print("Graph saved at:", plot_path)
    print("\nMC Dropout + Selective Prediction Completed for", DISEASE)


if __name__ == "__main__":
    main()

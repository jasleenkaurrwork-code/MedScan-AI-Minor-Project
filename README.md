# MedScan AI — Selective Medical Image Diagnosis

**Uncertainty-Aware Deep Learning for Safer Clinical Decision Support**

A multi-domain medical image diagnostic system that uses **DenseNet121** with **Monte Carlo Dropout** to provide uncertainty-aware predictions. The system automatically accepts confident diagnoses and refers uncertain cases to human experts, ensuring clinical safety through selective prediction.

---

## Key Features

- **4 Disease Domains**: Brain Tumor (MRI), Chest X-Ray, Diabetic Retinopathy (Fundus), Breast Cancer (Histopathology)
- **Uncertainty Quantification**: MC Dropout (T=20 passes) produces calibrated confidence and uncertainty scores
- **Selective Prediction**: Automatic accept/refer decisions — uncertain cases are flagged for expert review
- **Visual Explanations**: Grad-CAM heatmaps on DenseNet121's denseblock4 show which regions influenced the diagnosis
- **Image Validation**: GradientBoosting classifier (24 features) prevents cross-modality input errors with ~100% accuracy
- **PDF Reports**: Automated diagnostic reports with patient details, predictions, heatmaps, and clinical explanations
- **Web Interface**: Streamlit app with Glassmorphism UI

---

## Results Summary

| Disease | Classes | Test Images | Test Accuracy | Best Selective Accuracy |
|---|---|---|---|---|
| **Brain Tumor** | 4 (Glioma, Meningioma, No Tumor, Pituitary) | 1,311 | **99.47%** | 100% at 90% coverage |
| **Chest X-Ray** | 2 (Normal, Abnormal) | 3,800 | **82.50%** | 94.17% at 10% coverage |
| **Diabetic Retinopathy** | 2 (No DR, DR Present) | 550 | **98.00%** | 100% at 70% coverage |
| **Breast Cancer** | 2 (Non-IDC, IDC) | 3,000 | **88.20%** | 95.18% at 73.37% coverage |

### Coverage vs Accuracy Curves

<p>
<img src="New Training/BT/coverage_accuracy_BT.png" width="400" alt="Brain Tumor Coverage"/>
<img src="New Training/CX/coverage_accuracy_CX_binary.png" width="400" alt="Chest X-Ray Coverage"/>
</p>
<p>
<img src="New Training/DR/coverage_accuracy_DR_binary.png" width="400" alt="Diabetic Retinopathy Coverage"/>
<img src="New Training/BC/coverage_accuracy_BC.png" width="400" alt="Breast Cancer Coverage"/>
</p>

### Confusion Matrices

<p>
<img src="New Training/BT/confusion_matrix_BT.png" width="400" alt="Brain Tumor Confusion Matrix"/>
<img src="New Training/CX/confusion_matrix_CX.png" width="400" alt="Chest X-Ray Confusion Matrix"/>
</p>
<p>
<img src="New Training/DR/confusion_matrix_DR.png" width="400" alt="Diabetic Retinopathy Confusion Matrix"/>
<img src="New Training/BC/confusion_matrix_BC.png" width="400" alt="Breast Cancer Confusion Matrix"/>
</p>

---

## Project Structure

```
MedScan-AI/
├── README.md                        ← This file
├── .gitignore
│
├── frontend/                        ← Streamlit Web Application
│   ├── app.py                       ← Main app (686 lines): UI + inference + PDF generation
│   ├── gradcam.py                   ← Grad-CAM utility for visual explanations
│   └── image_type_classifier.pkl    ← Trained GradientBoosting classifier (24 features)
│
└── New Training/                    ← Training Code + Models + Results
    ├── generate_final_plots.py      ← Generates confusion matrices and coverage plots
    ├── BT/                          ← Brain Tumor
    │   ├── preprocess_bt.py         ← Data preprocessing (resize, split)
    │   ├── train_bt.py              ← DenseNet121 training script
    │   ├── mc_dropout_bt.py         ← MC Dropout inference (T=20)
    │   ├── threshold_analysis_bt.py ← Selective prediction threshold sweep
    │   ├── DenseNet121_BT.pth       ← Trained model weights (~28 MB)
    │   ├── MC_results_BT.csv        ← MC Dropout per-sample results
    │   ├── confusion_matrix_BT.png
    │   ├── coverage_accuracy_BT.png
    │   └── uncertainty_boxplot_BT.png
    ├── CX/                          ← Chest X-Ray (same structure)
    ├── DR/                          ← Diabetic Retinopathy (same structure)
    └── BC/                          ← Breast Cancer (same structure)
```

---

## How It Works

### Architecture Pipeline

```
User selects scan type → Uploads image
    → Image-Type Validator: 24 features → GradientBoosting → BT/CX/DR/BC
        → IF MISMATCH → Block + error message
        → IF VALID → Continue
    → Preprocess: resize 224×224, RGB, ImageNet normalize
    → DenseNet121 + MC Dropout: 20 stochastic forward passes
    → Mean prediction = final class, Variance = uncertainty
    → Selective Prediction:
        → IF uncertainty ≤ threshold → ACCEPT → show diagnosis + confidence
        → IF uncertainty > threshold → REFER → "Consult medical expert"
    → Grad-CAM heatmap overlay from denseblock4
    → Patient details → Download PDF diagnostic report
```

### MC Dropout Uncertainty Estimation

Each image is processed through `T = 20` forward passes with Dropout(p=0.5) active:

```
Mean prediction:   p̄(x) = (1/T) × Σ p̂ₜ(x)
Uncertainty:       σ²(x) = (1/T) × Σ ‖p̂ₜ(x) − p̄(x)‖²
```

Low variance = model is confident → **ACCEPT**
High variance = model is uncertain → **REFER to expert**

### Selective Prediction Thresholds

| Disease | Threshold (σ²) | Target |
|---|---|---|
| Brain Tumor | 0.000586 | ≥95% selective accuracy |
| Chest X-Ray | 0.000036 | Best achievable |
| Diabetic Retinopathy | 0.037 | ≥95% selective accuracy |
| Breast Cancer | 0.004151 | ≥95% selective accuracy |

---

## Training Configuration

All four models share a unified training pipeline:

| Parameter | Value |
|---|---|
| **Base Model** | DenseNet121 (ImageNet pretrained) |
| **Fine-tuning** | All layers unfrozen |
| **Optimizer** | Adam (lr = 1e-4) |
| **Loss** | Weighted CrossEntropyLoss |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Epochs** | 25 |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Augmentation** | RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), ColorJitter |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| **Framework** | PyTorch 2.1.0+cu128 |

---

## Disease-Specific Details

### Brain Tumor (BT) — 4-Class MRI Classification

| Metric | Value |
|---|---|
| **Dataset** | Brain Tumor MRI Dataset |
| **Total Images** | 7,023 (Train: 4,857 / Val: 855 / Test: 1,311) |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary |
| **Best Val Accuracy** | 99.65% (Epoch 19) |
| **Test Accuracy** | 99.47% (only 7 errors in 1,311 images) |

**Class-wise:** Glioma 99.67%, Meningioma 99.02%, No Tumor 99.51%, Pituitary 99.67%

**Selective Prediction:** 100% accuracy at 90% coverage — only 10% of scans referred.

---

### Chest X-Ray (CX) — Binary Classification

| Metric | Value |
|---|---|
| **Dataset** | CheXpert (Stanford) |
| **Total Images** | 15,200 (Train: 12,160 / Val: 1,520 / Test: 1,520) |
| **Classes** | Normal (No Finding) vs Abnormal (Pleural Effusion, Pneumonia, Pneumothorax) |
| **Best Val Accuracy** | 83.82% (Epoch 4) |
| **Test Accuracy** | 82.50% |

**Why lower accuracy?** CheXpert has noisy NLP-extracted labels and inherent visual ambiguity in chest radiographs.

**Selective Prediction:** 94.17% at 10% coverage — the most confident predictions are highly reliable.

---

### Diabetic Retinopathy (DR) — Binary Classification

| Metric | Value |
|---|---|
| **Dataset** | APTOS 2019 Blindness Detection |
| **Total Images** | 3,662 (Train: 2,563 / Val: 549 / Test: 550) |
| **Classes** | No DR (Grade 0) vs DR Present (Grades 1–4) |
| **Best Val Accuracy** | 99.27% (Epoch 21) |
| **Test Accuracy** | 98.00% |

**Class-wise:** No DR 96.45%, DR Present 99.63% (high sensitivity for disease detection)

**Selective Prediction:** 100% at 70% coverage — exceeds 95% clinical target at full coverage.

---

### Breast Cancer (BC) — Binary IDC Classification

| Metric | Value |
|---|---|
| **Dataset** | IDC Breast Histopathology |
| **Total Images** | 16,000 (Train: 10,000 / Val: 3,000 / Test: 3,000) |
| **Classes** | Non-IDC (Normal) vs IDC (Invasive Ductal Carcinoma) |
| **Best Val Accuracy** | 89.77% (Epoch 6) |
| **Test Accuracy** | 88.20% |

**Class-wise:** Non-IDC 86.67%, IDC 89.73%

**Selective Prediction:** 95.18% at 73.37% coverage — crosses 95% clinical threshold.

---

## Image-Type Validation

A **GradientBoosting classifier** trained on 24 structural and color features prevents cross-modality errors:

| Category | Features |
|---|---|
| **Color (HSV)** | Mean/std saturation, mean brightness, mean hue of saturated pixels |
| **Dark/Bright Ratios** | % pixels below 30/50/80 brightness, % above 200, gray pixel ratio |
| **Edge Analysis** | Canny edge density, horizontal vs vertical Sobel ratio |
| **Spatial** | Center vs border brightness ratio, center/border mean |
| **Circular Region** | Inner circle vs outer ring brightness (dark corners in MRI) |
| **Percentiles** | 5th, 25th, 75th, 95th percentile of brightness |

**Accuracy:** ~100% across 7,261 test images (Precision/Recall/F1 ≈ 1.00 for all 4 modalities)

---

## Frontend — Streamlit Web Application

### UI Theme: Light Glassmorphism
- Soft gradient background (mint green → teal → light blue)
- Semi-transparent cards with `backdrop-filter: blur()`
- Inter font (Google Fonts)
- Color-coded result cards: green (accept), amber (refer), red (error)

### Features
- Drag-and-drop image upload (JPG/PNG)
- Real-time image validation before inference
- Class probability bars with expandable details
- Side-by-side: Original image vs Grad-CAM heatmap
- Patient details form → downloadable PDF diagnostic report

---

## Quick Start

### Prerequisites

```bash
pip install streamlit torch torchvision opencv-python Pillow numpy fpdf2 scikit-learn
```

### Run the Application

```bash
cd frontend
streamlit run app.py
```

Opens at `http://localhost:8501`

### Retrain a Model (example: Diabetic Retinopathy)

```bash
# Step 1: Preprocess dataset
python "New Training/DR/preprocess_dr_binary.py"

# Step 2: Train DenseNet121
python "New Training/DR/train_dr_binary.py"

# Step 3: MC Dropout inference
python "New Training/DR/mc_dropout_dr_binary.py"

# Step 4: Find optimal threshold
python "New Training/DR/threshold_analysis_dr_binary.py"
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | DenseNet121 training and inference |
| `streamlit` | Web application UI |
| `opencv-python` | Image processing, Grad-CAM overlay |
| `Pillow` | Image I/O |
| `numpy` | Array operations |
| `scikit-learn` | GradientBoosting image validator |
| `fpdf2` | PDF diagnostic report generation |
| `matplotlib` | Training plots and visualizations |

---

## Technical Specs

| Component | Details |
|---|---|
| **Models** | 4 × DenseNet121 (~28 MB each) |
| **MC Dropout** | T=20 passes, p=0.5, BatchNorm eval + Dropout train |
| **Grad-CAM Target** | `features.denseblock4` |
| **Inference Time** | ~0.8–1.2s per image (RTX 4050) |
| **Image Validator** | GradientBoosting, 24 features, ~100% accuracy |
| **Frontend** | Streamlit, Light Glassmorphism theme |
| **PDF** | fpdf2-generated reports with patient info + heatmaps |

---

## Authors

| Name | Roll Number |
|---|---|
| Himanshu Pal | 500125939 |
| Jasleen Kaur | 500125601 |
| Bhavya Singh | 500125548 |
| Umesh Mamgai | 500124217 |

**Mentor:** Dr. Sonam Saluja
**Department:** Computer Science and Engineering, UPES Dehradun

---

## License

This project was developed as a Minor Project at the University of Petroleum and Energy Studies (UPES), Dehradun.

# 🖥️ Frontend — MedScan AI Web Application

## Project Title
**Selective Medical Image Diagnosis Using Uncertainty-Aware Deep Learning for Safer Clinical Decision Support**

---

## 📋 Overview

This folder contains the **Streamlit-based web application** that serves as the user-facing interface for the MedScan AI diagnostic system. It integrates all four disease models (Brain Tumor, Chest X-Ray, Diabetic Retinopathy, Breast Cancer) into a single, unified clinical decision support tool with real-time Grad-CAM visual explanations and downloadable PDF diagnostic reports.

---

## 📁 Folder Structure

```
frontend/
├── README.md                    ← This file
├── app.py                       ← Main application (UI + inference + PDF generation)
├── gradcam.py                   ← Grad-CAM utility for DenseNet121 visual explanations
└── image_type_classifier.pkl    ← Trained GradientBoosting classifier for image type detection
```

---

## 📝 Detailed File Descriptions

### 1. `app.py` — Main Application

**What it does:**
- Renders the full web UI using Streamlit with a **Light Glassmorphism** theme
- Provides a sidebar for scan type selection (Brain MRI, Chest X-Ray, Fundus Image, Breast Histopathology)
- Validates uploaded images against the selected scan type using a trained ML classifier
- Loads the appropriate DenseNet121 model and runs **MC Dropout inference** (20 forward passes)
- Computes mean prediction, confidence, and uncertainty (variance)
- Applies selective prediction: accepts confident results or flags uncertain ones for expert referral
- Generates **Grad-CAM heatmap overlays** to visually explain the model's focus areas
- Produces downloadable **PDF diagnostic reports** with patient details, results, visuals, and clinical explanations

**Key Components:**

| Component | Description |
|---|---|
| `DenseNetMC` class | DenseNet121 wrapper with MC Dropout (p=0.5) before classifier |
| `detect_image_type()` | ML-based image type classification (BT / CX / DR / BC) using 24 structural + color features with ~100% accuracy |
| `run_mc_inference()` | Executes 20 forward passes with Dropout enabled for uncertainty estimation |
| `get_clinical_explanation()` | Generates disease-specific clinical text based on prediction results |
| `generate_pdf_report()` | Creates formatted PDF reports using fpdf2 with patient info, visuals, and diagnosis |

### 2. `gradcam.py` — Grad-CAM Visual Explanation

**What it does:**
- Implements the **Grad-CAM** algorithm for DenseNet121
- Hooks into `denseblock4` (the last convolutional block) to capture activations and gradients
- Generates class-activation heatmaps showing which image regions influenced the prediction
- Overlays the heatmap on the original image with a JET colormap

**Key Functions:**

| Function | Purpose |
|---|---|
| `GradCAM.__init__()` | Registers forward and backward hooks on the target layer |
| `GradCAM.generate()` | Runs forward/backward pass, computes weighted activation map |
| `apply_heatmap()` | Overlays Grad-CAM heatmap on the original image |
| `get_gradcam_for_densenet()` | Complete pipeline: model → Grad-CAM → heatmap overlay |

---

## 🎨 UI Design

### Theme: Light Glassmorphism
| Element | Style |
|---|---|
| Background | Soft gradient: mint green → teal → light blue |
| Sidebar | Same gradient, seamless with main area |
| Cards/Containers | Semi-transparent white with `backdrop-filter: blur()` |
| Title/Headings | Deep teal (#00695c) and forest green (#1b5e20) |
| Text | Dark slate (#37474f) for readability |
| Success Results | Light green card with green left border |
| Refer to Expert | Light amber card with orange left border |
| Image Mismatch Error | Light pink card with red left border |
| Typography | Inter font (Google Fonts) |

### Features
- **File uploader** with drag-and-drop support (JPG, JPEG, PNG)
- **Image type validation** prevents mismatched uploads before inference
- **Detailed class probabilities** shown in expandable section with progress bars
- **Side-by-side visual comparison**: Original image vs Grad-CAM heatmap
- **Patient details form** (Name, Age 0–100, Gender) for PDF report generation
- **Download button** for PDF diagnostic report

---

## 🛡️ Image Type Validation

The system validates uploaded images **before** running heavy model inference using a **trained GradientBoosting classifier** with 24 structural + color features:

### Features Used (24 total)
| Category | Features |
|---|---|
| **Color (HSV)** | Mean/std saturation, mean brightness, std brightness, mean hue of saturated pixels |
| **Dark/Bright Ratios** | % pixels below 30/50/80 brightness, % pixels above 200 brightness, gray pixel ratio |
| **Edge Analysis** | Canny edge density, horizontal vs vertical Sobel edge ratio (detects ribs in X-rays) |
| **Spatial Brightness** | Center vs border brightness ratio, center mean, border mean |
| **Circular Region** | Inner circle vs outer ring brightness ratio (detects dark corners in MRI) |
| **Percentiles** | 5th, 25th, 75th, 95th percentile of brightness |

### Detection Results

| Detected Category | Description | Accepted For |
|---|---|---|
| `BT` | Brain MRI (grayscale, bright center, dark corners) | Brain Tumor ✅ only |
| `CX` | Chest X-Ray (grayscale, high edge density, ribs) | Chest X-Ray ✅ only |
| `DR` | Fundus image (colorful, red/orange hue) | Diabetic Retinopathy ✅ only |
| `BC` | Histopathology (colorful, pink/purple hue) | Breast Cancer ✅ only |

**Validation Accuracy:** ~100% across 7,261 test images

| Disease | Precision | Recall | F1-Score |
|---|---|---|---|
| Brain Tumor (BT) | 1.00 | 0.99 | 1.00 |
| Chest X-Ray (CX) | 1.00 | 1.00 | 1.00 |
| Diabetic Retinopathy (DR) | 1.00 | 1.00 | 1.00 |
| Breast Cancer (BC) | 1.00 | 1.00 | 1.00 |

---

## 📄 PDF Report

The downloadable diagnostic report includes:

| Section | Content |
|---|---|
| **Header** | MedScan AI branding, teal header bar, generated date/time (top-right) |
| **Diagnostic Report** | White text on teal banner |
| **Patient Information** | Full Name, Date, Age, Gender, Scan Type |
| **Diagnosis** | Result (class or Refer to Expert), Predicted Class, Confidence %, Uncertainty, Decision |
| **Visual Explanation** | Side-by-side Original Image and Grad-CAM Heatmap |
| **Clinical Explanation** | Disease-specific text explaining the model's findings |

---

## 🔧 How to Run

```bash
# Navigate to the frontend directory
cd frontend

# Run the Streamlit application
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `torch` / `torchvision` | Deep learning inference (DenseNet121) |
| `opencv-python` | Image processing, HSV analysis, Grad-CAM overlay |
| `Pillow` | Image loading and manipulation |
| `numpy` | Array operations |
| `scikit-learn` | GradientBoosting classifier for image type detection |
| `fpdf2` | PDF report generation |

**Install all dependencies:**
```bash
pip install streamlit torch torchvision opencv-python Pillow numpy fpdf2 scikit-learn
```

---

## 🛠️ Technical Details

| Parameter | Value |
|---|---|
| **Framework** | Streamlit |
| **Model** | DenseNet121 with MC Dropout (p=0.5) |
| **MC Dropout Passes** | T = 20 per image |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Normalization** | ImageNet: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| **Uncertainty Threshold** | Configurable per disease (see model configs) |
| **Image Validation** | Trained GradientBoosting classifier (24 features, ~100% accuracy) |
| **Grad-CAM Target** | `denseblock4` (last conv block of DenseNet121) |
| **PDF Library** | fpdf2 |
| **Theme** | Light Glassmorphism (white/teal/green palette) |

---

## 📌 Application Flow

```
User opens http://localhost:8501
    → Selects scan type from sidebar (Brain MRI / Chest X-Ray / Fundus / Histopathology)
    → Uploads medical image (JPG/JPEG/PNG)
    → System validates image type using HSV color analysis
        → IF mismatch → Shows error, asks for correct image
        → IF valid → Proceeds to inference
    → Loads DenseNet121 model for selected disease
    → Runs 20 MC Dropout forward passes
    → Computes mean prediction + uncertainty (variance)
    → IF uncertainty ≤ threshold → Shows predicted class + confidence
    → ELSE → Shows "Refer to Medical Expert" warning
    → Generates Grad-CAM heatmap overlay on uploaded image
    → Displays results: Original vs Heatmap, Clinical Explanation
    → User fills patient details (Name, Age, Gender)
    → Downloads PDF Diagnostic Report
```

---

## 🔗 Model Dependencies

This frontend loads trained models from the parent directory:

| Disease | Model File | Location |
|---|---|---|
| Brain Tumor | `DenseNet121_BT.pth` | `../New Training/BT/` |
| Chest X-Ray | `DenseNet121_CX_binary.pth` | `../New Training/CX/` |
| Diabetic Retinopathy | `DenseNet121_DR_binary.pth` | `../New Training/DR/` |
| Breast Cancer | `DenseNet121_BC.pth` | `../New Training/BC/` |

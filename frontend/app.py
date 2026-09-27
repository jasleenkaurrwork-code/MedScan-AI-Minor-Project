"""
MedScan AI — Selective Medical Image Diagnosis
Uncertainty-Aware Deep Learning for Safer Clinical Decision Support
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os, sys, io, tempfile, pickle
from datetime import datetime
from fpdf import FPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcam import get_gradcam_for_densenet

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MedScan AI — Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== THEME: Light Glassmorphism ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #e0f2f1 35%, #e3f2fd 70%, #f1f8e9 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e9 0%, #e0f2f1 50%, #e3f2fd 100%);
        border-right: 1px solid rgba(0, 150, 136, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #00796b !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #37474f !important;
    }

    .main-title {
        text-align: center;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 28px;
        border: 1px solid rgba(0, 150, 136, 0.2);
        box-shadow: 0 8px 32px rgba(0, 150, 136, 0.1);
    }
    .main-title h1 {
        color: #00695c !important;
        font-size: 2.4em;
        margin: 0;
        letter-spacing: 2px;
        font-weight: 700;
    }
    .main-title p {
        color: #546e7a !important;
        font-size: 1em;
        margin-top: 8px;
    }

    .result-success {
        background: rgba(232, 245, 233, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-left: 4px solid #43a047;
        border-radius: 14px;
        padding: 22px;
        margin: 12px 0;
    }
    .result-refer {
        background: rgba(255, 243, 224, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-left: 4px solid #fb8c00;
        border-radius: 14px;
        padding: 22px;
        margin: 12px 0;
    }
    .result-success h3 { color: #2e7d32 !important; }
    .result-refer h3 { color: #e65100 !important; }
    .result-success p, .result-refer p { color: #33403a !important; }

    .info-box {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 150, 136, 0.15);
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
    }
    .info-box h4 { color: #00796b !important; }
    .info-box p { color: #37474f !important; }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 150, 136, 0.15);
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #546e7a !important; }
    [data-testid="stMetricValue"] { color: #00695c !important; }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1b5e20 !important; }
    .stMarkdown p { color: #37474f !important; }

    /* File uploader filename text */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] .uploadedFileName {
        color: #1b5e20 !important;
    }
    [data-testid="stFileUploaderFileName"] { color: #1b5e20 !important; }

    /* Expander header text */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary span {
        color: #1b5e20 !important;
    }

    /* Form input labels */
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label {
        color: #1b5e20 !important;
    }

    /* Progress bar text inside expander */
    [data-testid="stExpander"] [data-testid="stProgressBarText"],
    [data-testid="stExpander"] .stProgress p,
    [data-testid="stExpander"] p {
        color: #1b5e20 !important;
    }

    /* Drag-and-drop helper text (Limit, JPG, etc.) */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #ffffff !important;
    }

    /* File uploader dismiss (X) button */
    [data-testid="stFileUploader"] button[aria-label="Remove file"] svg,
    [data-testid="stFileUploader"] button[kind="minimal"] svg {
        color: #c62828 !important;
        fill: #c62828 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL CONFIG ====================
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "New Training")

DISEASE_CONFIG = {
    "Brain MRI (Brain Tumor)": {
        "key": "BT",
        "model_path": os.path.join(BASE_DIR, "BT", "DenseNet121_BT.pth"),
        "num_classes": 4,
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "threshold": 0.000586,
        "description": "Detects and classifies brain tumors from MRI scans into 4 types."
    },
    "Fundus Image (Diabetic Retinopathy)": {
        "key": "DR",
        "model_path": os.path.join(BASE_DIR, "DR", "DenseNet121_DR_binary.pth"),
        "num_classes": 2,
        "classes": ["No DR", "DR Present"],
        "threshold": 0.037,
        "description": "Detects Diabetic Retinopathy from retinal fundus images."
    },
    "Chest X-Ray": {
        "key": "CX",
        "model_path": os.path.join(BASE_DIR, "CX", "DenseNet121_CX_binary.pth"),
        "num_classes": 2,
        "classes": ["Normal", "Abnormal"],
        "threshold": 0.000036,
        "description": "Detects chest abnormalities from X-ray images."
    },
    "Breast Histopathology": {
        "key": "BC",
        "model_path": os.path.join(BASE_DIR, "BC", "DenseNet121_BC.pth"),
        "num_classes": 2,
        "classes": ["Non-IDC (Normal)", "IDC (Cancer)"],
        "threshold": 0.004151,
        "description": "Detects Invasive Ductal Carcinoma from histopathology slides."
    }
}

MC_PASSES = 20
IMG_SIZE = 224


# ==================== MODEL ====================
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


@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetMC(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            new_key = "base.classifier.1.{}".format(key[len("classifier."):])
        else:
            new_key = "base.{}".format(key)
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


@st.cache_resource
def load_image_classifier():
    """Load the trained image-type classifier (GradientBoosting)."""
    clf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_type_classifier.pkl")
    with open(clf_path, "rb") as f:
        return pickle.load(f)


def detect_image_type(image):
    """Detect the category of a medical image using a trained ML classifier.
    Uses 24 structural + color features (HSV, edges, spatial brightness).
    Returns: 'BT', 'CX', 'DR', or 'BC'
    """
    img = np.array(image.resize((224, 224)))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    h = hsv[:, :, 0].astype(float)
    s = hsv[:, :, 1].astype(float)
    v = hsv[:, :, 2].astype(float)

    mean_s = s.mean()
    mean_v = v.mean()
    std_v = v.std()
    sat_mask = s > 30
    sat_count = sat_mask.sum()
    mean_h_sat = h[sat_mask].mean() if sat_count > 100 else 0

    # Edge detection
    edges = cv2.Canny(img, 50, 150)
    edge_density = (edges > 0).mean()

    # Center vs border brightness
    center = v[56:168, 56:168]
    border_mask = np.ones_like(v, dtype=bool)
    border_mask[56:168, 56:168] = False
    border = v[border_mask]
    center_mean = center.mean()
    border_mean = border.mean()
    center_border_ratio = center_mean / (border_mean + 1e-6)

    # Circular region analysis (MRI has dark corners)
    Y, X = np.ogrid[:224, :224]
    dist = np.sqrt((X - 112)**2 + (Y - 112)**2)
    outer_ring = (dist > 90) & (dist <= 112)
    inner_circle = dist <= 70
    outer_brightness = v[outer_ring].mean()
    inner_brightness = v[inner_circle].mean()
    ring_ratio = inner_brightness / (outer_brightness + 1e-6)

    # Horizontal vs vertical edge ratio (ribs in X-rays)
    sobelx = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    horiz_edge_ratio = np.abs(sobely).mean() / (np.abs(sobelx).mean() + 1e-6)

    features = np.array([
        mean_s, s.std(), mean_v, std_v, mean_h_sat,
        (v < 30).mean(), (v < 50).mean(), (v < 80).mean(),
        (v > 200).mean(), (s < 25).mean(),
        sat_count / (224 * 224),
        edge_density,
        center_border_ratio, center_mean, border_mean,
        ring_ratio, inner_brightness, outer_brightness,
        horiz_edge_ratio,
        gray.std(),
        np.percentile(v, 5), np.percentile(v, 25),
        np.percentile(v, 75), np.percentile(v, 95),
    ]).reshape(1, -1)

    clf = load_image_classifier()
    return clf.predict(features)[0]


def run_mc_inference(model, input_tensor, device, num_passes=20):
    mc_outputs = []
    with torch.no_grad():
        for _ in range(num_passes):
            output = model(input_tensor.to(device))
            probs = torch.softmax(output, dim=1)
            mc_outputs.append(probs.unsqueeze(0))
    mc_outputs = torch.cat(mc_outputs, dim=0)
    mean_probs = mc_outputs.mean(dim=0).squeeze()
    variance = mc_outputs.var(dim=0).mean(dim=1).squeeze()
    predicted_class = mean_probs.argmax().item()
    confidence = mean_probs[predicted_class].item()
    uncertainty = variance.item()
    return predicted_class, confidence, uncertainty, mean_probs.cpu().numpy()


def generate_explanation(disease_key, predicted_class, class_name, confidence, uncertainty, is_referred):
    explanations = {
        "BT": {
            "success": "The brain MRI scan was analyzed across {} MC Dropout forward passes. The model identified consistent neural activation patterns, indicating a high-confidence classification as {}. The uncertainty score of {:.6f} is well below the safety threshold.".format(MC_PASSES, class_name, uncertainty),
            "refer": "The brain MRI scan showed ambiguous or overlapping features across {} forward passes. The uncertainty score of {:.6f} exceeds the safety threshold. Immediate specialist review is recommended.".format(MC_PASSES, uncertainty)
        },
        "DR": {
            "success": "The retinal fundus image was analyzed for Diabetic Retinopathy. The model consistently detected {} across {} passes with {:.1f}% confidence. The low uncertainty of {:.6f} confirms unambiguous retinal patterns.".format(class_name, MC_PASSES, confidence * 100, uncertainty),
            "refer": "The retinal fundus image contains features that could not be consistently classified across {} passes. The uncertainty of {:.6f} suggests subtle findings requiring expert ophthalmological evaluation.".format(MC_PASSES, uncertainty)
        },
        "CX": {
            "success": "The chest X-ray was analyzed for abnormalities. The model classified it as {} with {:.1f}% confidence across {} MC passes with uncertainty {:.6f}, indicating clear radiographic features.".format(class_name, confidence * 100, MC_PASSES, uncertainty),
            "refer": "The chest X-ray contains ambiguous opacities or overlapping features that caused inconsistent predictions across {} passes (uncertainty: {:.6f}). Radiologist review is strongly recommended.".format(MC_PASSES, uncertainty)
        },
        "BC": {
            "success": "The histopathology slide was analyzed for Invasive Ductal Carcinoma. The model identified {} with {:.1f}% confidence. The low uncertainty ({:.6f}) confirms clear cellular morphology patterns.".format(class_name, confidence * 100, uncertainty),
            "refer": "The histopathology slide contains cellular features with ambiguous characteristics across {} forward passes. The uncertainty of {:.6f} suggests mixed tissue morphology. Pathologist review is recommended.".format(MC_PASSES, uncertainty)
        }
    }
    return explanations[disease_key]["refer"] if is_referred else explanations[disease_key]["success"]


# ==================== PDF REPORT ====================
def generate_pdf_report(name, age, gender, scan_type, class_name, confidence,
                        uncertainty, is_referred, explanation, original_img, heatmap_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Header bar
    pdf.set_fill_color(0, 121, 107)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_fill_color(77, 182, 172)
    pdf.rect(0, 38, 210, 2, "F")

    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "MedScan AI", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 210, 190)
    pdf.cell(0, 6, "Selective Medical Image Diagnosis | Uncertainty-Aware Deep Learning", ln=True, align="C")

    # Generated on date in top-right
    pdf.set_y(8)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(200, 230, 210)
    pdf.cell(0, 5, datetime.now().strftime("%d/%m/%Y %I:%M %p"), align="R", ln=True)

    pdf.set_y(42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(0, 121, 107)
    pdf.cell(0, 12, "  DIAGNOSTIC REPORT", ln=True, fill=True)

    pdf.ln(4)
    pdf.set_draw_color(77, 182, 172)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Patient Info
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 121, 107)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(95, 7, "Full Name:  {}".format(name))
    pdf.cell(95, 7, "Date:  {}".format(datetime.now().strftime("%d/%m/%Y")), ln=True)
    pdf.cell(95, 7, "Age:  {} years".format(age))
    pdf.cell(95, 7, "Gender:  {}".format(gender), ln=True)
    pdf.cell(95, 7, "Scan Type:  {}".format(scan_type), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(77, 182, 172)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Diagnosis
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 121, 107)
    pdf.cell(0, 8, "Diagnosis", ln=True)
    pdf.set_font("Helvetica", "B", 11)
    if is_referred:
        pdf.set_text_color(180, 120, 20)
        pdf.cell(0, 7, "Result:  REFER TO MEDICAL EXPERT", ln=True)
    else:
        pdf.set_text_color(30, 120, 60)
        pdf.cell(0, 7, "Result:  {}".format(class_name), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(95, 7, "Predicted Class:  {}".format(class_name))
    pdf.cell(95, 7, "Confidence:  {:.1f}%".format(confidence * 100), ln=True)
    pdf.cell(95, 7, "Uncertainty:  {:.6f}".format(uncertainty))
    pdf.cell(95, 7, "Decision:  {}".format("Refer to Expert" if is_referred else "Accepted (Confident)"), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(77, 182, 172)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Visual Explanation
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 121, 107)
    pdf.cell(0, 8, "Visual Explanation (Grad-CAM)", ln=True)
    pdf.ln(2)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
        original_img.resize((300, 300)).save(f1.name)
        orig_path = f1.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        heatmap_img.resize((300, 300)).save(f2.name)
        heat_path = f2.name

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    x_start = 15
    pdf.cell(85, 5, "Original Image", align="C")
    pdf.cell(10, 5, "")
    pdf.cell(85, 5, "Model Focus Area (Heatmap)", align="C", ln=True)
    pdf.image(orig_path, x=x_start, w=85)
    pdf.image(heat_path, x=x_start + 95, y=pdf.get_y() - 85, w=85)

    os.unlink(orig_path)
    os.unlink(heat_path)

    pdf.ln(4)
    pdf.set_draw_color(77, 182, 172)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Clinical Explanation
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 121, 107)
    pdf.cell(0, 8, "Clinical Explanation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    clean_text = explanation.replace("**", "")
    pdf.multi_cell(0, 6, clean_text)

    # Footer
    pdf.ln(8)
    pdf.set_draw_color(77, 182, 172)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())

    return bytes(pdf.output())


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🩺 MedScan AI")
    st.markdown("---")

    st.markdown("#### Select Scan Type")
    scan_type = st.selectbox(
        "Choose the medical scan type:",
        list(DISEASE_CONFIG.keys()),
        label_visibility="collapsed"
    )
    config = DISEASE_CONFIG[scan_type]

    st.markdown("---")
    st.markdown("#### 📋 About This Scan")
    st.markdown(config["description"])


# ==================== MAIN ====================
st.markdown("""
<div class="main-title">
    <h1>🩺 MedScan AI</h1>
    <p>Selective Medical Image Diagnosis Using Uncertainty-Aware Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload Medical Image")
uploaded_file = st.file_uploader(
    "Upload your medical scan image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ==================== ANALYSIS ====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.markdown("### 🔬 Analysis Results")

    # Image type validation
    detected_type = detect_image_type(image)
    expected_type = config["key"]

    if detected_type != expected_type:
        type_names = {
            "BT": "Brain MRI (grayscale, dark background)",
            "CX": "Chest X-Ray (grayscale, bright)",
            "DR": "Fundus Image (colorful, red/orange)",
            "BC": "Histopathology (pink/purple tissue)"
        }
        st.markdown("""
        <div style="background: rgba(255, 235, 238, 0.85);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(198, 40, 40, 0.3); border-left: 4px solid #c62828;
                    border-radius: 14px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #c62828 !important;">🚫 IMAGE MISMATCH DETECTED</h3>
            <p style="color: #37474f !important;"><strong>You selected:</strong> {sel}</p>
            <p style="color: #37474f !important;"><strong>But the uploaded image looks like:</strong> {det}</p>
            <p style="color: #e65100 !important;"><strong>👉 Please upload a relevant image that matches your selected scan type.</strong></p>
        </div>
        """.format(sel=scan_type, det=type_names.get(detected_type, "Unknown")), unsafe_allow_html=True)
        st.image(image.resize((300, 300)), caption="Uploaded Image", use_container_width=False)
        st.stop()

    # Load model & run inference
    with st.spinner("Loading {} model...".format(config["key"])):
        model, device = load_model(config["model_path"], config["num_classes"])

    input_tensor = preprocess_image(image)

    with st.spinner("Running {} MC Dropout passes...".format(MC_PASSES)):
        predicted_class, confidence, uncertainty, class_probs = run_mc_inference(
            model, input_tensor, device, MC_PASSES
        )

    with st.spinner("Generating visual explanation (Grad-CAM)..."):
        gradcam_tensor = preprocess_image(image).to(device)
        heatmap_image, _, _ = get_gradcam_for_densenet(
            model, gradcam_tensor, image, target_class=predicted_class
        )

    class_name = config["classes"][predicted_class]
    is_referred = uncertainty > config["threshold"]
    explanation = generate_explanation(
        config["key"], predicted_class, class_name,
        confidence, uncertainty, is_referred
    )

    # ==================== DISPLAY RESULTS ====================

    if is_referred:
        st.markdown("""
        <div class="result-refer">
            <h3>⚠️ REFER TO MEDICAL EXPERT</h3>
            <p><strong>Status:</strong> Prediction abstained for patient safety</p>
            <p><strong>Reason:</strong> High uncertainty detected — model is not confident enough</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-success">
            <h3>✅ DIAGNOSIS: {}</h3>
            <p><strong>Confidence:</strong> {:.1f}%</p>
            <p><strong>Status:</strong> Model is confident in this prediction</p>
        </div>
        """.format(class_name, confidence * 100), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Class", class_name)
    with col2:
        st.metric("Confidence", "{:.1f}%".format(confidence * 100))
    with col3:
        st.metric("Uncertainty", "{:.6f}".format(uncertainty))
    with col4:
        st.metric("Decision", "🟢 Accept" if not is_referred else "🟡 Refer")

    st.markdown("---")

    # Visual Explanation
    st.markdown("### 🖼️ Visual Explanation (Grad-CAM)")
    col_orig, col_gradcam = st.columns(2)
    with col_orig:
        st.markdown("**Original Image**")
        st.image(image.resize((350, 350)), use_container_width=True)
    with col_gradcam:
        st.markdown("**Model Focus Area (Heatmap)**")
        st.image(heatmap_image.resize((350, 350)), use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <h4>🔍 Reading the Heatmap</h4>
        <p><strong style="color: #e74c3c;">Red/Yellow areas</strong> = Regions the model focused on most heavily.<br>
        <strong style="color: #3498db;">Blue/Green areas</strong> = Regions with less influence on the prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Clinical Explanation
    st.markdown("### 📝 Clinical Explanation")
    st.markdown("""
    <div class="info-box">
        <p>{}</p>
    </div>
    """.format(explanation), unsafe_allow_html=True)

    # Class Probabilities
    with st.expander("📊 Detailed Class Probabilities"):
        for i, cls_name in enumerate(config["classes"]):
            prob = class_probs[i]
            st.progress(float(prob), text="{}: {:.2f}%".format(cls_name, prob * 100))

    st.markdown("---")

    # ==================== DOWNLOAD REPORT ====================
    st.markdown("### 📥 Download Diagnostic Report")
    st.markdown("""
    <div class="info-box">
        <p>Fill in patient details below to generate a downloadable PDF report.</p>
    </div>
    """, unsafe_allow_html=True)

    col_name, col_age, col_gender = st.columns(3)
    with col_name:
        patient_name = st.text_input("Full Name", placeholder="Enter patient's full name")
    with col_age:
        patient_age = st.selectbox("Age", options=list(range(0, 101)), index=25)
    with col_gender:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say", "Other"])

    if st.button("📄 Generate & Download Report", use_container_width=True):
        if not patient_name.strip():
            st.warning("Please enter the patient's full name.")
        else:
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(
                    name=patient_name.strip(),
                    age=patient_age,
                    gender=patient_gender,
                    scan_type=scan_type,
                    class_name=class_name,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    is_referred=is_referred,
                    explanation=explanation,
                    original_img=image,
                    heatmap_img=heatmap_image
                )
            st.download_button(
                label="⬇️ Download Report (PDF)",
                data=pdf_bytes,
                file_name="MedScan_Report_{}.pdf".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
                mime="application/pdf",
                use_container_width=True
            )
            st.success("Report generated successfully!")

import os
import warnings

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. CLEAN APP CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Face Detector", page_icon="👤", layout="centered"
)

# Hide default Streamlit clutter & Import Google Font (Inter)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        * {font-family: 'Inter', sans-serif;}
    </style>
""",
    unsafe_allow_html=True,
)


# --- 2. MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )

    model_path = os.path.join("model", "deepfake_detector.pth")
    if not os.path.exists(model_path):
        return None

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# --- 3. PREDICTION LOGIC ---
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image, model):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = "Real Face" if predicted.item() == 0 else "Deepfake Face"
    confidence_score = confidence.item() * 100

    return label, confidence_score


# --- 4. FRONTEND UI DESIGN ---

# Header Section
st.markdown(
    """
    <div style="text-align: center; padding-bottom: 10px;">
        <h1 style="color: #0F172A; margin-bottom: 5px; font-weight: 700; letter-spacing: -0.5px;">
            👤 Face Deepfake Detector
        </h1>
        <p style="color: #64748B; font-size: 1.1em; margin-bottom: 15px;">
            Verify if a human face is authentic or AI-generated
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Redesigned "Test Image" Link (Sleek Pill Design)
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="display: inline-block; background-color: #F1F5F9; padding: 8px 16px; border-radius: 20px; color: #475569; font-size: 0.9em; margin: 0; border: 1px solid #E2E8F0;">
            Don't have a picture? 
            <a href="https://thispersondoesnotexist.com/" target="_blank" style="color: #2563EB; text-decoration: none; font-weight: 600; margin-left: 4px;">
                Generate a synthetic face ↗
            </a>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.error(
        "Model file not found! Ensure 'deepfake_detector.pth' is in the 'model' folder."
    )
else:
    # Updated Uploader Text
    uploaded_file = st.file_uploader(
        "**Upload a portrait to analyze the face:**",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.markdown(
            "<hr style='margin-top: 10px; margin-bottom: 30px; border-color: #F1F5F9;'>",
            unsafe_allow_html=True,
        )

        image = Image.open(uploaded_file).convert("RGB")

        img_col, result_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown(
                "<p style='color: #64748B; font-weight: 600; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;'>Uploaded Portrait</p>",
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)

        with result_col:
            st.markdown(
                "<p style='color: #64748B; font-weight: 600; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;'>Analysis Result</p>",
                unsafe_allow_html=True,
            )

            with st.spinner("Analyzing facial features and noise..."):
                label, confidence_score = predict(image, model)

            if label == "Deepfake Face":
                bg_color, border_color, text_color = "#FEF2F2", "#FCA5A5", "#991B1B"
                icon = "🚨"
            else:
                bg_color, border_color, text_color = "#F0FDF4", "#86EFAC", "#166534"
                icon = "✅"

            st.markdown(
                f"""
            <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;">
                <h3 style="color: {text_color}; margin-top: 0; font-size: 1.6em; font-weight: 700;">{icon} {label}</h3>
                <p style="color: {text_color}; font-size: 1.1em; margin-bottom: 0; opacity: 0.9;">Confidence: <b style="font-size: 1.2em;">{confidence_score:.1f}%</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<p style='font-size: 0.85em; color: #64748B; margin-bottom: 5px; font-weight: 600;'>CONFIDENCE LEVEL</p>",
                unsafe_allow_html=True,
            )
            confidence_bar = st.progress(0)
            confidence_bar.progress(int(confidence_score))

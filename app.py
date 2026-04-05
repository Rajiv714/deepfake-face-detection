import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    model_path = os.path.join("model", "deepfake_detector.pth")

    if not os.path.exists(model_path):
        st.error("Model file not found! Ensure 'deepfake_detector.pth' is in the 'model' folder.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to predict image class with confidence
def predict(image, model):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        confidence, predicted = torch.max(probabilities, 1)

    label = "**Real Image**" if predicted.item() == 0 else "**Deepfake Image**"
    confidence_score = confidence.item() * 100  # Convert to percentage

    return label, confidence_score
    
# Streamlit UI
st.markdown(
    """
    <h2 style="
        text-align: center;
        font-family: Arial, sans-serif;
        color: #B22222;  /* Dark red */
        text-shadow: 1px 1px 3px #CD5C5C;  /* Lighter shadow with minimal blur */
        margin-bottom: 20px;">
        🔎 Deepfake Image Detector
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown("###### To Generate and Download DeepFake Image Click  [Here](https://thispersondoesnotexist.com/)")
# File uploader
uploaded_file = st.file_uploader("**Want to know if it's a deepfake? Upload the image!**",type=["jpg", "jpeg", "png"],accept_multiple_files=False,)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # Center the image using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image,width=300)

    model = load_model()
    
    if model:  # Properly indented inside the 'if uploaded_file is not None' block
        label, confidence_score = predict(image, model)
        st.write("##### Prediction: " + f" It is a **{label}** ")
        st.write(f"**Confidence:** {confidence_score:.2f}%")
        confidence_bar = st.progress(0)
        confidence_bar.progress(int(confidence_score))
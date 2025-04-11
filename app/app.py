import streamlit as st
from PIL import Image
import sys
import os
from config import DATASET_PATH, MODEL_PATH
from model import load_model, predict, generate_gradcam
from utils import load_class_names
from ui import render_sidebar, render_title, render_uploaded_image, render_predictions, render_gradcam

# Fix for Streamlit + torch.classes RuntimeError
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# Page configuration
st.set_page_config(
    page_title="Bird Species Classifier",
    layout="centered",
    page_icon="🦜"
)

# Sidebar controls
top_k, show_gradcam = render_sidebar()

# Render title and description
render_title()

# Load class names and model
class_names = load_class_names(DATASET_PATH)
model = load_model(MODEL_PATH, len(class_names))

# Upload image
st.markdown("### 📤 Upload a Bird Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    predictions = predict(image, model, class_names, top_k)

    render_uploaded_image(image)
    render_predictions(predictions)

    if show_gradcam:
        gradcam = generate_gradcam(image, model, class_names)
        render_gradcam(gradcam)

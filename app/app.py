import streamlit as st
from PIL import Image
import sys
import os

import streamlit as st

# ✅ MUST be the very first Streamlit command
st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Check secrets AFTER setting page config
APP_ONLINE = st.secrets.get("APP_ONLINE", "true").lower() == "true"

# 🚧 Maintenance mode screen
if not APP_ONLINE:
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF6B6B;'>🚧 Under Maintenance</h1>
        <p style='text-align: center; font-size: 20px;'>
            We're currently making improvements.<br>
            Please check back soon. 🙏
        </p>
        """,
        unsafe_allow_html=True
    )
    st.stop()


# Continue with normal app
from config import DATASET_PATH, MODEL_PATH
from model import load_model, predict, generate_gradcam
from utils import load_class_names
from ui import render_sidebar, render_title, render_uploaded_image, render_predictions, render_gradcam

# Ensure project root is in the path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# Fix for Streamlit + torch.classes RuntimeError
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

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
    render_uploaded_image(image)
    predictions = predict(image, model, class_names, top_k)
    render_predictions(predictions)

    if show_gradcam:
        gradcam = generate_gradcam(image, model, class_names)
        render_gradcam(gradcam)

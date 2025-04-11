import streamlit as st
from PIL import Image

# Global CSS for custom font and sleek dark theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #0f1117;
            color: #f0f0f0;
        }

        h1, h2, h3, h4 {
            color: #00C6A7;
        }

        .stButton > button {
            background-color: #00C6A7;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #00b39f;
            transform: scale(1.02);
        }

        .prediction-box {
            background-color: rgba(28, 30, 38, 0.75);
            backdrop-filter: blur(10px);
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 0 12px rgba(0, 198, 167, 0.1);
            margin-top: 1rem;
        }

        .center-text {
            text-align: center;
        }

        .stMarkdown code {
            background-color: #1c1e26;
        }

        .css-1v0mbdj, .css-1aumxhk {
            background-color: #1c1e26 !important;
            border-radius: 10px;
        }

        .css-1aumxhk h1 {
            color: #00C6A7;
        }
    </style>
""", unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.title("🔧 Options")

    top_k = st.sidebar.slider("Top-K Predictions", 1, 5, 3)
    show_gradcam = st.sidebar.button("Generate Grad-CAM")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Model Info")
    st.sidebar.markdown("**Architecture:** ResNet50  \n**Input Size:** 224x224  \n**Grad-CAM:** Optional")

    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Dataset Info", expanded=False):
        st.markdown("""
        **Name:** CUB-200-2011  
        **Published by:** Caltech  
        **Species:** 200 bird species  
        **Images:** ~11,788  
        **Annotations:**  
        - Bounding boxes  
        - Part locations  
        - Attribute labels  

        📎 [DOI: 10.22002/D1.20098](https://doi.org/10.22002/D1.20098)  
        🗓️ **Published:** April 11, 2022  
        👤 **Authors:**  
        - Catherine Wah  
        - Steve Branson  
        - Peter Welinder  
        - Pietro Perona  
        - Serge Belongie  
        """)

    return top_k, show_gradcam


def render_title():
    st.markdown("<h1 class='center-text'>🦜 Bird Species Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center-text'>Upload a bird image to classify and explore model attention with Grad-CAM.</p>", unsafe_allow_html=True)
    st.markdown("---")


def render_uploaded_image(image):
    st.markdown("#### 🖼️ Uploaded Image")
    st.image(image, width=400)


def render_predictions(predictions):
    st.markdown("#### 🔍 Top Predictions")
    for cls, prob in predictions:
        percent_text = f"{prob * 100:.2f}%"
        percent_width = int(prob * 100)
        bar_width = f"width: {max(percent_width, 4)}%;" if percent_width > 0 else "width: 0%;"

        st.markdown(f"""
            <div class='prediction-box'>
                <strong>{cls}</strong>: {percent_text}
                <div style="height: 10px; background-color: #55555533; border-radius: 5px;">
                    <div style="{bar_width} height: 10px; background-color: #00C6A7; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_gradcam(gradcam_image):
    st.markdown("#### 🌡️ Grad-CAM Heatmap")
    st.image(gradcam_image, width=400)

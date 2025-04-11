import streamlit as st
from PIL import Image

def render_sidebar():
    st.sidebar.markdown("""
        <style>
            /* Style for dark mode compatibility */
            .sidebar-content {
                padding: 1rem;
                font-family: 'Segoe UI', sans-serif;
            }
            .sidebar-content h3 {
                margin-bottom: 0.5rem;
                color: var(--text-color);
            }
            .sidebar-content p, .sidebar-content li {
                font-size: 0.9rem;
                color: var(--text-color);
            }
        </style>
    """, unsafe_allow_html=True)

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
    st.markdown("""
        <style>
            .main { background-color: #f4f6f8; }
            .stButton > button {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 8px;
            }
            .prediction-box {
                background-color: #ffffff;
                padding: 1rem;
                border-radius: 1rem;
                box-shadow: 0 0 12px rgba(0, 0, 0, 0.05);
                margin-top: 1rem;
            }
            .center-text { text-align: center; }
            h1, h2, h3 { color: #2c3e50; }
        </style>
    """, unsafe_allow_html=True)

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

        # Ensure a minimum visible width for very small values
        bar_width = f"width: {max(percent_width, 4)}%;" if percent_width > 0 else "width: 0%;"

        st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <strong>{cls}</strong>: {percent_text}
                <div style="height: 10px; background-color: #55555522; border-radius: 5px;">
                    <div style="{bar_width} height: 10px; background-color: #4caf50; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_gradcam(gradcam_image):
    st.markdown("#### 🌡️ Grad-CAM Heatmap")
    st.image(gradcam_image, width=400)

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import base64
import plotly.express as px

# Load class names before sidebar
try:
    with open("data/labels.txt") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    st.error(f"⚠️ Failed to load labels.txt: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PackInspect - Bottle Anomaly Detection",
    page_icon="📦",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3ABEFF;
        text-align: center;
    }
    .section-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .result-label {
        font-size: 1.2rem;
        margin-top: 10px;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1rem;
        color: #AAAAAA;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .sidebar-img {
        border-radius: 12px;
        border: 1px solid #444;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        margin-bottom: 10px;
        width: 100%;
    }
    img:hover {
        transform: scale(1.03);
        transition: transform 0.3s ease;
    }
    /* Remove background and borders from radio containers */
    div[data-baseweb="radio"] {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* Optional: Slight gap between buttons */
    .stRadio > div {
        gap: 0.75rem;
    }
    /* Add hover effect for radio labels */
    div[data-baseweb="radio"] label:hover {
        background-color: #333333 !important;
        transition: background-color 0.2s ease;
    }
    /* Add spacing between radio buttons vertically */
    div[data-baseweb="radio"] > div {
        gap: 1rem !important;
    }
    /* Use lighter border for deselected buttons */
    div[data-baseweb="radio"] label {
        border: 1px solid #555 !important;
        border-radius: 10px !important;
    }
    /* Make selectbox and slider labels bigger */
    label, .stSlider > label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #fff !important;
    }
    /* Target selectbox and slider labels in Streamlit */
    div[data-testid="stSelectboxLabel"] > label,
    div[data-testid="stSliderLabel"] > label {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #fff !important;
    }
    /* Stronger selector for all labels in main area */
    section.main label, div[data-testid="stSelectboxLabel"] > label, div[data-testid="stSliderLabel"] > label {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar image
with open("assets/overview_dataset.jpg", "rb") as img_file:
    sidebar_img_b64 = base64.b64encode(img_file.read()).decode()

# Sidebar
with st.sidebar:
    # Header and image
    st.markdown("""
        <div style="text-align:center; font-size:2rem; font-weight:bold; margin-bottom:0.5rem;">PackInspect</div>
        <div style="text-align:center;">
            <img src="data:image/png;base64,{}" class="sidebar-img" />
        </div>
        <div style='text-align:center; font-size:1.2rem; font-weight:bold; margin: 1rem 0 1.5rem 0;'>Anomaly Detection Tool</div>
    """.format(sidebar_img_b64), unsafe_allow_html=True)

    st.markdown(
        """
        <div style='font-size:1.3rem; font-weight:bold; color:#fff; margin-bottom:0.8rem; display:flex; align-items:center;'>
            📁 Navigation
        </div>
        """, unsafe_allow_html=True
    )

    selected = st.radio(
        "", ["Dashboard", "Defect Log"],
        key="nav",
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 1.2rem 0;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        [data-testid="stRadio"] > div {
            flex-direction: column;
            gap: 1rem;
        }
        [data-testid="stRadio"] label {
            background-color: #23232b;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            border: 1.5px solid #393947;
            color: white;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        [data-testid="stRadio"] label:hover {
            background-color: #333333 !important;
            transition: background-color 0.2s ease;
        }
        [data-testid="stRadio"] input:checked + div > label {
            background-color: #22c55e !important;
            color: black !important;
            font-weight: bold;
        }
        /* Remove background and borders from radio containers */
        div[data-baseweb="radio"] {
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        .stRadio > div {
            gap: 0.75rem;
        }
        div[data-baseweb="radio"] label {
            border: 1.5px solid #393947 !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.expander("How to Use", expanded=False):
        st.markdown("""
1. **Upload or capture an image**  
   Use file uploader or camera input.

2. **View prediction results**  
   The model will classify the image as *Good* or *Defect*.

3. **Check Defect Log**  
   See past predictions and download CSV report if needed.
        """)

    with st.expander("Model Info", expanded=False):
        st.markdown("**Model Classes:**")
        for cls in class_names:
            st.markdown(f"- {cls}")

    st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 1.2rem 0;'>", unsafe_allow_html=True)

    accuracy_value = 0.925
    st.markdown("**Model Accuracy:**")
    st.progress(accuracy_value)
    st.markdown(f"<small>Approx. Accuracy: <b>{accuracy_value*100:.2f}%</b></small>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; color:#888; font-size:0.8rem; margin-top:2rem;'><small>Made with ❤️ for smart manufacturing | © 2025</small></div>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/keras_model.keras", compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# Prediction function
def predict(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)[0][0]
    label = class_names[0] if prediction < 0.5 else class_names[1]
    confidence = 1 - prediction if label == class_names[0] else prediction
    return label, confidence

# Dashboard Page
if 'history' not in st.session_state:
    st.session_state.history = []

if selected == "Dashboard":
    st.markdown("<h1 class='main-header' style='margin-bottom: 2.5rem;'>PackInspect - Bottle Anomaly Detection System</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<span style='font-size:1rem; font-weight:700; color:#fff;'>Choose Input Method:</span>",
            unsafe_allow_html=True
        )
        input_method = st.selectbox(
            "",
            options=["Upload Image", "Camera Input"],
            key="input_method",
            label_visibility="collapsed"
        )

    with col2:
        if input_method == "Upload Image":
            st.markdown(
                "<span style='font-size:1rem; font-weight:700; color:#fff;'>Select Mode:</span>",
                unsafe_allow_html=True
            )
            mode = st.selectbox(
                "",
                options=["Single Image", "Batch Upload"],
                key="mode",
                label_visibility="collapsed"
            )
        else:
            mode = None

    st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:1rem; font-weight:700; color:#fff;'>Set confidence threshold</span>",
        unsafe_allow_html=True
    )
    threshold = st.slider(
        "",
        0.0, 1.0, 0.5, 0.01,
        key="threshold",
        label_visibility="collapsed"
    )

    image = None
    filename = ""

    if input_method == "Upload Image":
        if mode == "Single Image":
            st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 0.5rem 0 0.7rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                "<span style='font-size:1.12rem; font-weight:700; color:#fff;'>Upload an image</span>",
                unsafe_allow_html=True
            )
            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                filename = uploaded_file.name
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    label, confidence = predict(image)
                    glow_color = 'rgba(34, 197, 94, 0.4)' if label.lower() == 'good' else 'rgba(248, 113, 113, 0.4)'
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:20px; box-shadow: 0 0 40px {glow_color}; background-color: #000000; padding:0;" />
                        <div style="color:#bbb; font-size:0.85rem; margin-top:6px;">🖼️ Original: {filename}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    with st.spinner("Analyzing image..."):
                        label, confidence = predict(image)
                    if label.lower() == "good":
                        st.markdown(
                            f"""
                            <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #22c55e;'>
                                <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                    <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>✅</span>
                                    <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#22c55e;'>GOOD</span>
                                </div>
                                <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                                <div style='font-size:1.1rem;'><b>Status:</b> No defects detected</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"""
                            <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #f87171;'>
                                <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                    <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>⚠️</span>
                                    <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#f87171;'>DEFECT DETECTED</span>
                                </div>
                                <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                                <div style='font-size:1.1rem;'><b>Status:</b> Requires attention</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.session_state.history.insert(0, {
                        "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
                        "class": label,
                        "confidence": f"{confidence*100:.2f}%"
                    })
        else:  # Batch Upload
            st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 0.5rem 0 0.7rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                "<span style='font-size:1.12rem; font-weight:700; color:#fff;'>Upload image(s)</span>",
                unsafe_allow_html=True
            )
            uploaded_files = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if uploaded_files:
                for uploaded_file in reversed(uploaded_files):
                    image = Image.open(uploaded_file)
                    filename = uploaded_file.name
                    col1, col2 = st.columns([1, 1.2])
                    with col1:
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode()
                        label, confidence = predict(image)
                        glow_color = 'rgba(34, 197, 94, 0.4)' if label.lower() == 'good' else 'rgba(248, 113, 113, 0.4)'
                        st.markdown(f"""
                        <div style="text-align:center;">
                            <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:20px; box-shadow: 0 0 40px {glow_color}; background-color: #000000; padding:0;" />
                            <div style="color:#bbb; font-size:0.85rem; margin-top:6px;">🖼️ Original: {filename}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        with st.spinner("Analyzing image..."):
                            label, confidence = predict(image)
                        if label.lower() == "good":
                            st.markdown(
                                f"""
                                <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #22c55e;'>
                                    <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                        <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>✅</span>
                                        <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#22c55e;'>GOOD</span>
                                    </div>
                                    <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                                    <div style='font-size:1.1rem;'><b>Status:</b> No defects detected</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"""
                                <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #f87171;'>
                                    <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                        <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>⚠️</span>
                                        <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#f87171;'>DEFECT DETECTED</span>
                                    </div>
                                    <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                                    <div style='font-size:1.1rem;'><b>Status:</b> Requires attention</div>
                                </div>
                                """, unsafe_allow_html=True)
                        st.session_state.history.insert(0, {
                            "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
                            "class": label,
                            "confidence": f"{confidence*100:.2f}%"
                        })
    else:
        camera_file = st.camera_input("Capture Image")
        if camera_file:
            image = Image.open(camera_file)
            filename = getattr(camera_file, 'name', 'Captured Image')
            col1, col2 = st.columns([1, 1.2])
            with col1:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                label, confidence = predict(image)
                glow_color = 'rgba(34, 197, 94, 0.4)' if label.lower() == 'good' else 'rgba(248, 113, 113, 0.4)'
                st.markdown(f"""
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{img_b64}" style="width:320px; max-width:100%; border-radius:20px; box-shadow: 0 0 40px {glow_color}; background-color: #000000; padding:0;" />
                    <div style="color:#bbb; font-size:0.85rem; margin-top:6px;">🖼️ Original: {filename}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                with st.spinner("Analyzing image..."):
                    label, confidence = predict(image)
                if label.lower() == "good":
                    st.markdown(
                        f"""
                        <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #22c55e;'>
                            <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>✅</span>
                                <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#22c55e;'>GOOD</span>
                            </div>
                            <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                            <div style='font-size:1.1rem;'><b>Status:</b> No defects detected</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"""
                        <div style='background: rgba(0,0,0,0.0); border-radius: 20px; padding: 2.2rem 2.5rem 2rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.18); border: 2px solid #f87171;'>
                            <div style='display: flex; align-items: center; margin-bottom:0.5rem;'>
                                <span style='font-size:2.2rem; margin-right: 1.1rem; line-height: 1;'>⚠️</span>
                                <span style='font-size:2.2rem; font-weight:900; letter-spacing:1px; color:#f87171;'>DEFECT DETECTED</span>
                            </div>
                            <div style='font-size:1.1rem; margin-bottom:0.2rem;'><b>Confidence:</b> {confidence*100:.2f}%</div>
                            <div style='font-size:1.1rem;'><b>Status:</b> Requires attention</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.session_state.history.insert(0, {
                    "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
                    "class": label,
                    "confidence": f"{confidence*100:.2f}%"
                })

# Defect Log Page
elif selected == "Defect Log":
    st.title("📋 Recent Detection History")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history[:5])
        df.index = df.index + 1 
        st.dataframe(df) 
        full_df = pd.DataFrame(st.session_state.history)
        chart_data = full_df['class'].value_counts().reset_index()
        chart_data.columns = ['Label', 'Count']

        st.markdown("<hr style='border: 0; height: 1px; background: #444; margin: 1.2rem 0;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Pie Chart - Defect Distribution")
            pie_chart = px.pie(
                chart_data,
                names='Label',
                values='Count',
                title='Good vs Defect',
                color_discrete_map={
                    'Good': '#22c55e',
                    'Defect': '#f87171'
                }
            )
            st.plotly_chart(pie_chart, use_container_width=True)

        with col2:
            st.markdown("### 📊 Bar Chart - Defect Count")
            bar_chart = px.bar(
                chart_data,
                x='Label',
                y='Count',
                text='Count',
                title='Defect Count Overview',
                color='Label',
                color_discrete_map={
                    'Good': '#22c55e',
                    'Defect': '#f87171'
                }
            )
            st.plotly_chart(bar_chart, use_container_width=True)

        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        csv_path = os.path.join(logs_dir, "defect_log.csv")
        full_df.to_csv(csv_path, index=True) 

        csv = full_df.to_csv(index=True).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "defect_log.csv", "text/csv")
    else:
        st.info("No detection history yet.")

# Footer
st.markdown(f"<div class='footer'>PackInspect | Last updated: {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

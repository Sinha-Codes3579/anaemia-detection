import streamlit as st
import cv2
import numpy as np
import torch
import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from inference.pipeline import AnaemiaPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anaemia Detection from Nail Images",
    page_icon="🩺",
    layout="centered"
)

# ── Load pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    config = Config()
    cnn_path  = os.path.join(config.MODEL_DIR, 'anemia_model_efficientnet.pth')
    yolo_path = os.path.join(config.YOLO_DIR, 'nail_detector', 'weights', 'best.pt')

    if not os.path.exists(cnn_path):
        st.error(f"CNN model not found: {cnn_path}")
        return None
    if not os.path.exists(yolo_path):
        st.error(f"YOLO model not found: {yolo_path}")
        return None

    return AnaemiaPipeline(config, cnn_path, yolo_path)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🩺 Nail-based Anaemia Detection")
st.markdown("""
**Non-invasive anaemia screening from fingernail images using Deep Learning**

*YOLOv8 nail detection → EfficientNet-B0 classification*
""")

st.info(
    "📌 **Disclaimer:** This tool is for research purposes only "
    "and is not a substitute for clinical diagnosis."
)

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a fingernail or hand image",
    type=['jpg', 'jpeg', 'png'],
    help="Best results with clear, well-lit images of fingernails"
)

if uploaded_file is not None:
    # Show original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, use_column_width=True)

    # Save temp file for pipeline
    temp_path = '/tmp/uploaded_nail.jpg'
    image_np  = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, image_bgr)

    # Run pipeline
    with st.spinner("🔍 Detecting nails and analysing..."):
        pipeline = load_pipeline()
        if pipeline is not None:
            result = pipeline.predict(temp_path)

    if pipeline is not None and 'error' not in result:
        # Show annotated image
        with col2:
            st.subheader("🔎 Detection Result")
            annotated_rgb = cv2.cvtColor(
                result['annotated_image'], cv2.COLOR_BGR2RGB
            )
            st.image(annotated_rgb, use_column_width=True)

        st.divider()

        # ── Main result ────────────────────────────────────────────────────
        pred  = result['final_prediction']
        conf  = result['confidence']
        color = "🔴" if pred == 'anaemic' else "🟢"

        if pred == 'anaemic':
            st.error(f"## {color} Result: ANAEMIC  ({conf:.1%} confidence)")
            st.warning(
                "⚠️ Possible signs of anaemia detected in nail coloration. "
                "Please consult a healthcare professional for proper diagnosis."
            )
        else:
            st.success(f"## {color} Result: NON-ANAEMIC  ({conf:.1%} confidence)")
            st.info("✅ No signs of anaemia detected in nail coloration.")

        st.divider()

        # ── Nail-level breakdown ───────────────────────────────────────────
        st.subheader(f"🔬 Nail Analysis  ({result['nail_count']} nail(s) detected)")

        if result['fallback_used']:
            st.warning("⚠️ No nail detected — full image used as fallback.")

        for r in result['nail_results']:
            with st.expander(f"Nail {r['nail_id']} — {r['prediction'].upper()}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Prediction",  r['prediction'].capitalize())
                    st.metric("CNN Confidence", f"{r['cnn_confidence']:.1%}")
                with c2:
                    st.metric("YOLO Detection Conf", f"{r['yolo_confidence']:.1%}")
                    st.metric("Bounding Box",
                              f"{r['bbox'][0]},{r['bbox'][1]} → "
                              f"{r['bbox'][2]},{r['bbox'][3]}")

                # Probability bar chart
                st.write("**Class probabilities:**")
                probs = r['probabilities']
                st.progress(probs.get('anaemic', 0),
                            text=f"Anaemic: {probs.get('anaemic', 0):.1%}")
                st.progress(probs.get('non_anaemic', 0),
                            text=f"Non-anaemic: {probs.get('non_anaemic', 0):.1%}")

        st.divider()

        # ── Model info ─────────────────────────────────────────────────────
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            | Component | Details |
            |---|---|
            | Nail Detector | YOLOv8n (mAP50: 98.1%, Recall: 100%) |
            | Classifier | EfficientNet-B0 (Val Accuracy: 73.75%) |
            | Dataset | 484 patients, 4260 images, patient-stratified split |
            | Training | Patient-level split — zero data leakage |
            | Reference | Comparable to DenseNet169: 69.83% (Frontiers, 2025) |
            """)

    elif 'error' in result:
        st.error(f"Pipeline error: {result['error']}")

else:
    # Show sample instructions when no image uploaded
    st.markdown("""
    ### How to use:
    1. Upload a clear photo of fingernails or a hand
    2. The system will automatically detect nail regions
    3. Each nail is analysed for signs of anaemia
    4. A final prediction is made by majority vote
""")
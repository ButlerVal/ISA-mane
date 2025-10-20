import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import warnings

warnings.filterwarnings("ignore")

# Configure TensorFlow for CPU
tf.config.set_visible_devices([], 'GPU')

# Set page config first
st.set_page_config(
    page_title="Breast Cancer Detection", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# Constants
IMG_SIZE = (96, 96)

# Cache model loading to prevent reloading on every interaction
@st.cache_resource
def load_models():
    models = {}
    
    with st.spinner("Loading CNN Model..."):
        try:
            models['cnn'] = load_model("cnn_model.h5", compile=False)
            st.sidebar.success("✅ CNN Model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ CNN Model error: {str(e)[:100]}")
            models['cnn'] = None
    
    with st.spinner("Loading EfficientNet Model..."):
        try:
            models['efficientnet'] = load_model("efficientnet_model.h5", compile=False)
            st.sidebar.success("✅ EfficientNet Model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ EfficientNet Model error: {str(e)[:100]}")
            models['efficientnet'] = None
    
    return models

# Basic histopathology feature detection
def is_histopathology_image(img):
    try:
        # Convert to HSV for better feature analysis
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

        # Check for tissue-like pink/purple range
        lowerb = np.array([120, 30, 50], dtype=np.uint8)
        upperb = np.array([170, 255, 255], dtype=np.uint8)
        pink_mask = cv2.inRange(hsv, lowerb, upperb)
        pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])

        return pink_ratio > 0.02
    except Exception as e:
        st.warning(f"Image validation skipped: {e}")
        return True  # Allow processing if validation fails

# Image Preprocessing for RGB
def preprocess_image_rgb(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Image Preprocessing for Grayscale
def preprocess_image_grayscale(image):
    image = image.resize(IMG_SIZE)
    image = image.convert('L')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Main UI
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")

# Load models with caching
models = load_models()

# Check if any model loaded
if models['cnn'] is None and models['efficientnet'] is None:
    st.error("⚠️ No models could be loaded. Please check your model files.")
    st.info("Make sure `cnn_model.h5` and `efficientnet_model.h5` are in the same directory as app.py")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

        # Check image type
        if not is_histopathology_image(image):
            st.warning("🚫 This doesn't look like a typical histopathology image. Proceeding anyway...")
        
        # Make predictions
        predictions = []
        model_names = []
        
        with st.spinner("Analyzing image..."):
            # Try CNN Model with RGB
            if models['cnn'] is not None:
                try:
                    processed_img = preprocess_image_rgb(image)
                    cnn_prob = models['cnn'].predict(processed_img, verbose=0)[0][0]
                    predictions.append(cnn_prob)
                    model_names.append("CNN")
                    st.write(f"🔸 CNN Prediction: {cnn_prob:.4f}")
                except Exception as e:
                    st.warning(f"⚠️ CNN model failed: {str(e)[:200]}")
            
            # Try EfficientNet Model
            if models['efficientnet'] is not None:
                try:
                    # Try grayscale first
                    processed_img = preprocess_image_grayscale(image)
                    eff_prob = models['efficientnet'].predict(processed_img, verbose=0)[0][0]
                    predictions.append(eff_prob)
                    model_names.append("EfficientNet")
                    st.write(f"🔸 EfficientNet Prediction: {eff_prob:.4f}")
                except Exception as e:
                    st.warning(f"⚠️ EfficientNet (grayscale) failed: {str(e)[:100]}")
                    # Try RGB as fallback
                    try:
                        processed_img = preprocess_image_rgb(image)
                        eff_prob = models['efficientnet'].predict(processed_img, verbose=0)[0][0]
                        predictions.append(eff_prob)
                        model_names.append("EfficientNet (RGB)")
                        st.write(f"🔸 EfficientNet Prediction (RGB): {eff_prob:.4f}")
                    except Exception as e2:
                        st.error(f"❌ EfficientNet failed completely: {str(e2)[:200]}")

        # Calculate ensemble if we have any predictions
        if len(predictions) > 0:
            ensemble_prob = np.mean(predictions)
            label = "Malignant" if ensemble_prob >= 0.5 else "Benign"
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", label)
            
            with col2:
                st.metric("Confidence", f"{ensemble_prob*100:.2f}%")
            
            st.info(f"📊 Models used: {', '.join(model_names)}")
            
            # Add interpretation
            if ensemble_prob >= 0.5:
                st.warning("⚠️ This result suggests malignancy. Please consult with medical professionals.")
            else:
                st.success("✅ This result suggests benign tissue. Always verify with medical professionals.")
        else:
            st.error("❌ All models failed to make predictions. Please try a different image.")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.exception(e)

# Add footer
st.markdown("---")
st.caption("Developed with ❤️ for Breast Cancer Detection")
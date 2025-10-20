import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import cv2
import os
import warnings

warnings.filterwarnings("ignore")


# Load models with compile=False to avoid weight loading issues
try:
    cnn_model = load_model("cnn_model.h5", compile=False)
    st.sidebar.success("✅ CNN Model loaded")
except Exception as e:
    st.sidebar.error(f"❌ CNN Model error: {e}")
    cnn_model = None

try:
    efficientnet_model = load_model("efficientnet_model.h5", compile=False)
    st.sidebar.success("✅ EfficientNet Model loaded")
except Exception as e:
    st.sidebar.error(f"❌ EfficientNet Model error: {e}")
    efficientnet_model = None

# Constants
IMG_SIZE = (96, 96)

# Basic histopathology feature detection (color, texture, etc.)
def is_histopathology_image(img):
    # Convert to HSV for better feature analysis
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

    # Check for tissue-like pink/purple range
    lowerb = np.array([120, 30, 50], dtype=np.uint8)
    upperb = np.array([170, 255, 255], dtype=np.uint8)
    pink_mask = cv2.inRange(hsv, lowerb, upperb)
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])

    return pink_ratio > 0.02  # Adjustable threshold

# Image Preprocessing for RGB (original)
def preprocess_image_rgb(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Image Preprocessing for Grayscale
def preprocess_image_grayscale(image):
    # Resize the image
    image = image.resize(IMG_SIZE)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    
    # Add channel dimension (height, width, 1)
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension (1, height, width, 1)
    image = np.expand_dims(image, axis=0)
    
    return image

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Detection (Ensemble)", layout="centered", initial_sidebar_state="collapsed")
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")

# Show model status
if cnn_model is None and efficientnet_model is None:
    st.error("⚠️ No models could be loaded. Please check your model files.")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    # Check image type
    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology image. Please upload a valid sample.")
    else:
        predictions = []
        model_names = []
        
        # Try CNN Model with RGB
        if cnn_model is not None:
            try:
                processed_img = preprocess_image_rgb(image)
                cnn_prob = cnn_model.predict(processed_img, verbose=0)[0][0]
                predictions.append(cnn_prob)
                model_names.append("CNN")
                st.write(f"CNN Prediction: {cnn_prob:.4f}")
            except Exception as e:
                st.warning(f"CNN model failed: {e}")
        
        # Try EfficientNet Model with Grayscale
        if efficientnet_model is not None:
            try:
                processed_img = preprocess_image_grayscale(image)
                st.write(f"Processed shape for EfficientNet: {processed_img.shape}")
                eff_prob = efficientnet_model.predict(processed_img, verbose=0)[0][0]
                predictions.append(eff_prob)
                model_names.append("EfficientNet")
                st.write(f"EfficientNet Prediction: {eff_prob:.4f}")
            except Exception as e:
                st.warning(f"EfficientNet model failed with grayscale. Error: {e}")
                # Try with RGB as fallback
                try:
                    processed_img = preprocess_image_rgb(image)
                    eff_prob = efficientnet_model.predict(processed_img, verbose=0)[0][0]
                    predictions.append(eff_prob)
                    model_names.append("EfficientNet")
                    st.write(f"EfficientNet Prediction (RGB): {eff_prob:.4f}")
                except Exception as e2:
                    st.error(f"EfficientNet failed completely: {e2}")

        # Calculate ensemble if we have any predictions
        if len(predictions) > 0:
            ensemble_prob = np.mean(predictions)
            label = "Malignant" if ensemble_prob >= 0.5 else "Benign"
            
            # Results
            st.success(f"✅ Prediction: **{label}**")
            st.info(f"**Confidence:** {ensemble_prob*100:.2f}%")
            st.write(f"Models used: {', '.join(model_names)}")
        else:
            st.error("❌ All models failed to make predictions.")

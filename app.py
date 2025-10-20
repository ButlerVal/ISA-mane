import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import cv2
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
# Set page configuration. This should be the first Streamlit command.
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Constants ---
IMG_SIZE = (96, 96)

# --- Model Loading ---
# Use Streamlit's caching to load the models once and store them in memory.
# This prevents reloading the models on every user interaction, which is slow.

@st.cache_resource
def load_cnn_model():
    """Load the custom CNN model from disk."""
    try:
        model = load_model("cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

@st.cache_resource
def load_efficientnet_model():
    """Load the EfficientNet model from disk."""
    try:
        model = load_model("efficientnet_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading EfficientNet model: {e}")
        return None

# Load the models using the cached functions
cnn_model = load_cnn_model()
efficientnet_model = load_efficientnet_model()


# --- Image Processing Functions ---

def is_histopathology_image(img):
    """
    A simple heuristic to check if an uploaded image is likely a histopathology slide.
    It checks for the prevalence of pink and purple colors common in H&E stains.
    """
    # Convert PIL Image to an array that OpenCV can use
    img_array = np.array(img)
    # Convert from RGB to HSV color space for better color analysis
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define the range for pink/purple colors in HSV
    # These values might need tweaking for different staining processes
    lower_bound = (120, 30, 50)
    upper_bound = (170, 255, 255)
    
    # Create a mask that only includes pixels within the pink/purple range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Calculate the ratio of pink/purple pixels to the total number of pixels
    pink_ratio = np.sum(mask > 0) / (img.size[0] * img.size[1])

    # If more than 2% of the pixels are pink/purple, assume it's a valid image
    return pink_ratio > 0.02

def preprocess_for_cnn(image):
    """Preprocesses the image for the custom CNN model."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0) # Add batch dimension

def preprocess_for_efficientnet(image):
    """
    Preprocesses the image for the EfficientNet model.
    EfficientNet requires a specific input format.
    """
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = eff_preprocess(image_array) # Use the specific EfficientNet preprocessor
    return np.expand_dims(image_array, axis=0) # Add batch dimension


# --- Streamlit User Interface ---

st.title("🧬 Histopathology Breast Cancer Classifier")
st.markdown("Upload a histopathology image to classify it as **Benign** or **Malignant** using an ensemble of two deep learning models.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and cnn_model and efficientnet_model:
    # Open and display the image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a prediction button
    if st.button("Classify Image", use_container_width=True):
        with st.spinner("Analyzing image..."):
            
            # First, check if the image appears to be a valid histopathology slide
            if not is_histopathology_image(image):
                st.error("🚫 This doesn't look like a histopathology image. Please upload a valid H&E stained tissue sample.")
            else:
                # Preprocess the image for each model
                cnn_img = preprocess_for_cnn(image)
                eff_img = preprocess_for_efficientnet(image)

                # Get predictions from both models
                cnn_prob = cnn_model.predict(cnn_img)[0][0]
                eff_prob = efficientnet_model.predict(eff_img)[0][0]
                
                # Simple averaging for the ensemble prediction
                ensemble_prob = (cnn_prob + eff_prob) / 2

                # Determine the final class based on the ensemble probability
                is_malignant = ensemble_prob > 0.5
                final_class = "Malignant" if is_malignant else "Benign"
                confidence = ensemble_prob if is_malignant else 1 - ensemble_prob

                # Display the result
                st.subheader("Classification Result")
                if is_malignant:
                    st.error(f"**Result: {final_class}**")
                else:
                    st.success(f"**Result: {final_class}**")
                
                st.metric(label="Model Confidence", value=f"{confidence:.2%}")
                
                with st.expander("🔬 View Model Details"):
                    st.markdown("The final prediction is an average of the outputs from two different models:")
                    st.progress(ensemble_prob)
                    st.markdown(f"- **Custom CNN Prediction:** `{cnn_prob:.4f}`")
                    st.markdown(f"- **EfficientNet Prediction:** `{eff_prob:.4f}`")
                    st.markdown(f"- **Ensemble Average:** `{ensemble_prob:.4f}`")
                    st.info("A value closer to 1.0 indicates a higher probability of being Malignant.", icon="ℹ️")

elif uploaded_file:
    st.error("Models could not be loaded. Please check the logs.")

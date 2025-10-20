import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import cv2
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")

# --- Page Configuration ---
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Breast Cancer Detection App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
IMG_SIZE = (96, 96)
MODEL_CONFIDENCE_THRESHOLD = 0.5
VALIDATION_RESIZE = (512, 512)  # Added for optimizing validation on large images

# --- Model Loading ---
# Using st.cache_resource to load models only once. Streamlit caches the result,
# so this function is only slow on the very first run.
@st.cache_resource
def load_models():
    """Loads and returns the pre-trained CNN and EfficientNet models as TFLite interpreters for optimized inference."""
    try:
        cnn_interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
        cnn_interpreter.allocate_tensors()
        
        efficientnet_interpreter = tf.lite.Interpreter(model_path="efficientnet_model.tflite")
        efficientnet_interpreter.allocate_tensors()
        
        return cnn_interpreter, efficientnet_interpreter
    except Exception as e:
        # Display an error message if models can't be loaded
        st.error(f"Error loading models: {e}", icon="🚨")
        return None, None

# --- Image Processing and Validation ---
def is_histopathology_image(image):
    """
    Analyzes the uploaded image to determine if it's likely a histopathology slide.
    This function checks for the distinct pink and purple colors of H&E staining.
    """
    # Resize for faster processing, especially on large histopathology images
    image = image.resize(VALIDATION_RESIZE)
    
    # Convert the image to a format OpenCV can use
    img_array = np.array(image.convert('RGB'))
    # Switch to HSV color space for more effective color detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define the HSV color range for pinks and purples common in H&E stains
    lower_bound = np.array([120, 40, 100])
    upper_bound = np.array([170, 255, 255])
    
    # Create a mask that isolates the target colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Calculate the percentage of the image that contains these colors
    color_ratio = np.sum(mask > 0) / (image.size[0] * image.size[1])

    # If over 5% of the image is pink/purple, we'll consider it valid.
    return color_ratio > 0.05

def preprocess_for_cnn(image):
    """Prepares an image for the custom CNN model."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize and ensure float32 for TFLite
    return np.expand_dims(image_array, axis=0)

def preprocess_for_efficientnet(image):
    """Prepares an image for the EfficientNetB0 model using its specific requirements."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    # Use the dedicated preprocessing function for EfficientNet
    preprocessed_array = eff_preprocess(image_array)
    return np.expand_dims(preprocessed_array, axis=0).astype(np.float32)  # Ensure float32 for TFLite

# --- Main Application UI ---

# Sidebar with project information from your PDF
# This UI part loads instantly, before any models are loaded.
with st.sidebar:
    st.title("About This Project")
    st.image("https://placehold.co/300x150/f0f2f6/333333?text=Histopathology+Image&font=inter", use_container_width=True)
    st.markdown("""
        This app is a demonstration of the MSc thesis project: **"An Ensemble Deep Learning Approach for Detecting Breast Cancer from Histopathology Images."**
    """)
    st.info("By **Isa**", icon="🧑‍💻")
    st.markdown("""
        ### Why This Project?
        Early detection of breast cancer can save lives. This tool uses AI to assist pathologists by analyzing histopathology images to distinguish between **Benign** (non-harmful) and **Malignant** (dangerous) tissue, aiming to make diagnosis faster and more accurate.
    """)
    st.markdown("""
        ### How It Works
        1.  **Upload a histopathology image.**
        2.  The app validates if it's a real medical slide.
        3.  Two AI models (**Custom CNN** & **EfficientNetB0**) analyze it.
        4.  An **ensemble prediction** is made by averaging their results for higher accuracy.
    """)

# Main page content
st.title("🔬 Breast Cancer Detection from Histopathology Images")
st.markdown("Upload a breast tissue slide image to classify it as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "tif"],
    help="Upload a histopathology image of breast tissue."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Error opening image: {e}. Please upload a valid image file.", icon="🚨")
    else:
        # Create two columns for image and analysis
        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            # The user sees the image and then clicks the button to start analysis
            if st.button("Analyze Image", use_container_width=True):
                # Models are loaded ONLY when the button is clicked
                with st.spinner("Loading models & analyzing image... This may take a moment on the first run."):
                    
                    # Load the models using the cached function
                    cnn_model, efficientnet_model = load_models()
                    
                    # Proceed only if models were loaded successfully
                    if cnn_model and efficientnet_model:
                        # --- Feature 1: Validate Image Type ---
                        if not is_histopathology_image(image):
                            st.error("⚠️ This does not appear to be a histopathology image.", icon="🚫")
                            st.markdown("The system detected that the uploaded file lacks the color and texture features of a typical H&E-stained medical slide. **Please upload a valid image to proceed.**")
                        else:
                            st.success("✅ Valid histopathology image detected.", icon="👍")
                            
                            # --- Feature 2: Run Predictions ---
                            # Preprocess image for each model
                            cnn_img = preprocess_for_cnn(image)
                            eff_img = preprocess_for_efficientnet(image)

                            try:
                                # Get predictions for CNN
                                cnn_input_details = cnn_model.get_input_details()
                                cnn_output_details = cnn_model.get_output_details()
                                cnn_model.set_tensor(cnn_input_details[0]['index'], cnn_img)
                                cnn_model.invoke()
                                cnn_prob = cnn_model.get_tensor(cnn_output_details[0]['index'])[0][0]
                                
                                # Get predictions for EfficientNet
                                eff_input_details = efficientnet_model.get_input_details()
                                eff_output_details = efficientnet_model.get_output_details()
                                efficientnet_model.set_tensor(eff_input_details[0]['index'], eff_img)
                                efficientnet_model.invoke()
                                eff_prob = efficientnet_model.get_tensor(eff_output_details[0]['index'])[0][0]
                                
                                # Ensemble by averaging
                                ensemble_prob = (cnn_prob + eff_prob) / 2

                                # Determine final classification and confidence
                                is_malignant = ensemble_prob > MODEL_CONFIDENCE_THRESHOLD
                                final_class = "Malignant" if is_malignant else "Benign"
                                confidence = ensemble_prob if is_malignant else 1 - ensemble_prob

                                # --- Feature 3: Display Results ---
                                st.markdown("---")
                                st.subheader("Analysis Complete: Ensemble Result")

                                if is_malignant:
                                    st.error(f"**Classification: {final_class}**", icon="❗")
                                else:
                                    st.success(f"**Classification: {final_class}**", icon="✅")

                                st.metric(label="Model Confidence", value=f"{confidence:.2%}")
                                st.progress(confidence)

                                with st.expander("🔬 View Individual Model Predictions"):
                                    st.markdown("The final result is an average of the outputs from two distinct AI models to improve reliability.")
                                    st.markdown(f"- **Custom CNN Model Prediction:** `{cnn_prob:.4f}`")
                                    st.markdown(f"- **EfficientNetB0 Model Prediction:** `{eff_prob:.4f}`")
                                    st.info("Values closer to `1.0` indicate a higher probability of being Malignant.", icon="ℹ️")
                            except Exception as e:
                                st.error(f"Error during model prediction: {e}", icon="🚨")
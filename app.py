import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    # Disable GPU if causing issues
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow import keras
    from tensorflow.keras.models import load_model
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    st.stop()

# Constants
IMG_SIZE = (96, 96)
MODEL_PATHS = {
    'cnn': 'cnn_model.h5',
    'efficientnet': 'efficientnet_model.h5'
}

# Load models with error handling
@st.cache_resource
def load_models():
    """Load both models with comprehensive error handling"""
    models = {}
    errors = []
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            if not os.path.exists(model_path):
                errors.append(f"❌ {model_name} model file not found: {model_path}")
                continue
            
            # Load model with compile=False to avoid optimizer issues
            model = load_model(model_path, compile=False)
            
            # Recompile with current TensorFlow version
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            models[model_name] = model
            
        except Exception as e:
            errors.append(f"❌ Error loading {model_name}: {str(e)}")
    
    return models, errors

# Basic histopathology feature detection
def is_histopathology_image(img):
    """
    Check if image looks like histopathology slide
    Returns True if image has tissue-like characteristics
    """
    try:
        # Convert to numpy array if PIL Image
        img_array = np.array(img)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Check for tissue-like pink/purple range (H&E staining)
        pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
        pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
        
        # Also check for purple/blue nuclei
        purple_mask = cv2.inRange(hsv, (100, 30, 50), (140, 255, 255))
        purple_ratio = np.sum(purple_mask > 0) / (img.size[0] * img.size[1])
        
        # Combined check
        return (pink_ratio > 0.02) or (purple_ratio > 0.05)
    
    except Exception as e:
        st.warning(f"Could not validate image type: {e}")
        return True  # Assume valid if check fails

# Image Preprocessing
def preprocess_image(image):
    """
    Resize and normalize image for model input
    """
    # Resize to target size
    image = image.resize(IMG_SIZE)
    
    # Convert to array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Prediction function
def make_prediction(models, processed_img):
    """
    Run ensemble prediction using both models
    """
    predictions = {}
    
    try:
        if 'cnn' in models:
            cnn_prob = models['cnn'].predict(processed_img, verbose=0)[0][0]
            predictions['cnn'] = float(cnn_prob)
        
        if 'efficientnet' in models:
            eff_prob = models['efficientnet'].predict(processed_img, verbose=0)[0][0]
            predictions['efficientnet'] = float(eff_prob)
        
        # Calculate ensemble prediction
        if predictions:
            ensemble_prob = np.mean(list(predictions.values()))
            predictions['ensemble'] = float(ensemble_prob)
        
        return predictions
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Breast Cancer Detection",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Title and description
    st.title("🧬 Histopathology Breast Cancer Classifier")
    st.markdown("""
    This application uses an ensemble of deep learning models (CNN + EfficientNet) 
    to classify breast histopathology images as **Benign** or **Malignant**.
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        models, errors = load_models()
    
    # Display any loading errors
    if errors:
        st.error("**Model Loading Issues:**")
        for error in errors:
            st.error(error)
        
        if not models:
            st.stop()
        else:
            st.warning(f"Running with {len(models)} model(s) only.")
    else:
        st.success(f"✅ Successfully loaded {len(models)} model(s)")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Histopathology Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a breast histopathology image (H&E stained)"
    )
    
    if uploaded_file:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="🔬 Uploaded Image", use_container_width=True)
            
            # Check if image looks like histopathology
            if not is_histopathology_image(image):
                st.warning("""
                ⚠️ This doesn't appear to be a histopathology image. 
                The model is trained on H&E stained breast tissue samples.
                Results may not be reliable.
                """)
            
            # Preprocess image
            with st.spinner("Processing image..."):
                processed_img = preprocess_image(image)
            
            # Make predictions
            with st.spinner("Running predictions..."):
                predictions = make_prediction(models, processed_img)
            
            if predictions and 'ensemble' in predictions:
                ensemble_prob = predictions['ensemble']
                label = "Malignant (IDC+)" if ensemble_prob >= 0.5 else "Benign (IDC-)"
                confidence = ensemble_prob if ensemble_prob >= 0.5 else (1 - ensemble_prob)
                
                # Display results
                with col2:
                    st.markdown("### 📊 Results")
                    
                    # Main prediction
                    if "Malignant" in label:
                        st.error(f"**Prediction:** {label}")
                    else:
                        st.success(f"**Prediction:** {label}")
                    
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Individual model predictions
                    with st.expander("🔍 Model Details"):
                        if 'cnn' in predictions:
                            st.write(f"**CNN Model:** {predictions['cnn']*100:.2f}% malignant")
                        if 'efficientnet' in predictions:
                            st.write(f"**EfficientNet:** {predictions['efficientnet']*100:.2f}% malignant")
                        st.write(f"**Ensemble:** {predictions['ensemble']*100:.2f}% malignant")
                
                # Disclaimer
                st.markdown("---")
                st.info("""
                ⚠️ **Medical Disclaimer:** This tool is for research and educational purposes only. 
                It should NOT be used for clinical diagnosis. Always consult qualified healthcare 
                professionals for medical advice.
                """)
            else:
                st.error("Failed to generate predictions. Please check the models and try again.")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

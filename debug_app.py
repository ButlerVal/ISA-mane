import streamlit as st
import sys
import os

st.title("Debug App - Check what's working")

# Check Python version
st.write(f"Python version: {sys.version}")

# Check if files exist
st.write("### File Check:")
st.write(f"cnn_model.h5 exists: {os.path.exists('cnn_model.h5')}")
st.write(f"efficientnet_model.h5 exists: {os.path.exists('efficientnet_model.h5')}")

# Try importing libraries
try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow {tf.__version__} imported")
except Exception as e:
    st.error(f"❌ TensorFlow import failed: {e}")

try:
    import cv2
    st.success(f"✅ OpenCV {cv2.__version__} imported")
except Exception as e:
    st.error(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    st.success(f"✅ NumPy {np.__version__} imported")
except Exception as e:
    st.error(f"❌ NumPy import failed: {e}")

try:
    from PIL import Image
    st.success("✅ Pillow imported")
except Exception as e:
    st.error(f"❌ Pillow import failed: {e}")

# Try loading models
st.write("### Model Loading Test:")

# Prefer tensorflow.keras's load_model when tensorflow is available, fall back to keras
try:
    try:
        from keras.models import load_model
    except Exception:
        from keras.models import load_model
except Exception as e:
    st.error(f"❌ Cannot import load_model from keras/tensorflow.keras: {e}")
    load_model = None

def _safe_shape(model, attr_name):
    """Return attribute attr_name or try to fall back to the first layer's attribute."""
    if model is None:
        return None
    # direct attribute if present
    val = getattr(model, attr_name, None)
    if val is not None:
        return val
    # fall back to layers (Sequential/Functional models)
    layers = getattr(model, "layers", None)
    if layers and len(layers) > 0:
        first = layers[0]
        return getattr(first, attr_name, None)
    return None

# Load and inspect CNN model
try:
    if load_model is None:
        raise RuntimeError("load_model is not available")

    st.write("Attempting to load CNN model...")
    cnn = load_model("cnn_model.h5", compile=False)
    st.success("✅ CNN model loaded successfully")

    in_shape = _safe_shape(cnn, "input_shape")
    out_shape = _safe_shape(cnn, "output_shape")

    if in_shape is None:
        st.warning("CNN model input shape not available")
    else:
        st.write(f"Input shape: {in_shape}")

    if out_shape is None:
        st.warning("CNN model output shape not available")
    else:
        st.write(f"Output shape: {out_shape}")

except Exception as e:
    st.error(f"❌ CNN model loading failed: {e}")

# Load and inspect EfficientNet model
try:
    if load_model is None:
        raise RuntimeError("load_model is not available")

    st.write("Attempting to load EfficientNet model...")
    eff = load_model("efficientnet_model.h5", compile=False)
    st.success("✅ EfficientNet model loaded successfully")

    in_shape = _safe_shape(eff, "input_shape")
    out_shape = _safe_shape(eff, "output_shape")

    if in_shape is None:
        st.warning("EfficientNet model input shape not available")
    else:
        st.write(f"Input shape: {in_shape}")

    if out_shape is None:
        st.warning("EfficientNet model output shape not available")
    else:
        st.write(f"Output shape: {out_shape}")

except Exception as e:
    st.error(f"❌ EfficientNet model loading failed: {e}")

st.write("### Debug Complete")
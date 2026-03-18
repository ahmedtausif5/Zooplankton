import os
import torch
import gdown
import streamlit as st
from PIL import Image

# Import from our modular files
from model import HierarchicalZooplanktonViT
from inference import infer_single_image

# --- CONFIGURATION ---
st.set_page_config(page_title="Zooplankton Classifier", page_icon="🔬", layout="centered")

# Paste your Google Drive ID here
DRIVE_FILE_ID = ''1NlvYztsQ9156BEuSBkl1uGMDTKe-EJc-''
MODEL_SAVE_PATH = 'models/best_zooplankton_model_v2.pth'

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Downloads (if needed) and loads the model into cache to prevent reloading."""
    # Ensure directory exists
    os.makedirs('models', exist_ok=True)
    
    # Fetch from Google Drive if not present locally
    if not os.path.exists(MODEL_SAVE_PATH):
        with st.spinner("Downloading model weights from Google Drive (~330MB)... Please wait."):
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            gdown.download(url, MODEL_SAVE_PATH, quiet=False)
            
    # Figure out the best hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize the blank architecture
    model = HierarchicalZooplanktonViT(pretrained=False, freeze_backbone=True)
    
    # Load the learned weights
    saved_weights = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(saved_weights)
    
    # Finalize setup
    model.to(device)
    model.eval()
    
    return model, device

# --- UI LAYOUT ---
st.title("🔬 Hierarchical Zooplankton Classifier")
st.markdown("Upload a `.tif`, `.png`, or `.jpg` image of a zooplankton specimen (or debris) to run it through the multi-head Vision Transformer.")

# Load the model silently in the background
model, device = load_model()

# Image Uploader Widget
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Specimen', use_container_width=True)
    
    # Prediction trigger
    if st.button('Classify Specimen', type='primary', use_container_width=True):
        with st.spinner('Running strict top-down inference...'):
            # Pass the PIL image directly into our inference script
            results = infer_single_image(image, model, device)
            
        st.success("Classification Complete!")
        
        # Display results in 3 neat columns
        st.subheader("Hierarchical Predictions")
        col1, col2, col3 = st.columns(3)
        
        # Format the confidence scores as percentages
        col1.metric("Level 0 (Bio vs Non-Bio)", results['Level 0']['Class'], f"{results['Level 0']['Confidence']:.2%}")
        col2.metric("Level 1 (Order)", results['Level 1']['Class'], f"{results['Level 1']['Confidence']:.2%}")
        col3.metric("Level 2 (Family)", results['Level 2']['Class'], f"{results['Level 2']['Confidence']:.2%}")
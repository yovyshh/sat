import streamlit as st
import torch
from torchvision import models
from PIL import Image
from train_classifier import test_transform, NUM_CLASSES, class_names

st.set_page_config(page_title="Satellite Deforestation Detector", layout="wide")
st.title("🛰️ Satellite Image Classifier & Deforestation Detector")

# Load the trained model weights
@st.cache_resource
def load_model():
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, NUM_CLASSES)
    model.load_state_dict(torch.load("local_resnet50_eurosat.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# UI Layout: Side-by-side file uploaders
col1, col2 = st.columns(2)
with col1:
    file_year1 = st.file_uploader("Upload Before Image (Full Green Canopy)", type=["png", "jpg", "jpeg"])
with col2:
    file_year2 = st.file_uploader("Upload After Image (With Deforested Brown Clearcut)", type=["png", "jpg", "jpeg"])

if file_year1 and file_year2:
    img1 = Image.open(file_year1).convert('RGB')
    img2 = Image.open(file_year2).convert('RGB')
    
    # Force both images to have identical dimensions to fix alignment issues
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    with col1:
        st.image(img1, caption="Before Scene", use_container_width=True)
    with col2:
        st.image(img2, caption="After Scene", use_container_width=True)
        
    if st.button("🧠 Run Deforestation Detection Engine"):
        with st.spinner("Analyzing satellite patches..."):
            width, height = img1.size
            patch_size = 64
            forest_idx = class_names.index('Forest')
            changes_detected = 0
            
            # Diagnostic lists to see what the model actually predicts
            before_predictions = []
            after_predictions = []
            
            # Clean and correct loop to evaluate the uploaded images
            for i in range(0, width - patch_size + 1, patch_size):
                for j in range(0, height - patch_size + 1, patch_size):
                    p1 = img1.crop((i, j, i + patch_size, j + patch_size))
                    p2 = img2.crop((i, j, i + patch_size, j + patch_size))
                    
                    # Transform raw images to tensor batches correctly 
                    t1 = test_transform(p1).unsqueeze(0)
                    t2 = test_transform(p2).unsqueeze(0)
                    
                    with torch.no_grad():
                        pred1 = torch.argmax(model(t1), 1).item()
                        pred2 = torch.argmax(model(t2), 1).item()
                    
                    before_predictions.append(class_names[pred1])
                    after_predictions.append(class_names[pred2])
                    
                    # Deforestation detection logic filter [cite: 276]
                    if pred1 == forest_idx and pred2 != forest_idx:
                        changes_detected += 1
            
            # --- UI REPORTING ---
            st.metric(label="Total Deforestation Patches Detected", value=changes_detected)
            
            # Diagnostic help section
            with st.expander("🔍 Debug Diagnostic Mode: See what the model saw"):
                st.write(f"**Sample Before Predictions:** {set(before_predictions)}")
                st.write(f"**Sample After Predictions:** {set(after_predictions)}")
                
            if changes_detected > 0:
                st.warning("Alert: Tree cover loss detected across multiple coordinate segments.")
            else:
                st.success("No significant deforestation events detected in this frame.")
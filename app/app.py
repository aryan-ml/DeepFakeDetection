import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import json
import os
import pandas as pd

# ###########################################
# # 1. PAGE CONFIG
# ###########################################
# st.set_page_config(
#     page_title="DeepFake Detector AI",
#     page_icon="🕵️‍♂️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

###########################################
# 2. MODEL & CONFIG LOADING
###########################################
tf.keras.mixed_precision.set_global_policy("float32")
tf.keras.backend.clear_session()

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "final_resnet_savedmodel.keras")
CONFIG_PATH = os.path.join(current_dir, "config.json")

st.title("DeepFake Video Detector (ResNet50 Model)")
st.markdown('**Developed by: Aryan**')
st.caption("Helps you spot FAKE people... digitally, at least :)")

@st.cache_resource
def load_model_and_config():
    # 1. Load Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    # 2. Load Config
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        threshold = float(cfg["threshold"])
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return model, 0.5 # Default threshold if fails

    return model, threshold

# Call the function once
model, THRESHOLD = load_model_and_config()

if model is not None:
    st.success("Model loaded successfully!")

###########################################
# 3. ADVANCED FACE EXTRACTION
###########################################
detector = MTCNN()
IMG_SIZE = 224
MAX_FRAMES = 60  # Limit to 50 frames to prevent memory crash

def extract_and_predict(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # We need two lists: 
    # 1. 'processed_faces' for the AI (normalized, weird looking)
    # 2. 'display_faces' for the Human UI (normal colors)
    processed_faces = []
    display_faces = []
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to speed up (analyze every 5th frame)
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)
        
        if len(detections) > 0:
            x, y, w, h = detections[0]["box"]
            x, y = max(0, x), max(0, y)
            
            # Crop the face
            face = rgb[y:y+h, x:x+w]
            
            try:
                # Resize for AI
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                
                # 1. Store for UI (Clean RGB image)
                display_faces.append(face_resized)
                
                # 2. Store for AI (Preprocessed with ResNet logic)
                face_preproc = tf.keras.applications.resnet.preprocess_input(face_resized)
                processed_faces.append(face_preproc)
                
            except Exception:
                pass 

        frame_idx += 1
        if len(processed_faces) >= MAX_FRAMES:
            break
            
    cap.release()
    
    if len(processed_faces) == 0:
        return None, 0, [], []

    # Convert to numpy for batch prediction
    processed_faces = np.array(processed_faces)
    
    # Get scores for EVERY frame individually
    per_frame_preds = model.predict(processed_faces, batch_size=16, verbose=0).ravel()
    
    # Average them for the final score
    video_prob = float(np.mean(per_frame_preds))
    label = "FAKE" if video_prob >= THRESHOLD else "REAL"
    
    return label, video_prob, per_frame_preds, display_faces

###########################################
# 4. STREAMLIT UI
###########################################

with st.sidebar:
    st.title("DeepFake Detector")
    st.info("This tool extracts faces frame-by-frame and analyzes them for manipulation artifacts.")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

st.markdown("<h2 style='text-align: center;'>🕵️‍♂️ DeepFake Analysis Dashboard</h2>", unsafe_allow_html=True)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Video")
        st.video(tfile.name)
        
    with col2:
        with st.spinner("Scanning video for faces..."):
            label, prob, per_frame_scores, face_images = extract_and_predict(tfile.name)
            
        if label:
            # Result Card
            if label == "FAKE":
                st.error(f"### 🚨 VERDICT: **FAKE**")
                st.metric("Confidence", f"{prob*100:.2f}%", delta="High Risk", delta_color="inverse")
            else:
                st.success(f"### ✅ VERDICT: **REAL**")
                st.metric("Confidence", f"{(1-prob)*100:.2f}%", delta="Safe", delta_color="normal")
        else:
            st.warning("No faces detected in the video.")
            
    # ---------------------------------------------------------
    # NEW SECTION: GRID DISPLAY (No HTML)
    # ---------------------------------------------------------
    if label and len(face_images) > 0:
        st.divider()
        st.subheader(f"📸 Analyzed Faces ({len(face_images)} Frames Extracted)")
        
        # Line Chart of Scores
        st.caption("Suspicion level per frame over time:")
        st.line_chart(per_frame_scores)

        st.write("### 🖼️ Frame-by-Frame Breakdown")
        
        # Display images in a grid of 5 columns
        cols = st.columns(5)
        
        for idx, (img, score) in enumerate(zip(face_images, per_frame_scores)):
            with cols[idx % 5]:
                # Display the image
                st.image(img, use_column_width=True)
                
                # Display the score below it
                if score > THRESHOLD:
                    st.error(f"**{score*100:.0f}% FAKE**")
                else:
                    st.success(f"**REAL**")
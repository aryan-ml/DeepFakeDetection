import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import json
import os
import pandas as pd

###########################################
# 1. PAGE CONFIG & SAFETY
###########################################
st.set_page_config(layout="wide", page_title="DeepFake Detector") # Wide mode makes the face grid look better

# Force inference to use float32 (reduces GPU workspace)
tf.keras.mixed_precision.set_global_policy("float32")

# Clear TF session before loading model
tf.keras.backend.clear_session()

###########################################
# 2. LOAD MODEL + THRESHOLD (Cached)
###########################################

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "final_resnet_savedmodel.keras")
CONFIG_PATH = os.path.join(current_dir, "config.json")

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
        return model, 0.5 

    return model, threshold

# Initialize
model, THRESHOLD = load_model_and_config()

if model is None:
    st.stop()

###########################################
# 3. ANALYSIS FUNCTION
###########################################

detector = MTCNN()
IMG_SIZE = 224
MAX_FRAMES = 60

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    processed_faces = []
    display_faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAMPLE FRAMES: every 5th frame
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if len(detections) > 0:
            x, y, w, h = detections[0]["box"]
            x, y = max(0, x), max(0, y)
            
            # Crop Face
            face = rgb[y:y+h, x:x+w]
            
            try:
                # 1. Resize for UI (Display)
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                display_faces.append(face_resized.copy())

                # 2. Preprocess for AI (Prediction)
                face_preproc = tf.keras.applications.resnet.preprocess_input(face_resized)
                processed_faces.append(face_preproc)
            except:
                pass

        frame_idx += 1
        if len(processed_faces) >= MAX_FRAMES:
            break

    cap.release()
    
    if len(processed_faces) == 0:
        return None, 0, [], []

    # Predict
    processed_faces = np.array(processed_faces)
    per_frame_preds = model.predict(processed_faces, batch_size=16, verbose=0).ravel()
    
    # Aggregate
    video_prob = float(np.mean(per_frame_preds))
    label = "FAKE" if video_prob >= THRESHOLD else "REAL"

    return label, video_prob, per_frame_preds, display_faces

###########################################
# 4. STREAMLIT UI (Your Layout)
###########################################

st.title("DeepFake Video Detector (ResNet50 Model)")
st.markdown('**Developed by: Aryan**')
st.caption("Helps you spot FAKE people... digitally, at least :)")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())

    st.video(uploaded_video)
    
    with st.spinner("⏳ Processing the video... Please wait."):
        label, prob, frame_scores, face_images = analyze_video(temp_file.name)

    if label is None:
        st.error("⚠ No face detected in the video.")
    else:
        # --- Main Result ---
        if label == "FAKE":
            st.error(f"🔴 Prediction: **{label}**")
        else:
            st.success(f"🟢 Prediction: **{label}**")

        st.write(f"### Probability of FAKE: **{prob:.3f}**")
        
        # --- NEW FEATURES BELOW ---
        st.divider()
        st.subheader("📊 Detailed Analysis")
        
        # 1. The Graph
        st.write("**Frame-by-Frame Suspicion Level:**")
        chart_data = pd.DataFrame(frame_scores, columns=["Fake Probability"])
        st.line_chart(chart_data)
        
        # 2. The Face Grid
        st.write(f"**Extracted Faces ({len(face_images)} frames):**")
        
        # Create a grid of 5 columns
        cols = st.columns(5)
        for idx, (img, score) in enumerate(zip(face_images, frame_scores)):
            with cols[idx % 5]:
                st.image(img, use_column_width=True)
                if score > THRESHOLD:
                    st.caption(f"🚨 **{score*100:.0f}% FAKE**")
                else:
                    st.caption(f"✅ **REAL**")

###########################################
# END OF FILE
###########################################
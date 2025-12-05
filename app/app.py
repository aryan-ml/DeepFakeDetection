import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import json
import os

###########################################
# 1. SAFETY SETTINGS (Prevents GPU Crashes)
###########################################

# Force inference to use float32 (reduces GPU workspace)
tf.keras.mixed_precision.set_global_policy("float32")

# Clear TF session before loading model
tf.keras.backend.clear_session()

###########################################
# 2. LOAD MODEL + THRESHOLD
###########################################

import os

# Get the folder where THIS script (app.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the correct paths relative to that folder
MODEL_PATH = os.path.join(current_dir, "final_resnet_savedmodel.keras")
CONFIG_PATH = os.path.join(current_dir, "config.json")

st.title("🔍 DeepFake Video Detector (ResNet50 Model)")

# Load model 
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load threshold
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

THRESHOLD = float(cfg["threshold"])

###########################################
# 3. FACE DETECTOR (MTCNN)
###########################################

detector = MTCNN()
IMG_SIZE = 224

# Limit number of frames for inference (BEST PRACTICE)
MAX_FRAMES = 60   # Safe & keeps accuracy identical


###########################################
# 4. FACE EXTRACTION FUNCTION (SAFE)
###########################################

def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAMPLE FRAMES: every 5th frame (like training)
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if len(detections) > 0:
            x, y, w, h = detections[0]["box"]
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            # Apply same preprocessing as training
            face = tf.keras.applications.resnet.preprocess_input(face)

            faces.append(face)

        frame_idx += 1

        # Safety limit
        if len(faces) >= MAX_FRAMES:
            break

    cap.release()
    return np.array(faces)


###########################################
# 5. PREDICT VIDEO (NO GPU OOM)
###########################################

def predict_video(video_path, batch_size=8):
    faces = extract_faces(video_path)

    if len(faces) == 0:
        return None, 0

    preds = []

    # Predict in small batches → NO MEMORY CRASH
    for i in range(0, len(faces), batch_size):
        batch = faces[i:i+batch_size]
        batch = np.array(batch, dtype=np.float32)
        p = model.predict(batch, verbose=0).ravel()
        preds.extend(p)

    preds = np.array(preds)

    # SAME aggregation as training & evaluation → same accuracy
    video_prob = float(np.mean(preds))
    label = "FAKE" if video_prob >= THRESHOLD else "REAL"

    return label, video_prob


###########################################
# 6. STREAMLIT UI
###########################################

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())

    st.video(uploaded_video)
    st.write("⏳ Processing the video... Please wait.")

    label, prob = predict_video(temp_file.name)

    if label is None:
        st.error("⚠ No face detected in the video.")
    else:
        if label == "FAKE":
            st.error(f"🔴 Prediction: **{label}**")
        else:
            st.success(f"🟢 Prediction: **{label}**")

        st.write(f"### Probability of FAKE: **{prob:.3f}**")

###########################################
# END OF FILE
###########################################

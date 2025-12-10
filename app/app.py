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
# 1. PAGE CONFIG
###########################################
st.set_page_config(
    page_title="DeepFake Detector AI",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

###########################################
# 2. MODEL & CONFIG LOADING
###########################################
tf.keras.mixed_precision.set_global_policy("float32")
tf.keras.backend.clear_session()

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "final_resnet_savedmodel.keras")
CONFIG_PATH = os.path.join(current_dir, "config.json")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)
THRESHOLD = float(cfg["threshold"])

###########################################
# 3. ADVANCED FACE EXTRACTION
###########################################
detector = MTCNN()
IMG_SIZE = 224
MAX_FRAMES = 50  # Limit to 50 frames to prevent memory crash

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
    st.image("https://cdn-icons-png.flaticon.com/512/11590/11590673.png", width=80)
    st.title("DeepFake Detector")
    st.info("Now with **Frame-by-Frame** X-Ray analysis.")
    uploaded_video = st.file_uploader("📂 Upload Video", type=["mp4", "mov", "avi"])

st.markdown("<h2 style='text-align: center;'>🕵️‍♂️ DeepFake Analysis Dashboard</h2>", unsafe_allow_html=True)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📺 Original Video")
        st.video(tfile.name)
        
    with col2:
        with st.spinner("extracting faces & analyzing frames..."):
            label, prob, per_frame_scores, face_images = extract_and_predict(tfile.name)
            
        if label:
            # Result Card
            if label == "FAKE":
                st.error(f"### 🚨 FINAL VERDICT: **FAKE**")
                st.metric("Fake Confidence", f"{prob*100:.2f}%")
            else:
                st.success(f"### ✅ FINAL VERDICT: **REAL**")
                st.metric("Real Confidence", f"{(1-prob)*100:.2f}%")
        else:
            st.warning("No faces found.")

    # ---------------------------------------------------------
    # NEW SECTION: EXPLAINABLE AI (Frame Analysis)
    # ---------------------------------------------------------
    if label:
        st.divider()
        st.subheader("🧠 What did the AI see?")
        st.write("The model cropped faces from the video and scored each one individually.")
        
        # 1. Line Chart of Scores
        st.markdown("#### 📈 Frame-by-Frame Suspicion Level")
        chart_data = pd.DataFrame(per_frame_scores, columns=["Fake Probability"])
        st.line_chart(chart_data)
        st.caption("Peaks in the graph indicate specific moments where the AI detected manipulation.")

        # 2. Face Gallery
        st.markdown("#### 📸 Analyzed Frames")
        st.write("Below are the exact crops the AI analyzed. **Red** = Suspicious, **Green** = Clean.")
        
        # CSS to create a scrolling row of images
        st.markdown("""
        <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            padding: 10px;
            gap: 10px;
        }
        .face-card {
            min-width: 120px;
            text-align: center;
            background: #f0f2f6;
            padding: 5px;
            border-radius: 8px;
        }
        .fake-border { border: 3px solid #ff4b4b; }
        .real-border { border: 3px solid #09ab3b; }
        </style>
        """, unsafe_allow_html=True)

        # Build HTML for the gallery manually to allow custom styling
        html_content = '<div class="scroll-container">'
        
        # Show up to 10 key frames (to avoid overcrowding)
        # We zip the images with their scores
        for img, score in zip(face_images[:15], per_frame_scores[:15]):
            
            # Encode image to base64 for HTML display
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            import base64
            img_str = base64.b64encode(buffer).decode()
            
            # Color logic
            border_class = "fake-border" if score > THRESHOLD else "real-border"
            score_text = f"{score*100:.0f}% Fake"
            
            html_content += f"""
            <div class="face-card {border_class}">
                <img src="data:image/jpeg;base64,{img_str}" width="100" style="border-radius:5px;">
                <p style="font-size:12px; margin:5px 0;"><b>{score_text}</b></p>
            </div>
            """
        
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)
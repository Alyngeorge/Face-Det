import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from io import BytesIO
from datetime import datetime
from PIL import Image
from fer import FER
from streamlit_shadcn_ui import card, tabs, button

# ---------------------------
# Basic Dark Mode Toggle
# ---------------------------
st.set_page_config(page_title="🎭 Emotion Detector Using FER", layout="centered")

mode = st.toggle("🌙 Dark Mode")
if mode:
    st.markdown("""<style>body{background-color:#121212; color:white;}</style>""", unsafe_allow_html=True)

# ---------------------------
# Load FER Detector
# ---------------------------
detector = FER(mtcnn=True)

if "history" not in st.session_state:
    st.session_state.history = []

tab = tabs(options=["Home", "Upload", "Capture", "Live"], default_value="Home")

# ---------------------------
# Emotion Message Text
# ---------------------------
def get_message(emotion):
    return {
        "happy": "😊 You're radiating joy!",
        "sad": "😢 It's okay to feel down sometimes.",
        "angry": "😠 Breathe deep. You're strong.",
        "fear": "😨 You're safe and supported.",
        "surprise": "😮 Something unexpected?",
        "neutral": "😐 Steady and balanced.",
        "disgust": "😖 Something doesn't feel right.",
    }.get(emotion, f"😐 You seem {emotion}.")

# ---------------------------
# Emotion Analyzer
# ---------------------------
def analyze_emotions(image_bytes, frame=None, threshold=50):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = detector.detect_emotions(img)
    cards = []

    for i, result in enumerate(results):
        emotions = result["emotions"]
        mood, conf = max(emotions.items(), key=lambda x: x[1])
        conf *= 100
        if conf < threshold:
            continue
        box = result["box"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "timestamp": timestamp,
            "face": i + 1,
            "emotion": mood,
            "confidence": round(conf, 2)
        })

        if frame is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{mood.capitalize()} ({conf:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cards.append({
            "index": i + 1,
            "emotion": mood,
            "confidence": conf,
            "message": get_message(mood),
        })

    return cards, frame

# ---------------------------
# Display Emotion Cards
# ---------------------------
def display_cards(cards):
    for c in cards:
        st.markdown("---")
        st.markdown(f"**Emotion:** `{c['emotion'].capitalize()}`")
        st.markdown(
            f"""
            <div style='background:#eee;border-radius:6px;overflow:hidden;width:100%;'>
              <div style='background:#90caf9;width:{int(c['confidence'])}%;padding:6px;color:black;text-align:center;'>
                Confidence: {c['confidence']:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        card(title=f"Face {c['index']}", description=c["message"], variant="ghost")

# ---------------------------
# Upload Image
# ---------------------------
if tab == "Upload":
    st.subheader("🖼 Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_bytes = uploaded.read()
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        cards, annotated = analyze_emotions(image_bytes, frame)
        st.image(annotated, channels="BGR", use_container_width=True)
        display_cards(cards)

# ---------------------------
# Capture Webcam
# ---------------------------
elif tab == "Capture":
    st.subheader("📷 Capture Image")
    if st.button("📸 Snap"):
        cap = cv2.VideoCapture(0)
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()
            cards, annotated = analyze_emotions(image_bytes, frame)
            st.image(annotated, channels="BGR", use_container_width=True)
            display_cards(cards)
        else:
            st.error("Failed to capture image.")

# ---------------------------
# Live Feed
# ---------------------------
elif tab == "Live":
    st.subheader("🎥 Live Webcam Feed (60 sec)")
    if st.button("▶️ Start Live"):
        cap = cv2.VideoCapture(0)
        start = time.time()
        frame_space = st.empty()
        card_space = st.empty()

        while time.time() - start < 60:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam error.")
                break
            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()
            cards, annotated = analyze_emotions(image_bytes, frame)
            frame_space.image(annotated, channels="BGR", use_container_width=True)
            with card_space:
                display_cards(cards)
            time.sleep(5)

        cap.release()
        cv2.destroyAllWindows()
        st.success("✅ Live analysis ended.")

# ---------------------------
# Home + Export
# ---------------------------
elif tab == "Home":
    st.markdown("### 🧭 Dashboard")
    st.markdown("Select a tab to analyze emotions via image, webcam capture, or live feed.")
    if len(st.session_state.history) > 0:
        if button("📂 Export Emotion History to CSV", key="export", variant="default", size="lg"):
            df = pd.DataFrame(st.session_state.history)
            df.to_csv("emotion_history.csv", index=False)
            st.success("✅ Exported as emotion_history.csv")

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from utils.aspect_ratio_processor import AspectRatioProcessor

st.set_page_config(layout="wide")
st.title("Driver Drowsiness Detection (Cloud Version)")

# =============================
# LOAD MODEL (NO PATCH)
# =============================
@st.cache_resource
def load_model():
    return keras.models.load_model("model/model_dash_final.keras", compile=False)

model = load_model()

# =============================
# MEDIAPIPE INIT
# =============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =============================
# CONSTANT
# =============================
LEFT_EYE_INDICES = [362,263,387,386,385,384,398,381,380,374,373]
RIGHT_EYE_INDICES = [33,133,160,159,158,157,173,153,145,144,163]
MOUTH_INDICES = [78,308,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

ALL_ROI_IDX = list(set(LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_INDICES))

EAR_THRESH = 0.20

# =============================
# PREPROCESS
# =============================
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(gray, lut)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(corrected)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# =============================
# DRAW LANDMARK
# =============================
def draw_landmarks(image, face_landmarks):
    annotated = image.copy()
    h, w, _ = image.shape

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in ALL_ROI_IDX:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 2, (0,255,0), -1)

    return annotated

# =============================
# INPUT
# =============================
st.sidebar.header("Input Source")

input_mode = st.sidebar.radio("Choose Input", ["Camera", "Upload Image"])

image = None

if input_mode == "Camera":
    uploaded = st.camera_input("Take a picture")
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

else:
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# =============================
# PROCESS
# =============================
if image is not None:

    st.subheader("Processing Result")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = image.shape
            arp = AspectRatioProcessor(w, h)

            # --- EAR ---
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]

            ear_left = arp.get_aspect_ratio(left_eye)
            ear_right = arp.get_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2

            # --- MODEL ---
            processed = preprocess_image(image)
            annotated = draw_landmarks(processed, face_landmarks)

            input_tensor = annotated / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            pred = model.predict(input_tensor, verbose=0)[0][0]
            confidence = float(pred)

            is_yawning = confidence > 0.5
            is_microsleep = ear < EAR_THRESH

            # --- VISUAL ---
            display = image.copy()

            if is_yawning:
                cv2.putText(display, "YAWNING", (30,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            if is_microsleep:
                cv2.putText(display, "MICROSLEEP", (30,120),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            cv2.putText(display, f"Conf: {confidence:.2f}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            cv2.putText(display, f"EAR: {ear:.2f}", (30,160),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            st.image(display)

    else:
        st.warning("No face detected")

else:
    st.info("Please provide input image")
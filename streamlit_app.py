import streamlit as st
import cv2
import time
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf
from tensorflow import keras
from utils.aspect_ratio_processor import AspectRatioProcessor

st.set_page_config(layout="wide")
st.title("Driver Drowsiness Detection")

# ====================================================================
# --- PATCH KERAS DENSE LAYER (Fix quantization_config error) ---
# Kita letakkan di sini agar Keras sudah ditambal sebelum model dimuat
original_dense_init = keras.layers.Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    if kwargs:  
        kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)

keras.layers.Dense.__init__ = patched_dense_init
# ====================================================================

# --- INIT MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return keras.models.load_model("model/model_dash_final.keras", compile=False)

yawn_model = load_model()

# --- INIT SOUND ---
pygame.mixer.init()
sound_drowsy = pygame.mixer.Sound("sound/beep.wav")
sound_microsleep = pygame.mixer.Sound("sound/alarm.mp3")

# --- CONSTANT ---
LEFT_EYE_INDICES = [362,263,387,386,385,384,398,381,380,374,373]
RIGHT_EYE_INDICES = [33,133,160,159,158,157,173,153,145,144,163]
MOUTH_INDICES = [78,308,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

ALL_ROI_IDX = list(set(LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_INDICES))

EAR_THRESH = 0.20
EYE_PADDING = 5
MOUTH_PADDING = 10

# Global variables
frame_count = 0
confidence = 0.0
ear = 0.0
is_yawning = False

# --- PREPROCESSING (SAMA DENGAN TRAINING) ---
def preprocess_image(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(gray, lut)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# --- DRAW LANDMARK ---
def draw_landmarks_inference(image, face_landmarks):
    annotated = image.copy()
    h, w, _ = image.shape

    mesh_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,255,255))

    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mesh_spec
    )

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in ALL_ROI_IDX:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 3, (0,255,0), -1)

    return annotated

# --- SESSION STATE ---
if "microsleep_active" not in st.session_state:
    st.session_state.microsleep_active = False
    st.session_state.eyes_open_timer = None
    st.session_state.eyes_closed_timer = None
    st.session_state.yawning_active = False
    st.session_state.yawning_start_timer = None
    st.session_state.is_alarm_playing = False

# --- UI ---
col1, col2, col3 = st.columns(3)
with col1:
    start = st.button("Start Camera")
with col2:
    stop = st.button("Stop")
with col3:
    show_landmarks = st.checkbox("Tampilkan Landmark", value=True)

# --- STREAMLIT DISPLAY ---
frame_placeholder = st.empty()

if start:
    st.session_state['run_camera'] = True

if stop:
    st.session_state['run_camera'] = False

if st.session_state.get('run_camera', False):
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while cap.isOpened() and st.session_state.get('run_camera', False):
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        aspect_ratio_processor = AspectRatioProcessor(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # --- LANDMARK ROI ---
                left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
                right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]
                mouth = [face_landmarks.landmark[i] for i in MOUTH_INDICES]

                # --- EAR ---
                left_ratio = aspect_ratio_processor.get_aspect_ratio(left_eye)
                right_ratio = aspect_ratio_processor.get_aspect_ratio(right_eye)
                ear = (left_ratio + right_ratio) / 2.0

                now = time.time()

                # --- YAWNING DETECTION (MODEL) ---
                processed_frame = preprocess_image(frame)

                annotated = processed_frame.copy()

                annotated = draw_landmarks_inference(processed_frame, face_landmarks)

                input_tensor = annotated / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)
                if frame_count % 5 == 0:
                    pred = yawn_model.predict(input_tensor, verbose=0)[0][0]
                else:
                    frame_count += 1

                confidence = float(pred)
                is_yawning = confidence > 0.5

                # --- Yawning ---
                if is_yawning:
                    if not st.session_state.yawning_active:
                        st.session_state.yawning_active = True
                        st.session_state.yawning_start_timer = now
                        sound_drowsy.play()

                    if st.session_state.yawning_active:
                        if now - st.session_state.yawning_start_timer <= 5:
                            cv2.putText(frame, "YAWNING", (50,120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
                        else:
                            st.session_state.yawning_active = False
                
                # --- MICROSLEEP ---
                if ear < EAR_THRESH:
                    st.session_state.eyes_open_timer = None
                    if st.session_state.eyes_closed_timer is None:
                        st.session_state.eyes_closed_timer = now
                    if now - st.session_state.eyes_closed_timer > 1.5:
                        st.session_state.microsleep_active = True
                else:
                    st.session_state.eyes_closed_timer = None
                    if st.session_state.microsleep_active:
                        if st.session_state.eyes_open_timer is None:
                            st.session_state.eyes_open_timer = now
                        if now - st.session_state.eyes_open_timer > 1:
                            st.session_state.microsleep_active = False

                if st.session_state.microsleep_active:
                    cv2.putText(frame, "MICROSLEEP", (50,150),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                    if not st.session_state.is_alarm_playing:
                        sound_microsleep.play(loops=-1)
                        st.session_state.is_alarm_playing = True
                else:
                    if st.session_state.is_alarm_playing:
                        sound_microsleep.stop()
                        st.session_state.is_alarm_playing = False
                    
                # --- Confidence Model ---
                cv2.putText(frame, f"Conf: {confidence:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                # -- EAR ---
                cv2.putText(frame, f"EAR: {ear:.2f}", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # --- FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (w-150,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        # --- STREAMLIT DISPLAY ---
        if results.multi_face_landmarks and show_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = draw_landmarks_inference(frame, face_landmarks)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")


    cap.release()
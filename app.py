import cv2
import time
import mediapipe as mp
import numpy as np
import pygame
from tensorflow import keras
from utils.aspect_ratio_processor import AspectRatioProcessor

# ====================================================================
# PATCH KERAS
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
yawn_model = keras.models.load_model("model/model_dash_final.keras", compile=False)

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

# --- STATE ---
microsleep_active = False
eyes_open_timer = None
eyes_closed_timer = None
yawning_active = False
yawning_start_timer = None
is_alarm_playing = False

# --- CONTROL ---
show_landmarks = True

# --- GLOBAL ---
frame_count = 0
confidence = 0.0
ear = 0.0
is_yawning = False

# --- PREPROCESS ---
def preprocess_image(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(gray, lut)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# --- DRAW LANDMARK ---
def draw_landmarks_inference(image, face_landmarks):
    annotated = image.copy()
    h, w, _ = image.shape

    mesh_spec = mp_drawing.DrawingSpec(
        thickness=1,
        circle_radius=1,
        color=(255,255,255)
    )

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

# --- CAMERA ---
cap = cv2.VideoCapture(0)
prev_time = 0

print("Tekan 'q' untuk keluar | 'l' untuk toggle landmark")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    aspect_ratio_processor = AspectRatioProcessor(w, h)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # --- EAR ---
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]

            left_ratio = aspect_ratio_processor.get_aspect_ratio(left_eye)
            right_ratio = aspect_ratio_processor.get_aspect_ratio(right_eye)
            ear = (left_ratio + right_ratio) / 2.0

            now = time.time()

            # --- YAWNING MODEL ---
            processed = preprocess_image(frame)
            annotated = draw_landmarks_inference(processed, face_landmarks)

            input_tensor = annotated / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            if frame_count % 5 == 0:
                pred = yawn_model.predict(input_tensor, verbose=0)[0][0]
                confidence = float(pred)
            frame_count += 1

            is_yawning = confidence > 0.75

            # --- YAWNING LOGIC ---
            if is_yawning:
                if not yawning_active:
                    yawning_active = True
                    yawning_start_timer = now
                    sound_drowsy.play()

                if now - yawning_start_timer <= 5:
                    cv2.putText(frame, "YAWNING", (50,120),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                else:
                    yawning_active = False

            # --- MICROSLEEP ---
            if ear < EAR_THRESH:
                eyes_open_timer = None
                if eyes_closed_timer is None:
                    eyes_closed_timer = now
                if now - eyes_closed_timer > 1.5:
                    microsleep_active = True
            else:
                eyes_closed_timer = None
                if microsleep_active:
                    if eyes_open_timer is None:
                        eyes_open_timer = now
                    if now - eyes_open_timer > 1:
                        microsleep_active = False

            if microsleep_active:
                cv2.putText(frame, "MICROSLEEP", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                if not is_alarm_playing:
                    sound_microsleep.play(loops=-1)
                    is_alarm_playing = True
            else:
                if is_alarm_playing:
                    sound_microsleep.stop()
                    is_alarm_playing = False

            # --- INFO ---
            cv2.putText(frame, f"Conf: {confidence:.2f}", (30,70),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # --- VISUAL ONLY ---
            if show_landmarks:
                frame = draw_landmarks_inference(frame, face_landmarks)

    # --- FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (w-150,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l'):
        show_landmarks = not show_landmarks

cap.release()
cv2.destroyAllWindows()
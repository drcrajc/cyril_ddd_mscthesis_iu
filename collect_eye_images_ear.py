import cv2
import mediapipe as mp
# from mediapipe.python.solutions import face_mesh
import time
import uuid
import os
from scipy.spatial import distance as dist

# -------------------------------
# EAR calculation
# -------------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------------------------------
# Config
# -------------------------------
EAR_THRESHOLD = 0.23
SAVE_INTERVAL = 0.5  # seconds
BASE_PATH = "data/images"

open_path = os.path.join(BASE_PATH, "open_eye")
closed_path = os.path.join(BASE_PATH, "closed_eye")

os.makedirs(open_path, exist_ok=True)
os.makedirs(closed_path, exist_ok=True)

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
#face_mesh = face_mesh.FaceMesh(refine_landmarks=True)


# Eye landmark indices (MediaPipe)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)
last_saved = time.time()

print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        status = "OPEN"
        color = (0, 255, 0)

        if ear < EAR_THRESHOLD:
            status = "CLOSED"
            color = (0, 0, 255)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, status, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Save images periodically
        if time.time() - last_saved > SAVE_INTERVAL:
            filename = f"{uuid.uuid4()}.jpg"
            if status == "OPEN":
                cv2.imwrite(os.path.join(open_path, filename), frame)
            else:
                cv2.imwrite(os.path.join(closed_path, filename), frame)
            last_saved = time.time()

    cv2.imshow("EAR Data Collection", frame)

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

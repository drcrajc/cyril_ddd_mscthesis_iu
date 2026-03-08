import cv2
import mediapipe as mp
import os
from scipy.spatial import distance as dist

# ---------------- CONFIG ----------------
IMG_DIR = "data/images"
LBL_DIR = "data/labels"
EAR_THRESHOLD = 0.23

os.makedirs(LBL_DIR, exist_ok=True)

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------------------------------

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        continue

    landmarks = result.multi_face_landmarks[0].landmark

    left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
    right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    class_id = 1 if ear < EAR_THRESHOLD else 0  # 0=open_eye, 1=closed_eye

    # Combine both eyes for bounding box
    eye_points = left_eye + right_eye
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Convert to YOLO format
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    box_w = (x_max - x_min) / w
    box_h = (y_max - y_min) / h

    label_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

import mediapipe as mp

print("MediaPipe version:", mp.__version__)
print("Has solutions:", hasattr(mp, "solutions"))

fm = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
print("FaceMesh initialized successfully")

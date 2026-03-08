import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# =========================
# PARAMETERS
# =========================

IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 16
LR = 0.001

DATASET_PATH = "yolov5/datasets"

# =========================
# LOAD DATA
# =========================

images = []
labels = []

classes = ["open_eye", "closed_eye"]

for class_id, class_name in enumerate(classes):

    img_dir = os.path.join(DATASET_PATH, "images", class_name)
    lbl_dir = os.path.join(DATASET_PATH, "labels", class_name)

    for file in os.listdir(img_dir):

        if file.endswith(".jpg") or file.endswith(".png"):

            img_path = os.path.join(img_dir, file)
            label_path = os.path.join(lbl_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            with open(label_path) as f:
                line = f.readline().strip()

            parts = line.split()

            cls = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            # Convert YOLO bbox → pixel coordinates
            x_center = int(xc * w)
            y_center = int(yc * h)
            box_w = int(bw * w)
            box_h = int(bh * h)

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            eye = img[y1:y2, x1:x2]

            if eye.size == 0:
                continue

            eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
            eye = eye / 255.0

            images.append(eye)
            labels.append(cls)

images = np.array(images)
labels = np.array(labels)

print("Total samples:", len(images))

# =========================
# TRAIN VAL TEST SPLIT
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =========================
# MODEL
# =========================

base_model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = SGD(
    learning_rate=LR,
    momentum=0.9
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# TEST
# =========================

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)

pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# =========================
# SAVE MODEL
# =========================

model.save("resnet50v2_eye_model.h5")

# =========================
# TFLITE
# =========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("resnet50v2_eye_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved")
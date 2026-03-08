import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ============================
# PARAMETERS
# ============================

IMG_SIZE = 224
DATASET_PATH = "yolov5/datasets"

images_path = os.path.join(DATASET_PATH, "images")
labels_path = os.path.join(DATASET_PATH, "labels")

RESULT_DIR = "results_resnet50v2"
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================
# LOAD DATA
# ============================

images = []
labels = []

class_map = {
    "closed_eye": 0,
    "open_eye": 1
}

for class_name in os.listdir(images_path):

    class_folder = os.path.join(images_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    label_value = class_map[class_name]

    for img_name in os.listdir(class_folder):

        img_path = os.path.join(class_folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        images.append(img)
        labels.append(label_value)

images = np.array(images)
labels = np.array(labels)

print("Total Images:", len(images))

# ============================
# SPLIT DATA
# ============================

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

# ============================
# BUILD MODEL
# ============================

base_model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================
# TRAIN MODEL
# ============================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16
)

# ============================
# TEST MODEL
# ============================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# ============================
# PREDICTIONS
# ============================

pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

# ============================
# CONFUSION MATRIX
# ============================

cm = confusion_matrix(y_test, pred_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# ============================
# TRAINING PLOTS
# ============================

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Val"])
plt.savefig(os.path.join(RESULT_DIR, "accuracy_plot.png"))
plt.close()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Val"])
plt.savefig(os.path.join(RESULT_DIR, "loss_plot.png"))
plt.close()

# ============================
# CLASSIFICATION REPORT
# ============================

report = classification_report(
    y_test,
    pred_labels,
    target_names=["closed_eye","open_eye"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# ============================
# SAVE TO EXCEL
# ============================

summary_df = pd.DataFrame({
    "Metric":["Test Accuracy","Test Loss","Train Size","Val Size","Test Size"],
    "Value":[accuracy,loss,len(X_train),len(X_val),len(X_test)]
})

cm_df = pd.DataFrame(cm,
                     columns=["Pred Closed","Pred Open"],
                     index=["Actual Closed","Actual Open"])

history_df = pd.DataFrame(history.history)

excel_path = os.path.join(RESULT_DIR,"results.xlsx")

with pd.ExcelWriter(excel_path) as writer:

    summary_df.to_excel(writer,sheet_name="Summary",index=False)
    report_df.to_excel(writer,sheet_name="Metrics")
    cm_df.to_excel(writer,sheet_name="Confusion Matrix")
    history_df.to_excel(writer,sheet_name="Training History")

print("Results saved to Excel:", excel_path)
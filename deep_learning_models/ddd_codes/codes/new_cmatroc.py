import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from glob import glob

# === CONFIGURATION ===
image_root = 'skin_ds/test'
result_file = 'xception/test_results.xlsx'
cm_plot_file = 'confusion_matrix.png'
roc_plot_file = 'roc_curve.png'
grid_plot_file = 'annotated_grid.png'
supported_formats = ('*.jpg', '*.jpeg', '*.png')

# === STEP 1: LOAD PREDICTIONS ===
df = pd.read_excel(result_file)
df = df.rename(columns={
    'Image Name': 'filename',
    'Predicted Class': 'pred_label',
    'True Class': 'true_label'
})

true_labels = df['true_label']
pred_labels = df['pred_label']
classes = sorted(true_labels.unique())
n_classes = len(classes)

# Label encoding
le = LabelEncoder()
true_encoded = le.fit_transform(true_labels)
pred_encoded = le.transform(pred_labels)

# === STEP 2: CONFUSION MATRIX ===
cm = confusion_matrix(true_encoded, pred_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_plot_file, dpi=300)
plt.show()
print(f"[✅] Confusion matrix saved to {cm_plot_file}")

# === STEP 3: ROC CURVE ===
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(true_labels)
y_pred_bin = lb.transform(pred_labels)

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(roc_plot_file, dpi=300)
plt.show()
print(f"[✅] ROC curve saved to {roc_plot_file}")

# === STEP 4: IMAGE ANNOTATION PREPARATION ===
image_paths = []
for ext in supported_formats:
    image_paths.extend(glob(os.path.join(image_root, '**', ext), recursive=True))

# Map filename to full path
path_dict = {os.path.basename(p): p for p in image_paths}

# Add full paths to dataframe
df['full_path'] = df['filename'].map(path_dict)
df = df.dropna(subset=['full_path'])

# === STEP 5: GET 2 CORRECT + 2 INCORRECT IMAGES PER CLASS ===
annotated_set = []

for cls in classes:
    correct = df[(df['true_label'] == cls) & (df['true_label'] == df['pred_label'])].head(2)
    wrong = df[(df['true_label'] == cls) & (df['true_label'] != df['pred_label'])].head(2)
    annotated_set.extend(correct.to_dict('records'))
    annotated_set.extend(wrong.to_dict('records'))

# === STEP 6: PLOT FINAL GRID ===
cols = 4
rows = n_classes
plt.figure(figsize=(cols * 4, rows * 4))

for idx, row in enumerate(annotated_set):
    img = cv2.imread(row['full_path'])
    if img is None:
        print(f"[⚠️] Unable to load image: {row['full_path']}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    true_cls = row['true_label']
    pred_cls = row['pred_label']
    color = 'green' if true_cls == pred_cls else 'red'
    
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {true_cls}\nPred: {pred_cls}", fontsize=10, color=color)

plt.tight_layout()
plt.savefig(grid_plot_file, dpi=300)
plt.show()
print(f"[✅] Annotated image grid saved to {grid_plot_file}")

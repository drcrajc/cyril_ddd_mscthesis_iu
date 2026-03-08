import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# ===================== PATHS =====================
train_folder = "new_ddd_ds/train"
val_folder   = "new_ddd_ds/valid"
test_folder  = "new_ddd_ds/test"

# ===================== PARAMETERS =====================
num_classes = 2
input_size  = (224, 224)
batch_size  = 32
epochs      = 5

# ===================== DATA GENERATORS =====================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = val_test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ===================== VGG19 MODEL =====================
base_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===================== TRAIN =====================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# ===================== EVALUATE =====================
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ===================== SAVE TRAINING HISTORY =====================
history_df = pd.DataFrame(history.history)
history_df['test_loss'] = test_loss
history_df['test_accuracy'] = test_accuracy

history_filename = f"training_history_VGG19_{num_classes}classes.xlsx"
history_df.to_excel(history_filename, index=False)
print(f"Training history saved to '{history_filename}'")

# ===================== PREDICTIONS =====================
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

class_labels = list(test_generator.class_indices.keys())

test_results = []
for i, image_path in enumerate(test_generator.filepaths):
    image_name = os.path.basename(image_path)
    true_class = class_labels[test_generator.classes[i]]
    predicted_class = class_labels[predicted_classes[i]]

    test_results.append({
        "Image Name": image_name,
        "True Class": true_class,
        "Predicted Class": predicted_class,
        "Correct": "Yes" if true_class == predicted_class else "No"
    })

test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel("test_results_VGG19.xlsx", index=False)
print("Test results saved to 'test_results_VGG19.xlsx'")

# ===================== SAVE MODEL =====================
model.save("vgg19_model.h5")
print("Keras model saved as 'vgg19_model.h5'")

# ===================== TFLITE =====================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_filename = f"model_VGG19_{num_classes}classes.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as '{tflite_filename}'")

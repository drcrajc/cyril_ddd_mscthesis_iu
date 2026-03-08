import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# Paths to image folders
train_folder = "new_ddd_ds/train"
val_folder = "new_ddd_ds/valid"
test_folder = "new_ddd_ds/test"

# Number of classes in the dataset
num_classes = 2

# Input image dimensions
input_size = (224, 224)

# Training parameters
batch_size = 32
epochs = 50

# Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
validation_generator = val_test_datagen.flow_from_directory(
    val_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = val_test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load pre-trained ResNet50V2 model
base_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(input_size[0], input_size[1], 3),
    pooling=None,
    classifier_activation=None
)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save training history to DataFrame
history_df = pd.DataFrame(history.history)
history_df['test_loss'] = test_loss
history_df['test_accuracy'] = test_accuracy

# Predict on test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get class names
class_labels = list(test_generator.class_indices.keys())

# Match predictions to filenames
test_results = []
for i, image_path in enumerate(test_generator.filepaths):
    image_name = os.path.basename(image_path)
    true_class = class_labels[test_generator.labels[i]]
    predicted_class = class_labels[predicted_classes[i]]
    accuracy = 100 if true_class == predicted_class else 0
    test_results.append({
        "Image Name": image_name,
        "Predicted Class": predicted_class,
        "True Class": true_class,
        "Accuracy": f"{accuracy}%"
    })

# Save test results
test_results_df = pd.DataFrame(test_results)
test_results_df.to_excel("test_results_resnet50v2.xlsx", index=False)

# Save training history
history_filename = f"training_history_ResNet50V2_{num_classes}classes.xlsx"
history_df.to_excel(history_filename, index=False)
print(f"Training history saved to '{history_filename}'")

# Convert and save model as TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_filename = f"model_ResNet50V2_{num_classes}classes.tflite"
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved in TensorFlow Lite format: {tflite_model_filename}")

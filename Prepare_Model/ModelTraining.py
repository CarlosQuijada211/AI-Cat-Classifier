"""
This script trains a convolutional neural network (CNN) to classify cat images using transfer learning with MobileNetV2. 
It loads images from a directory, applies data augmentation and normalization, splits the data into training and validation sets, 
builds and compiles the model, trains it, and saves the trained model to disk.
"""

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from pathlib import Path
import keras

# Set data directory
data_dir = Path('data')
image_count = len(list(data_dir.glob('*/*.jpg')))
cleo = list(data_dir.glob('cleo/*.jpg'))

# Parameters
batch_size = 16
img_height = 224
img_width = 224

# Load datasets with an 80-20 train-validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'  # labels as integers: 0 or 1
    )  

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'  # labels as integers: 0 or 1
)

# Data augmentation and normalization
train_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),       # mirror images
    tf.keras.layers.RandomRotation(0.1),            # rotate up to ±10%
    tf.keras.layers.RandomZoom(0.1),                # zoom in/out 10%
    tf.keras.layers.RandomContrast(0.1),            # adjust contrast
    tf.keras.layers.RandomTranslation(0.1, 0.1)     # shift up/down, left/right
])

normalization_layer = layers.Rescaling(1./255)

# Apply data augmentation and normalization
train_ds = train_ds.map(lambda x, y: (train_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Build the model using transfer learning with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet',
)

base_model.trainable = False

# Add custom classification head
head = models.Sequential([
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax') 
])

# Combine base model and head
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
outputs = head(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',  # labels as integers: 0 or 1
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # start small, use EarlyStopping if needed
)

# Save the model
keras.saving.save_model(model, "cat_identifier_model.h5")  # TensorFlow SavedModel format


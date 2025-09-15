"""
This script loads a trained cat image classification model and applies it to each frame of a video. 
It preprocesses video frames, predicts class probabilities, overlays predictions on the frames, and saves the annotated video.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# Load the trained model
model = load_model('cat_identifier_model.h5')

# Parameters
batch_size = 16
img_height = 224
img_width = 224

class_names = ["Cleo", "Gataki"]

video_name = "placeholder"  # Replace with your video file name without extension

# Open video file
cap = cv2.VideoCapture(f"test_videos/{video_name}.mp4")

if not cap.isOpened():
    print("Error: No se pudo abrir el video de entrada")
    exit()

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"processed_videos/prediction_{video_name}.mp4", fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

font = cv2.FONT_HERSHEY_SIMPLEX

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    # Preprocess frame
    img = cv2.resize(frame, (img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=1)[0]

    # Draw probabilities on frame
    y0 = 30
    for i, prob in enumerate(predictions):
        text = f"{class_names[i]}: {prob:.2f}"
        cv2.putText(frame, text, (10, y0), font, 1.5, (0, 255, 0), 2)
        y0 += 50

    # Write to output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


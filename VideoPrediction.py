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

video_name = "both_separate_02"  # Replace with your video file name without extension

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

    # Draw probabilities on frame (top-left corner)
    y0 = 30
    for i, prob in enumerate(predictions):
        text = f"{class_names[i]}: {prob:.2f}"
        cv2.putText(frame, text, (10, y0), font, 1.0, (0, 255, 0), 2)
        y0 += 40

    # Find the predicted class
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    # Draw black strip at bottom
    strip_height = 60
    frame_height, frame_width = frame.shape[:2]
    cv2.rectangle(frame, (0, frame_height - strip_height), (frame_width, frame_height), (0, 0, 0), -1)

    # Write predicted class in white
    cv2.putText(frame, f"AI thinks: {predicted_class}", 
                (20, frame_height - 20), font, 1.5, (255, 255, 255), 3)

    # Write to output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


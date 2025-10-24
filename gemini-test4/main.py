
import cv2
import json
from deepface import DeepFace
import numpy as np
import os

# Load expression data from JSON
with open("expressions.json", "r") as f:
    expressions_data = json.load(f)

# Pre-load images into a cache
image_cache = {}
for expr, filenames in expressions_data.items():
    # Full path to the image
    image_path = os.path.join("images", filenames[0])
    # Read the image and store it in the cache
    image_cache[expr] = cv2.imread(image_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
# Frame counter
frame_counter = 0
# Detected expression
current_expression = "neutral"

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 10th frame
    if frame_counter % 10 == 0:
        try:
            # Analyze the frame for emotions
            analysis = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False
            )
            # Get the dominant emotion
            current_expression = analysis[0]["dominant_emotion"]
        except Exception as e:
            # Default to neutral on error
            current_expression = "neutral"

    # Get the corresponding image from the cache
    expression_image = image_cache.get(current_expression, image_cache["neutral"])

    # Create a combined view
    # Get dimensions for alignment
    cam_h, cam_w, _ = frame.shape
    expr_h, expr_w, _ = expression_image.shape

    # Create a black canvas
    canvas = np.zeros((max(cam_h, expr_h), cam_w + expr_w + 20, 3), dtype=np.uint8)

    # Place webcam feed on the left
    canvas[:cam_h, :cam_w] = frame

    # Place expression image on the right
    canvas[:expr_h, cam_w + 20 : cam_w + 20 + expr_w] = expression_image

    # Display the current expression text
    cv2.putText(
        canvas,
        f"Expression: {current_expression}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Show the combined view
    cv2.imshow("Facial Expression Recognition", canvas)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Increment frame counter
    frame_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

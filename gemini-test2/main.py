
import cv2
import json
from deepface import DeepFace
import numpy as np

# Load expression mappings from JSON
with open('expressions.json', 'r') as f:
    expressions = json.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Frame counter
frame_counter = 0

# Store the last detected expression
last_expression = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze every 10th frame
    if frame_counter % 10 == 0:
        try:
            # Detect emotion
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']

            # Update expression if it has changed
            if dominant_emotion != last_expression:
                last_expression = dominant_emotion
                print(f"Detected expression: {last_expression}")

        except Exception as e:
            print(f"Error analyzing frame: {e}")
            last_expression = "neutral" # Default to neutral on error

    # Get the corresponding image for the last detected expression
    if last_expression and last_expression in expressions:
        image_path = f"images/{expressions[last_expression][0].strip()}"
        expression_image = cv2.imread(image_path)
    else:
        # Default to neutral image if no expression is detected yet or expression is not in the mapping
        image_path = f"images/{expressions['neutral'][0].strip()}"
        expression_image = cv2.imread(image_path)


    # Resize webcam feed and expression image to have the same height
    cam_h, cam_w, _ = frame.shape
    expr_h, expr_w, _ = expression_image.shape
    
    # Set a fixed height for display
    display_h = 480
    
    # Calculate the new width to maintain aspect ratio
    cam_w = int(cam_w * (display_h / cam_h))
    expr_w = int(expr_w * (display_h / expr_h))

    # Resize the images
    frame_resized = cv2.resize(frame, (cam_w, display_h))
    expression_image_resized = cv2.resize(expression_image, (expr_w, display_h))

    # Combine the two images side-by-side
    combined_view = np.hstack((frame_resized, expression_image_resized))

    # Display the combined view
    cv2.imshow("Facial Expression Recognition", combined_view)

    frame_counter += 1

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

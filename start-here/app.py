
import cv2
import json
import numpy as np
from deepface import DeepFace
import os

# Load expression mappings from JSON
with open("expressions.json", "r") as f:
    expressions = json.load(f)

# Cache for loaded images
image_cache = {}

def get_expression_image(expression):
    """Gets the image for a given expression, loading and caching it if necessary."""
    if expression not in image_cache:
        image_path = os.path.join("images", expressions.get(expression, expressions["neutral"])[0])
        image_cache[expression] = cv2.imread(image_path)
    return image_cache[expression]

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    current_expression = "neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame for performance
        if frame_count % 10 == 0:
            try:
                # Analyze frame for facial expression
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                # Check if results is a list and not empty
                if isinstance(results, list) and len(results) > 0:
                    # Extract the dominant emotion from the first result
                    dominant_emotion = results[0]['dominant_emotion']
                    if dominant_emotion in expressions:
                        current_expression = dominant_emotion
                    else:
                        current_expression = "neutral"
                else:
                    # Handle cases where no face is detected or results are not as expected
                    current_expression = "neutral"
            except Exception as e:
                print(f"Error analyzing frame: {e}")
                current_expression = "neutral"

        # Get the corresponding character image
        expression_image = get_expression_image(current_expression)

        # Resize webcam frame and expression image to have the same height
        cam_h, cam_w, _ = frame.shape
        expr_h, expr_w, _ = expression_image.shape
        new_height = 480
        new_cam_w = int(cam_w * (new_height / cam_h))
        new_expr_w = int(expr_w * (new_height / expr_h))

        resized_frame = cv2.resize(frame, (new_cam_w, new_height))
        resized_expression_image = cv2.resize(expression_image, (new_expr_w, new_height))

        # Create a combined image
        combined_image = np.hstack([resized_frame, resized_expression_image])

        # Display the current expression name
        cv2.putText(combined_image, f"Expression: {current_expression}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("Facial Expression Recognition", combined_image)

        frame_count += 1

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

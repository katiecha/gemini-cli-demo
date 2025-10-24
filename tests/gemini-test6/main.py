
import cv2
import json
from deepface import DeepFace
import numpy as np
import os

# Load expression mappings
with open('expressions.json', 'r') as f:
    expressions = json.load(f)

# Cache for loaded images
image_cache = {}

def get_expression_image(expression):
    """Gets the image for a given expression, loading and caching it if necessary."""
    if expression not in image_cache:
        image_path = os.path.join('images', expressions.get(expression, expressions['neutral'])[0])
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
    expression_image = get_expression_image(current_expression)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            try:
                # Analyze frame for emotion
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(results, list) and results:
                    dominant_emotion = results[0]['dominant_emotion']
                    if dominant_emotion in expressions:
                        current_expression = dominant_emotion
                    else:
                        current_expression = "neutral"
                    expression_image = get_expression_image(current_expression)

            except Exception as e:
                print(f"Error analyzing frame: {e}")
                current_expression = "neutral"
                expression_image = get_expression_image(current_expression)


        # Resize expression image to match webcam feed height
        cam_h, cam_w, _ = frame.shape
        img_h, img_w, _ = expression_image.shape
        scale = cam_h / img_h
        resized_img = cv2.resize(expression_image, (int(img_w * scale), cam_h))

        # Create side-by-side view
        combined_view = np.hstack((frame, resized_img))

        # Display current expression
        cv2.putText(combined_view, f"Expression: {current_expression}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the combined view
        cv2.imshow('Facial Expression Recognition', combined_view)

        # Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

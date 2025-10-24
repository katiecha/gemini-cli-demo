
import cv2
import json
from deepface import DeepFace
import numpy as np

# Load expression mappings from JSON
with open('expressions.json', 'r') as f:
    expression_images = json.load(f)

# Cache for loaded images
image_cache = {}

def get_expression_image(expression):
    """Loads an image from the cache or reads it from disk."""
    if expression in image_cache:
        return image_cache[expression]
    else:
        image_filename = expression_images[expression]
        if isinstance(image_filename, list):
            image_filename = image_filename[0]
        image_path = f"images/{image_filename}"
        image = cv2.imread(image_path)
        image_cache[expression] = image
        return image

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
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame
        if frame_count % 10 == 0:
            try:
                # Analyze face for emotion
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # Check if results is a list and not empty
                if isinstance(results, list) and len(results) > 0:
                    dominant_emotion = results[0]['dominant_emotion']
                    if dominant_emotion in expression_images:
                        current_expression = dominant_emotion
                        expression_image = get_expression_image(current_expression)

            except Exception as e:
                print(f"Error analyzing frame: {e}")

        # Create a combined view
        cam_h, cam_w, _ = frame.shape
        img_h, img_w, _ = expression_image.shape
        
        # Resize expression image to match webcam feed height
        scale_factor = cam_h / img_h
        resized_img_w = int(img_w * scale_factor)
        resized_expression_image = cv2.resize(expression_image, (resized_img_w, cam_h))

        # Combine webcam feed and expression image
        combined_view = np.hstack((frame, resized_expression_image))
        
        # Display the resulting frame
        cv2.imshow('Facial Expression Recognition', combined_view)

        frame_count += 1

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

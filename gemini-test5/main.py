
import cv2
import json
import os
import numpy as np
from deepface import DeepFace

# --- Constants and Configuration ---
EXPRESSIONS_FILE = "expressions.json"
IMAGES_DIR = "images"
FRAME_ANALYSIS_INTERVAL = 10  # Analyze every 10 frames
WINDOW_NAME = "Facial Expression Recognition"
QUIT_KEY = "q"

# --- Helper Functions ---

def load_expressions(file_path):
    """Loads expression-to-image mappings from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}

def preload_images(expressions, base_dir):
    """Pre-loads and caches images for each expression."""
    image_cache = {}
    for expr, filenames in expressions.items():
        if filenames:
            image_path = os.path.join(base_dir, filenames[0])
            if os.path.exists(image_path):
                image_cache[expr] = cv2.imread(image_path)
            else:
                print(f"Warning: Image not found for expression '{expr}': {image_path}")
    return image_cache

def get_expression_image(expression, image_cache):
    """Gets the cached image for an expression, defaulting to neutral."""
    img = image_cache.get(expression)
    if img is not None:
        return img
    return image_cache.get("neutral")

def draw_text(image, text, position=(50, 50), font_scale=1.5, color=(255, 255, 255), thickness=2):
    """Draws text on an image with a shadow for better visibility."""
    # Shadow
    cv2.putText(image, text, (position[0] + 2, position[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


# --- Main Application Logic ---

def main():
    """Main function to run the facial expression recognition application."""
    expressions = load_expressions(EXPRESSIONS_FILE)
    if not expressions:
        return

    image_cache = preload_images(expressions, IMAGES_DIR)
    if not image_cache:
        print("Error: No images were loaded. Exiting.")
        return

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

        frame_height, frame_width, _ = frame.shape
        frame_for_analysis = frame.copy()

        if frame_count % FRAME_ANALYSIS_INTERVAL == 0:
            try:
                results = DeepFace.analyze(
                    frame_for_analysis,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(results, list) and results:
                    dominant_emotion = results[0]["dominant_emotion"]
                    if dominant_emotion in expressions:
                        current_expression = dominant_emotion
                    else:
                        current_expression = "neutral" # Default for unknown expressions
            except Exception as e:
                # If face is not detected or another error occurs, keep the last expression
                pass

        expression_image = get_expression_image(current_expression, image_cache)
        if expression_image is not None:
            # Resize expression image to match webcam frame height
            target_height = frame_height
            aspect_ratio = expression_image.shape[1] / expression_image.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized_expression_image = cv2.resize(expression_image, (target_width, target_height))

            # Create a combined view
            combined_view = np.hstack((frame, resized_expression_image))
            
            # Display the current expression on the combined view
            draw_text(combined_view, f"Expression: {current_expression.capitalize()}", position=(30, frame_height - 40))

        else:
            # Fallback if for some reason the image is not available
            combined_view = frame
            draw_text(combined_view, "Expression image not found", position=(30, frame_height - 40), color=(0, 0, 255))


        cv2.imshow(WINDOW_NAME, combined_view)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

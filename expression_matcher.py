import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
import json

class ExpressionMatcher:
    def __init__(self, expression_mapping_file='expressions.json', image_directory='images'):
        self.mapping_file = Path(expression_mapping_file)
        self.image_directory = Path(image_directory)
        self.expression_images = self.load_expression_mapping()
        self.cap = cv2.VideoCapture(0)
        self.image_cache = {}
        self.current_expression = "neutral"
        self.frame_count = 0

    def load_expression_mapping(self):
        if not self.mapping_file.exists():
            default_mapping = {"angry": [], "happy": [], "sad": [], "neutral": []}

            if self.image_directory.exists():
                images = [f.name for f in self.image_directory.iterdir()
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
                if images:
                    default_mapping["neutral"] = images

            with open(self.mapping_file, 'w') as f:
                json.dump(default_mapping, f, indent=2)

            return default_mapping

        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def get_image_for_expression(self, expression):
        expression = expression.lower()

        # Find image name for expression
        image_name = None
        if expression in self.expression_images and self.expression_images[expression]:
            image_name = self.expression_images[expression][0]
        elif self.expression_images.get("neutral"):
            image_name = self.expression_images["neutral"][0]
        else:
            for images in self.expression_images.values():
                if images:
                    image_name = images[0]
                    break

        if not image_name:
            return None

        # Load from cache or disk
        if image_name not in self.image_cache:
            img = cv2.imread(str(self.image_directory / image_name))
            if img is not None:
                self.image_cache[image_name] = img

        return self.image_cache.get(image_name)

    def detect_expression(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                     enforce_detection=False, detector_backend='opencv')
            if isinstance(result, list):
                result = result[0]
            return result['dominant_emotion']
        except:
            return self.current_expression

    def run(self):
        print("Expression Matcher started! Press 's' to save, 'q' to quit\n")
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect expression every 10 frames
            if self.frame_count % 10 == 0:
                self.current_expression = self.detect_expression(frame)
            self.frame_count += 1

            # Get character image
            char_img = self.get_image_for_expression(self.current_expression)
            h, w = frame.shape[:2]

            # Prepare character image or placeholder
            if char_img is not None:
                char_h, char_w = h, int(h * char_img.shape[1] / char_img.shape[0])
                char_resized = cv2.resize(char_img, (char_w, char_h))
            else:
                char_h, char_w = h, int(h * 0.75)
                char_resized = np.full((char_h, char_w, 3), (220, 220, 220), dtype=np.uint8)
                cv2.putText(char_resized, f"No image", (char_w//4, char_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            # Create display with bottom bar
            bar_height = 50
            display_frame = np.zeros((h + bar_height, w + char_w, 3), dtype=np.uint8)
            display_frame[0:h, 0:w] = frame
            display_frame[0:h, w:w + char_w] = char_resized
            cv2.putText(display_frame, f"Expression: {self.current_expression.capitalize()}",
                       (20, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Expression Matcher', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                Path('output').mkdir(exist_ok=True)
                filename = f'output/expression_{saved_count:03d}.png'
                cv2.imwrite(filename, display_frame)
                print(f"Saved: {filename}")
                saved_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    image_dir = Path('images')

    if not image_dir.exists():
        print(f"Error: Image directory 'images' not found!")
        print(f"Please create it and add your character images.")
        return

    images = [f for f in image_dir.iterdir()
             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    if not images:
        print(f"Error: No images found in 'images' directory!")
        return

    print(f"Found {len(images)} image(s) in 'images' directory\n")

    matcher = ExpressionMatcher()
    matcher.run()


if __name__ == "__main__":
    main()

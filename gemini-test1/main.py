
import cv2
import json
from deepface import DeepFace
import numpy as np

# Load expression mappings from JSON file
with open('expressions.json', 'r') as f:
    expression_images = json.load(f)

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Set the window to a normal, resizable window
cv2.namedWindow('Facial Expression Recognition', cv2.WINDOW_NORMAL)

current_expression = "neutral"
expression_img = cv2.imread(f'images/{expression_images[current_expression][0]}')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Analyze the face for emotion
        try:
            results = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']

            allowed_expressions = ["happy", "sad", "neutral", "angry"]
            if dominant_emotion in allowed_expressions and dominant_emotion in expression_images:
                current_expression = dominant_emotion
                expression_img = cv2.imread(f'images/{expression_images[current_expression][0]}')

        except Exception as e:
            print(f"Error analyzing face: {e}")

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the detected expression
        cv2.putText(frame, current_expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Resize expression image to match webcam feed height
    if expression_img is not None:
      resized_expression_img = cv2.resize(expression_img, (frame.shape[1], frame.shape[0]))
      # Combine webcam feed and expression image side-by-side
      combined_frame = np.hstack((frame, resized_expression_img))
    else:
      combined_frame = frame


    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

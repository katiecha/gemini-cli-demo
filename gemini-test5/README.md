# Real-Time Facial Expression Recognition

This application uses your webcam to detect facial expressions (happy, sad, neutral) in real-time and displays a corresponding character image alongside the video feed.

## Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   OpenCV dependencies. You can often install these with your system's package manager. For example, on Ubuntu/Debian:
        ```bash
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx
        ```

2.  **Clone the repository or download the files.**

3.  **Install Python dependencies:**
    Navigate to the project directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Make sure you have the `images` folder with `happy.png`, `sad.png`, and `neutral.png`, and the `expressions.json` file in the same directory as the script.

2.  Run the application from your terminal:
    ```bash
    python main.py
    ```

3.  A window will appear showing your webcam feed and the character image.

4.  Press the 'q' key to quit the application.

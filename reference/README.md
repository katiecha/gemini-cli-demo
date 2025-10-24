# Expression Matcher

A real-time facial expression recognition application that matches your expressions to character images using DeepFace and OpenCV.

## Overview

Expression Matcher uses your webcam to detect facial expressions in real-time and displays corresponding character images side-by-side. The application uses DeepFace's emotion detection to recognize expressions like happy, sad, angry, neutral, and more.

## Installation

### Prerequisites
- Python 3.11 or higher
- Webcam access
- macOS, Windows, or Linux

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gemini-cli-demo
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create an `images` directory and add your character images:
```bash
mkdir images
# Add your .png, .jpg, or .jpeg files to the images/ directory
```

5. Run the application:
```bash
python expression_matcher.py
```

### Configuration

The application will automatically create an `expressions.json` file on first run. Edit this file to map specific character images to different expressions:

```json
{
  "angry": ["angry_character.png"],
  "happy": ["happy_character.png"],
  "sad": ["sad_character.png"],
  "neutral": ["neutral_character.png"]
}
```

### Usage

- The application will display your webcam feed alongside the matched character image
- Press `s` to save a screenshot
- Press `q` to quit

## COMP 523 Demo Project

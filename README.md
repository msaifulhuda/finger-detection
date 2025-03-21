﻿# Finger Detection Project

This project is a simple Python application that captures video from a webcam, detects hand landmarks, and displays icons above raised fingers. The project uses OpenCV for video processing and MediaPipe for hand landmark detection.

## Features

- Captures video from the webcam.
- Detects hand landmarks using MediaPipe.
- Displays icons above raised index, middle, and ring fingers.
- Custom window icon for the application.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- pywin32

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/msaifulhuda/finger-detection.git
    cd finger-detection
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe pywin32
    ```

3. Place your icon images (`i.png`, `love.png`, `u.png`) in the `./icon/` directory.

## Usage

Run the main script to start the application:
```sh
python main.py
```

## Code Overview

The main functionality of the project is implemented in the `main.py` script. Here is a brief overview of the process:

1. Open video capture from the webcam.
2. Read frames from the video capture.
3. Flip the frame horizontally.
4. Convert the frame from BGR to RGB.
5. Process the frame to detect hand landmarks.
6. If hand landmarks are detected:
    - Draw hand landmarks on the frame.
    - Calculate the positions of the fingertips and joints for the index, middle, and ring fingers.
    - Determine if the index, middle, or ring finger is raised.
    - Display an icon above the raised finger.
7. Display the frame with icons in the 'Finger Icons' window.
8. Set the window icon.
9. Close the video capture and window when the 'Esc' key is pressed or the window is closed.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [pywin32](https://github.com/mhammond/pywin32)

Feel free to contribute to this project by submitting issues or pull requests.

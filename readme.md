# Perspective Transform Experiment

This project demonstrates the process of performing a perspective transform on images using Python. The main objective is to capture images, calibrate the camera using a chessboard pattern, and then apply perspective transformations to correct or manipulate the image view.

## Files in the Repository:

- **calibration_finder.py**: Detects and computes camera calibration from chessboard images.
- **calibration_verifier.py**: Verifies the camera calibration by applying the found parameters.
- **image_capture.py**: Captures images using a camera for calibration and transformation.
- **perspective_transform.py**: Implements the perspective transformation logic.
- **frames/**: Folder containing sample images.
- **camera_calibration.npz**: Saved calibration parameters.

## Requirements:
- Python 3.x
- OpenCV
- NumPy

## Usage:
1. Use `image_capture.py` to capture images of a chessboard pattern for calibration.
2. Run `calibration_finder.py` to find camera calibration.
3. Apply `perspective_transform.py` for perspective manipulation.
    ![Original Image](chessboard0.jpg)
    ![Transformed Image](transformed_image.jpg)


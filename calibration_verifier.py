import numpy as np
import cv2
import glob

# Load the stored calibration data
calibration_data = np.load('camera_calibration.npz')
mtx = calibration_data['mtx']  # Intrinsic matrix
dist = calibration_data['dist']  # Distortion coefficients
rvecs = calibration_data['rvecs']  # Rotation vectors
tvecs = calibration_data['tvecs']  # Translation vectors

# Load the chessboard images for verification
chessboard_size = (9, 6)  # Same dimensions used in calibration
square_size = 20.0  # Same square size in millimeters
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

images = glob.glob('frames/*.jpg')  # Replace with your image path

# Calculate reprojection error
total_error = 0
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Keep track of the index for rvecs and tvecs
used_images = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Project the object points back to image points using the calibration parameters
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[used_images], tvecs[used_images], mtx, dist)
        error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

        used_images += 1  # Increment the index for rvecs and tvecs

# Print mean reprojection error
if used_images > 0:
    mean_error = total_error / used_images
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
else:
    print("No images were used for reprojection error calculation.")

# Perform undistortion on a sample image
sample_image = cv2.imread(images[0])
h, w = sample_image.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(sample_image, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# Show and save the undistorted image
cv2.imshow('Original Image', sample_image)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.imwrite('undistorted_image.jpg', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

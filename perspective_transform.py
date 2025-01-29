import cv2
import numpy as np

def get_transformed_size(matrix, width, height):
    """
    Calculate the size needed for the output image to contain all transformed pixels
    """
    # Create array of corners points
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T

    # Transform corners
    transformed_corners = matrix @ corners
    transformed_corners = transformed_corners / transformed_corners[2]

    # Get the min and max points
    min_x = np.min(transformed_corners[0])
    max_x = np.max(transformed_corners[0])
    min_y = np.min(transformed_corners[1])
    max_y = np.max(transformed_corners[1])

    return (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))

def adjust_matrix_for_offset(matrix, min_x, min_y):
    """
    Adjust transformation matrix to account for negative coordinates
    """
    translation_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    return translation_matrix @ matrix

# Load the chessboard images
image1 = cv2.imread('chessboard0.jpg')
image2 = cv2.imread('chessboard1.jpg')

# Define the chessboard size (inner corners)
chessboard_size = (9, 6)

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners in both images
ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

if ret1 and ret2:
    # Refine corner locations for better accuracy
    corners1 = cv2.cornerSubPix(
        gray1, corners1, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    corners2 = cv2.cornerSubPix(
        gray2, corners2, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Define the source points as the four outermost corners of the first chessboard
    source_points = np.float32([
        corners1[0][0],                        # Top-left
        corners1[chessboard_size[0] - 1][0],   # Top-right
        corners1[-chessboard_size[0]][0],      # Bottom-left
        corners1[-1][0]                        # Bottom-right
    ])

    # Define the destination points as the four outermost corners of the second chessboard
    destination_points = np.float32([
        corners2[0][0],                        # Top-left
        corners2[chessboard_size[0] - 1][0],   # Top-right
        corners2[-chessboard_size[0]][0],      # Bottom-left
        corners2[-1][0]                        # Bottom-right
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    # Calculate required size for output image
    h, w = image1.shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T

    # Transform corners to find required image size
    transformed_corners = matrix @ corners
    transformed_corners = transformed_corners / transformed_corners[2]

    # Find the bounds of the transformed image
    min_x = np.floor(transformed_corners[0].min()).astype(int)
    max_x = np.ceil(transformed_corners[0].max()).astype(int)
    min_y = np.floor(transformed_corners[1].min()).astype(int)
    max_y = np.ceil(transformed_corners[1].max()).astype(int)

    # Calculate output size
    output_width = max_x - min_x
    output_height = max_y - min_y

    # Adjust transformation matrix for offset
    adjusted_matrix = adjust_matrix_for_offset(matrix, min_x, min_y)

    # Apply the perspective transformation with the new dimensions
    warped_image = cv2.warpPerspective(
        image1, 
        adjusted_matrix, 
        (output_width, output_height)
    )

    # Display the original and warped images
    cv2.imshow('Original Image 1', image1)
    cv2.imshow('Original Image 2', image2)
    cv2.imshow('Transformed Image 1', warped_image)

    # Save the output
    cv2.imwrite('transformed_image.jpg', warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Chessboard corners not found in one or both images! Ensure the images are clear and the chessboards are fully visible.")
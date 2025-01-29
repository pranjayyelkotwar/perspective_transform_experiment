# write a script that will start a opencv video capture and then whenever user presses 's' it will save the current frame to disk
# whenever user presses 'q' it will quit the video capture

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("chessboard%d.jpg" % count, frame)
        print("Frame saved as frame%d.jpg" % count)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


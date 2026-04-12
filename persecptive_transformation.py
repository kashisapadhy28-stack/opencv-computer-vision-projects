import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)
# cv.VideoCapture(index) → opens a video stream.
# 0 → default webcam.
# cap → object representing the video capture.
# Syntax:
# cap = cv.VideoCapture(0)    # 0 for webcam, or file path

while True:   #Starts an infinite loop to continuously read frames from the webcam.
    _,frame=cap.read()
#     cap.read() → reads a frame from the video stream.
# Returns two values:
# _ → Boolean (True if frame is read correctly).
# frame → the current video frame.
# Syntax:
# ret, frame = cap.read()
    cv.circle(frame,(155,120),5,(0,0,255),-1)
    cv.circle(frame,(480,120),5,(0,0,255),-1)
    cv.circle(frame,(20,475),5,(0,0,255),-1)
    cv.circle(frame,(620,475),5,(0,0,255),-1)
#syntax:
# cv.circle(frame, (x, y), radius, (B,G,R), thickness)
# Draws circles on the frame at specified points.
# (155,120), (480,120), (20,475), (620,475) → coordinates of four points.
# 5 → radius of circle.
# (0,0,255) → color (red in BGR format).
# -1 → filled circle.
# Purpose: Highlight points used for perspective transformation.
    pts1=np.float32([[155,120],[480,120],[20,475],[620,475]])
#     np.float32() → converts list of points to float32 type (required by OpenCV).
# pts1 → source points in original frame (corners of the quadrilateral).
    pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
# pts2 → destination points in the output image (rectangle).
# Maps quadrilateral in original frame to rectangle of size 400×600.
    matrix=cv.getPerspectiveTransform(pts1,pts2)
#     cv.getPerspectiveTransform(src, dst) → calculates 3x3 transformation matrix to map points from pts1 to pts2.
# matrix → the perspective transformation matrix.

# Syntax:

# matrix = cv.getPerspectiveTransform(src_points, dst_points)
    result=cv.warpPerspective(frame,matrix,(400,600))
#     cv.warpPerspective(src, matrix, dsize) → applies perspective transformation.
# frame → source image.
# matrix → transformation matrix from previous step.
# (400,600) → size of output image (width × height).
# result → transformed (bird’s eye view) image.
# Syntax:
# dst = cv.warpPerspective(src, matrix, (width, height))
    cv.imshow("frame",frame)
    cv.imshow("perspective transformation",result)
    key=cv.waitKey(1)
    if key==27:
        break
cap.release()
cv.destroyAllWindows()


# Summary

# This code performs real-time perspective transformation on a webcam video:

# Open webcam and read frames continuously.
# Draw 4 red points (corners of a quadrilateral to transform).
# Define source points (pts1) and destination points (pts2).
# Compute perspective transform matrix.
# Apply warpPerspective to get a bird’s-eye view rectangle.
# Show both original and transformed frames.
# Stop when Esc key is pressed.

# Key Concepts:

# Perspective transformation → changes viewpoint of an image.
# cv.getPerspectiveTransform → computes mapping from quadrilateral → rectangle.
# cv.warpPerspective → applies the transformation.
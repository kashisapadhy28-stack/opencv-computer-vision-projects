import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)
template=cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
# Reads template image in grayscale.
# cv.IMREAD_GRAYSCALE → converts the image to grayscale on load.
# template → stores the image used for matching.
w,h=template.shape[::-1]
# template.shape → returns (height, width) for grayscale image.
# [::-1] → reverses the order → (width, height)
# w, h → store width and height of template for drawing rectangles.
while True:
    # Starts an infinite loop to continuously read frames from the webcam.
    _,frame=cap.read()
#     Reads a single frame from webcam.
# _ → boolean indicating success of frame read.
# frame → current color frame from webcam.
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     Converts the current frame to grayscale.
# cv.COLOR_BGR2GRAY → conversion code from color (BGR) to grayscale.
    result=cv.matchTemplate(gray_frame,template,cv.TM_CCOEFF_NORMED)
#     Template matching: slides template over frame and computes similarity.
# cv.TM_CCOEFF_NORMED → method returning normalized correlation (-1 to 1).
# result → 2D array of similarity scores for each position.
# Syntax:
# result = cv.matchTemplate(image, template, method)
    loc=np.where(result>=0.6)
#     Finds coordinates where match score is ≥ 0.6.
# loc → array of (row, col) indices where template matches.
    for pt in zip(*loc[::-1]):
        # loc[::-1] → reverse indices to (x, y) format.
# *loc[::-1] → unpack arrays for zip.
# Loops through all detected match points.
# pt → top-left coordinate of detected template.
        cv.rectangle(frame,pt,(pt[0]+w,pt[1]+h),(0,255,0),3)
#         Draws green rectangle around matched area.
# pt → top-left corner.
# (pt[0]+w, pt[1]+h) → bottom-right corner using template size.
# (0,255,0) → green color in BGR.
# 3 → thickness of rectangle.
# Syntax:
# cv.rectangle(image, (x1, y1), (x2, y2), (B,G,R), thickness)
    cv.imshow("frame",frame)
    key=cv.waitKey(1)
    if key==27:
        break
cap.release()
cv.destroyAllWindows()


#Summary

# This code performs real-time template matching on webcam video:

# Opens webcam and reads frames continuously.
# Converts each frame to grayscale.
# Computes similarity of frame with template image using cv.matchTemplate.
# Detects points where template matches above threshold (0.6).
# Draws green rectangles around detected areas.
# Displays live webcam video with detected templates.
# Stops when Esc key is pressed.

# Key Concepts:

# Template matching (cv.matchTemplate) finds objects in images.
# Real-time processing using webcam frames.
# Drawing rectangles highlights matches on video feed.
#OBJECT DETECTION USING HSV COLOR SPACE


import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0)

# Create a window for trackbars
#cv.namedWindow(window_name)
# cv.resizeWindow(window_name, width, height)
cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 600, 200)

#trackbar callback function, does nothing but required for createTrackbar
def nothing(x):
    pass

# Create trackbars for HSV ranges
#syntax: cv.createTrackbar(trackbar_name, window_name, default_value, max_value, callback_function)
cv.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()   #frame read karna 
    if not ret:

        break

    # Convert frame(bgr) to HSV
    #syntax=converted = cv.cvtColor(src, conversion_code)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    #syntax=value = cv.getTrackbarPos(trackbar_name, window_name)
    L_H = cv.getTrackbarPos("L-H", "Trackbars")
    L_S = cv.getTrackbarPos("L-S", "Trackbars")
    L_V = cv.getTrackbarPos("L-V", "Trackbars")
    U_H = cv.getTrackbarPos("U-H", "Trackbars")
    U_S = cv.getTrackbarPos("U-S", "Trackbars")
    U_V = cv.getTrackbarPos("U-V", "Trackbars")

#Lower & Upper Range Define
#syntax=array = np.array([value1, value2, value3])
    lower_color = np.array([L_H, L_S, L_V])
    upper_color = np.array([U_H, U_S, U_V])

    # Create mask
    #syntax-mask = cv.inRange(src, lowerb, upperb)
    mask = cv.inRange(hsv, lower_color, upper_color)

    #bitwise and operation
    #syntax=result = cv.bitwise_and(src1, src2, mask=mask)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Show frames
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    key = cv.waitKey(1)
    if key == 27:  # ESC key to break
        break

# Release and destroy windows
cap.release()
cv.destroyAllWindows()

#summary
#Ye program webcam se live video capture karta hai. Frame ko HSV color space me convert karta hai. Trackbars ki madad se HSV range dynamically adjust ki ja sakti hai. inRange() function selected color ka mask create karta hai aur bitwise_and() function se sirf wahi color highlight hota hai.
import cv2 as cv
import numpy as np
img=cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
print(img.shape)
# Prints the dimensions of the image.
# Output: (height, width) for grayscale.
print(img[0,0])
# Prints pixel value at top-left corner (row=0, col=0).
# For grayscale, value ranges 0–255.
_,threshold_binary=cv.threshold(img,128,255,cv.THRESH_BINARY)
# cv.threshold(src, thresh, maxval, type) → applies thresholding to an image.
# src → input grayscale image.
# thresh=128 → threshold value.
# maxval=255 → pixel value for pixels above threshold.
# type=cv.THRESH_BINARY → binary threshold:
# pixel ≥ 128 → 255 (white)
# pixel < 128 → 0 (black)
# _ → returns threshold used (not needed here).
# threshold_binary → binary image after thresholding.
# Syntax:
# ret, binary_img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
_,threshold_binary_inv=cv.threshold(img,128,255,cv.THRESH_BINARY)
cv.imshow("image",img)
cv.imshow("binary",threshold_binary)
cv.imshow("inv.binary",threshold_binary_inv)
cv.waitKey(0)
cv.destroyAllWindows()     


#sumary
# This code demonstrates image thresholding:

# Reads grayscale image.
# Prints image shape and top-left pixel value.
# Converts image to binary (pixels above threshold → white, below → black).
# (Optional) Inverse binary threshold for comparison.
# Displays original, binary, and inverse images.

# Purpose: Thresholding is used for image segmentation, extracting shapes, and preparing images for contour or object detection.
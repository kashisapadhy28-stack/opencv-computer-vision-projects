import cv2 as cv
import numpy as np

# img=cv.imread("text.jpg",cv.IMREAD_GRAYSCALE)#####
img = cv.imread("logo.png", cv.IMREAD_GRAYSCALE)

sobelx=cv.Sobel(img,cv.CV_64F,1,0)
# cv.Sobel(src, ddepth, dx, dy) → applies Sobel filter to detect edges.
# img → source grayscale image.
# cv.CV_64F → depth of output image (64-bit float, to avoid overflow).
# dx=1, dy=0 → derivative order in x and y directions: detects vertical edges.
# sobelx → image highlighting vertical edges.
# Syntax:
# dst = cv.Sobel(src, ddepth, dx, dy)
sobely=cv.Sobel(img,cv.CV_64F,0,1)
# Same as SobelX, but dx=0, dy=1 → detects horizontal edges.
# sobely → horizontal edges image.
laplacian=cv.Laplacian(img,cv.CV_64F,ksize=5)
# cv.Laplacian(src, ddepth, ksize) → second-order derivative for edge detection.
# ksize=5 → size of the kernel (larger → stronger smoothing).
# Detects edges in all directions.
# laplacian → edge map using Laplacian method.
# Syntax:
# dst = cv.Laplacian(src, ddepth, ksize)
canny=cv.Canny(img,100,150)
# cv.Canny(image, threshold1, threshold2) → Canny edge detector.
# threshold1 = 100, threshold2 = 150 → lower and upper thresholds for edge detection.
# canny → binary image highlighting edges.
# Syntax:
# edges = cv.Canny(image, threshold1, threshold2)
cv.imshow("img",img)
cv.imshow("sobelx",sobelx)
cv.imshow("sobely",sobely)
cv.imshow("laplacian",laplacian)
cv.imshow("canny",canny)
cv.waitKey(0)
cv.destroyAllWindows()


#Summary

# This code performs edge detection on a grayscale image using multiple methods:

# Method	Function / Effect
# Sobel X	Detects vertical edges.
# Sobel Y	Detects horizontal edges.
# Laplacian	Detects edges in all directions using second derivative.
# Canny	Detects strong edges using multi-stage algorithm, produces clean binary edges.
# Edge detection is a fundamental step in image analysis, object detection, and computer vision.
# The code displays all methods side by side for comparison.
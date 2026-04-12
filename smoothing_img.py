import cv2 as cv
import numpy as np
img=cv.imread("test.jpg")
averaging=cv.blur(img,(5,5))
# cv.blur(src, ksize) → applies averaging filter (mean filter) to blur the image.
# img → source image.
# (5,5) → kernel size (5×5 pixels).
# averaging → resulting blurred image.
# Effect: Smoothens image by replacing each pixel with the average of its neighbors.
# Syntax:
# dst = cv.blur(src, (kernel_width, kernel_height))
gaussian=cv.GaussianBlur(img,(5,5),0)
# cv.GaussianBlur(src, ksize, sigmaX) → applies Gaussian filter.
# (5,5) → kernel size.
# 0 → automatically calculate sigma (spread of Gaussian).
# gaussian → blurred image using Gaussian weights (more weight to center pixels).
# Effect: Reduces noise while keeping edges slightly sharper than averaging.
# Syntax:
# dst = cv.GaussianBlur(src, (kernel_width, kernel_height), sigmaX)
median=cv.medianBlur(img,5)
# cv.medianBlur(src, ksize) → applies median filter.
# ksize=5 → kernel size (must be odd).
# median → blurred image.
# Effect: Each pixel is replaced with the median of its neighbors → excellent for removing salt-and-pepper noise.
# Syntax:
# dst = cv.medianBlur(src, ksize)
bilateral=cv.bilateralFilter(img,9,75,75)
# cv.bilateralFilter(src, d, sigmaColor, sigmaSpace) → edge-preserving smoothing filter.
# d=9 → diameter of pixel neighborhood.
# sigmaColor=75 → filter strength for pixel value differences.
# sigmaSpace=75 → filter strength for spatial distances.
# bilateral → smoothed image while keeping edges sharp.
# Effect: Removes noise but preserves edges better than other filters.
# Syntax:
# dst = cv.bilateralFilter(src, d, sigmaColor, sigmaSpace)
cv.imshow("Original",img)
cv.imshow("averaging image",averaging)
cv.imshow("gaussian",gaussian)
cv.imshow("median",median)
cv.imshow("bilateral",bilateral)
cv.waitKey(0)
cv.destroyAllWindows()


#SUMMARY:
# This code demonstrates different image smoothing techniques:

# Filter Type	Function / Effect
# Averaging	Replaces each pixel with the mean of neighbors; simple blur.
# Gaussian	Weighted mean with Gaussian; reduces noise with slightly sharper edges.
# Median	Replaces pixel with median of neighbors; great for salt-and-pepper noise.
# Bilateral	Smooths image but preserves edges; good for artistic or denoising effects.
# All filters are applied to the same input image.
# The results are displayed in separate windows for comparison.
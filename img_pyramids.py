import cv2 as cv
import numpy as np
img=cv.imread("test.jpg")
layers=img.copy()
# Makes a copy of the original image.
# This copy will be used to build the pyramid.
gaussian_pyramid=[layers]
# Initializes a list to store Gaussian pyramid layers.
# Starts with the original image as the first level.
for i in range(6):
    layers=cv.pyrDown(layers)
    gaussian_pyramid.append(layers)
# for i in range(6): → loop 6 times to create 6 levels of Gaussian pyramid.
# cv.pyrDown(layers) → reduces image size by half (both width & height) using Gaussian blur.
# gaussian_pyramid.append(layers) → add the reduced image to the pyramid list.
# Syntax:
# smaller_image = cv.pyrDown(src_image)
# Effect: Creates progressively smaller and blurred versions of the original image.
layers=gaussian_pyramid[5]
# Takes the smallest image from Gaussian pyramid (6th level).
cv.imshow("6",layers)
#Displays the 6th level of Gaussian pyramid (smallest, most blurred image).
laplacian_pyramid=[layers]
for i in range(5,0,-1):
    size=gaussian_pyramid[i-1].shape[1],gaussian_pyramid[i-1].shape[0]
    gaussian_expanded=cv.pyrUp(gaussian_pyramid[i],dstsize=size)
    laplacian=cv.subtract(gaussian_pyramid[i-1],gaussian_expanded)
    laplacian_pyramid.append(laplacian)
    cv.imshow(str(i),laplacian)

# laplacian_pyramid = [layers] → start Laplacian pyramid with smallest Gaussian image.
# for i in range(5,0,-1): → loop from level 5 down to 1.
# size = gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0] → get width and height of next higher level.
# gaussian_expanded = cv.pyrUp(gaussian_pyramid[i], dstsize=size) → expand smaller image back to original size.
# laplacian = cv.subtract(gaussian_pyramid[i-1], gaussian_expanded) → subtract expanded image from original Gaussian level to get edge/detail layer.
# laplacian_pyramid.append(laplacian) → add Laplacian layer to list.
# cv.imshow(str(i), laplacian) → display each Laplacian layer.
# Syntax:
# expanded = cv.pyrUp(smaller_image, dstsize=(width, height))
# laplacian = cv.subtract(original, expanded)
# Effect: Captures details and edges at each level.
cv.imshow("Original image",img)
cv.waitKey(0)
cv.destroyAllWindows(
    
)
#Summary

# This code demonstrates Gaussian and Laplacian pyramids:

# Pyramid Type	Function / Effect
# Gaussian	Reduces image size iteratively with blur → smooth, smaller images.
# Laplacian	Difference between Gaussian levels → captures edges/details at each scale.

# Steps:

# Create Gaussian pyramid (smaller, blurred images).
# Start Laplacian pyramid from smallest Gaussian layer.
# Expand smaller Gaussian image, subtract from higher level → Laplacian layer.
# Display all layers including the original image.

# Purpose: Useful in image blending, compression, multi-scale analysis, and edge detection.
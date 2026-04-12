import cv2 as cv
import numpy as np


img = cv.imread("test.jpg")
#syntax===image = cv.imread("example.png", cv.IMREAD_GRAYSCALE)
     #-cv.imread() – reads an image from a file.
     #--- "test.jpg" – the filename of the image you want to read.
     # ---img – variable storing the loaded image in color by default (BGR format).
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#syntax=====gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    # -----cv.cvtColor() – converts an image from one color space to another.
    #------ img – the source image.
    #----- cv.COLOR_BGR2GRAY – conversion code: turns BGR color image into grayscale.
    #------ gray_img – stores the grayscale version of the image.
template=cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
#synatx======template = cv.imread("file_path", cv.IMREAD_GRAYSCALE)
    # -----template – variable that will store the template image.
    #----- cv.imread("test.jpg", cv.IMREAD_GRAYSCALE) – reads the same image, but directly in grayscale.
    #------ cv.IMREAD_GRAYSCALE – flag telling OpenCV to read the image as grayscale.
w,h=template.shape[::1]
#synatx=====height, width = template.shape
    #------template.shape – gives the dimensions of the image as (height, width) for 2D grayscale images.
    #------ [::1] – slicing, in this case it doesn’t change anything (it just takes the full shape).
    #------ w, h – tries to store width and height.
result=cv.matchTemplate(gray_img,template,cv.TM_CCOEFF_NORMED)
#synatx====result = cv.matchTemplate(source_image, template_image, method)
    # -------cv.matchTemplate() – searches for the template inside the gray_img.
    #------- gray_img – the image where we search.
    #----- template – the small patch we are looking for.
    #------ cv.TM_CCOEFF_NORMED – method for matching, giving values between -1 and 1.
    #----- result – a 2D array storing similarity scores for each position.
loc=np.where(result>=0.8)
#syntax=====indices = np.where(array >= threshold)
    # ------np.where() – finds indices in an array where a condition is true.
    #------ result >= 0.8 – we check for locations where the template matching score is 80% or higher.
    # -------loc – stores the coordinates of all matches.
for pt in zip(*loc[::-1]):
    #synatx===for coordinate in zip(*indices[::-1]):
    # do something with coordinate
#     loc[::-1] – reverses the order of coordinates (because OpenCV uses (x, y) but np.where gives (row, col) → (y, x)).
# *loc[::-1] – unpacks the coordinates so zip can pair them.
# zip(*loc[::-1]) – gives (x, y) tuples for each matching point.
# for pt in ... – loop through each matching point.
# pt – current top-left coordinate of a detected match.
    cv.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,0),3)
    #syntax===cv.rectangle(image, (x1, y1), (x2, y2), (B, G, R), thickness)
#     cv.rectangle() – draws a rectangle on an image.
# img – the image where rectangle is drawn.
# pt – top-left corner of the rectangle.
# (pt[0]+w, pt[1]+h) – bottom-right corner (calculated using template width and height).
# (0,255,0) – rectangle color in BGR → green.
# 3 – thickness of the rectangle border.
    cv.imshow("img",img)
    #synatx===cv.imshow("Window Name", image)
#     cv.imshow() – displays an image in a window.
# "img" – the window title.
# img – the image to display.
cv.waitKey(0)
#synatx====cv.waitKey(delay_in_ms)
# Waits for a key press to close the window.
# 0 means wait indefinitely
cv.destroyAllWindows()
# Closes all OpenCV windows opened by cv.imshow.



#SUMMARY
# Read the image in color and grayscale.
# Convert the color image to grayscale for processing.
# Use the image itself as a template for matching.
# Find where the template matches in the grayscale image using cv.matchTemplate.
# Get all locations where the match score is ≥ 0.8.
# Draw green rectangles around each detected match on the original image.
# Display the image with rectangles and wait for a key press.
# Close all OpenCV windows when done.
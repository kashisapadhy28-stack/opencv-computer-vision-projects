import cv2 as cv                                                                                                                                                        
import numpy as np

img=cv.imread("test.jpg")
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
mean_c=cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,12)
gaussian_c=cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,12)
cv.imshow("img",img)
cv.imshow("Binary threshold",mean_c)
cv.imshow("Gaussian",gaussian_c)
cv.waitKey(0)
cv.destroyAllWindows()
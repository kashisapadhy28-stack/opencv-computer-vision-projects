import cv2 as cv
import numpy as np
img=cv.imread("test.jpg")
rows,cols,_=img.shape
scaled_img=cv.resize(img,(400,500))
scaled_img=cv.resize(img,None,fx=1,fy=1)
matrix_t=np.float32([[1,0,50],[0,1,50]])
translated_img=cv.warpAffine(img,matrix_t,(cols,rows))
center = (cols/2,rows/2)
matrix_r=cv.getRotationMatrix2D(center,90,1)
rotated_img=cv.warpAffine(img,matrix_r,(cols,rows))

cv.imshow("Original images",img)
cv.imshow("scaled image",scaled_img)
cv.imshow("translated img",translated_img)
cv.imshow("rotated img",rotated_img)
cv.waitKey(0)
cv.destroyAllWindows()

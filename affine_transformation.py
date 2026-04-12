import cv2 as cv
import numpy as np

img=cv.imread("test.jpg")
rows,cols,_=img.shape
cv.circle(img,(83,90),5,(0,0,255),-1)
cv.circle(img,(447,90),5,(0,0,255),-1)
cv.circle(img,(83,475),5,(0,0,255),-1)
pts1=np.float32([[83,90],[447,90],[83,475]])
pts2=np.float32([[0,0],[400,0],[0,600]])
matrix=cv.getAffineTransform(pts1,pts2)
result=cv.warpAffine(img,matrix,(cols,rows))
cv.imshow("image",img)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2
import numpy as np

img1 = cv2.imread("test.jpg")
img2 = cv2.imread("logo.png")

# Resize img2 same size as img1
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert img2 to gray
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#create mask
ret,mask=cv2.threshold(gray,150,200,cv2.THRESH_BINARY_INV)

#apply mask to img1
road=cv2.bitwise_and(img1,img1,mask=mask)

# Blend
#syntax=result = cv2.addWeighted(src1, alpha, src2, beta, gamma)
blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

cv2.imshow("Masked Part",road)
cv2.imshow("Blended Image", blend)
cv2.waitKey(0)
cv2.destroyAllWindows()

#summary
#Ye program do images ko same size me resize karta hai. Logo image par grayscale aur threshold apply karke mask banata hai. Bitwise operation se masked background nikalta hai aur addWeighted() function se dono images ko blend karta hai. Final output me masked part aur blended image display hoti hai.
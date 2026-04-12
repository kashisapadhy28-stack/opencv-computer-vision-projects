import cv2 
import numpy as np

img1=cv2.imread("test.jpg")
img2=cv2.imread("logo.png")

#syntax=resized = cv2.resize(src, (width, height))
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

Bit_and=cv2.bitwise_and(img1,img2)    #result = cv2.bitwise_and(src1, src2)
Bit_or=cv2.bitwise_or(img1,img2)         #result = cv2.bitwise_or(src1, src2)
Bit_xor=cv2.bitwise_xor(img1,img2)            #result = cv2.bitwise_xor(src1, src2)
Bit__not=cv2.bitwise_not(img1)       #result = cv2.bitwise_not(src)
 
cv2.imshow("img1",img1)         
cv2.imshow("Bit_and",Bit_and)
cv2.imshow("Bit_or",Bit_or)
cv2.imshow("Bit_xor",Bit_xor)
cv2.imshow("Bit_not",Bit__not)

cv2.waitKey(0)
cv2.destroyAllWindows()

#summary
#Ye program do images ko same size me resize karta hai aur bitwise operations (AND, OR, XOR, NOT) apply karta hai. Har operation binary logic ke according pixel values combine karta hai aur alag-alag output windows me result show karta hai.
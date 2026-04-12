import cv2
import numpy as np

#images read karna
img1=cv2.imread("test.jpg")
img2=cv2.imread("logo_threshold.png")

#image size nikalna
height, width = img1.shape[:2]
scale = 0.5
#img1 resize karega
img1 = cv2.resize(img1, (int(width * scale), int(height * scale)))
#img2 resize karega
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#img addition
sum=cv2.add(img1,img2)
#grayscale conversion
img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#ttreshold apply karega
ret,threshold=cv2.threshold(img2_gray,170,255,cv2.THRESH_BINARY)
cv2.imshow("Image 1",img1)
cv2.imshow("Image 2",img2)
cv2.imshow("sum",sum)
cv2.imshow("threshold_img",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

#summary:
#Ye program do images load karta hai, pehli image ko scale karta hai aur doosri image ko uske size ke equal resize karta hai. Dono images ko pixel-wise add karta hai, phir logo image ko grayscale me convert karke binary threshold apply karta hai. Sab results ko display karta hai.
import cv2
import numpy as np

img1 = cv2.imread("test.jpg")   # background
img2 = cv2.imread("logo.png")   # 

# Get size of img1
height, width = img1.shape[:2]

# Scale factor
scale = 0.5

# Resize img1 (background scaling)
img1 = cv2.resize(img1, (int(width * scale), int(height * scale)))

#img 2 ko resize kiya
#syntax=resized_img = cv2.resize(src, (width, height))
img2 = cv2.resize(img2, (200, 200))

#shape
#syntax=height, width, channels = image.shape
rows, cols, _ = img2.shape

#logo img ko background img ke upar rakhne ke liye position 
x, y = 50, 50

#yaha pe background ka sir fpart le rahe hai jaha pe logo rakhna hai
#roi = image[y1:y2, x1:x2]
roi = img1[y:y+rows, x:x+cols]

#img2 ko gray scale me convert karna
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#thresholding se logo ke background ko remove karna
#syntax=ret, thresh_img = cv2.threshold(src, thresh_value, max_value, type)
#THRESH_BINARY_INV= 0 se 170(black) <pixel< 170 se 255(white)
ret, mask = cv2.threshold(img2_gray, 170, 255, cv2.THRESH_BINARY_INV)

#mask invert karna taki logo ke background ko white aur logo ko black kar de
#syntax=inverted = cv2.bitwise_not(src)  #bitwise not
mask_inv = cv2.bitwise_not(mask)

#syntax=result = cv2.bitwise_and(src1, src2, mask=mask)
bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)

#logo ka part lena
logo_part = cv2.bitwise_and(img2, img2, mask=mask)

#adding img:
#SYNTAX=result = cv2.add(src1, src2)
final = cv2.add(bg_part, logo_part)

#BACKGROUND IMAGE ME LOGO KO PLACE KARNA
img1[y:y+rows, x:x+cols] = final

cv2.imshow("Final Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#summary:
# Image read karta hai
# Resize karta hai
# ROI select karta hai
# Threshold se mask banata hai
# Bitwise operations se logo extract karta hai
# Background + logo combine karta hai
# Final image show karta hai

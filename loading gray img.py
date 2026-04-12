import cv2

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found!")
    exit()

height, width = img.shape[:2]
scale = 0.5
img = cv2.resize(img, (int(width*scale), int(height*scale)))
#yaha tak sab name hai pehele wale code ke sath

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("Original", img)
cv2.imshow("Gray", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("gray_test.jpg", gray_img)

#cv2.cvtColor()-color convert function hai 
#cv2.COLOR_BGR2GRAY-conversion oce hai BGR  se garyscale mein 
#cv2.imshow("Gray", gray_img)- ye dusri window mein grayscale batayega
#imwrite-ye write/save function hai,"gray_test.jpg"-ye new file name,gary img ye save honi wali img


#summarry
# 1️⃣ OpenCV load
# 2️⃣ Image read
# 3️⃣ Check image exists
# 4️⃣ Height & width nikale
# 5️⃣ 50% resize to the original img toh ray img khud proper size mein ajeyga
# 6️⃣ Grayscale conversion
# 7️⃣ Show both images
# 8️⃣ Wait for key
# 9️⃣ Close window
# 🔟 Save grayscale image
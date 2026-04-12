import cv2 as cv   #Load OpenCV and call it cv

img = cv.imread("test.jpg")   #Read test.jpg and store it in variable img


#condition hai ki img nhi mil toh not found mili toh success aur ye block mien hi load ho jaye

if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")

    height, width = img.shape[:2]

    scale = 0.5   # 50% size
    img = cv.resize(img, (int(width*scale), int(height*scale)))    #This resizes image to 50% of original size.

cv.imshow("Image", img)     #img show,1st arg-window name,2nd arg-image to display
cv.waitKey(0)     #wait for key press 
cv.destroyAllWindows()    #close all opencv windows



#img (variable name),imread- image read ye function hai jo image ko load karega file se
#imae.shape- gives dimension of img
#color image ke liye (h,w,channel)
#[:2]-aage ke 2 value hi le h,w aur unko store karo variables mein
#cv.resize()-function to resize img


#summary
# 1️⃣ Import OpenCV
# 2️⃣ Read image
# 3️⃣ Check if image loaded
# 4️⃣ Get height and width
# 5️⃣ Resize image to 50%
# 6️⃣ Show image            
# 7️⃣ Wait for key
# 8️⃣ Close window
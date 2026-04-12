import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap-camera capture object hai(variable name),cv2.VideoCapture()-video stream open karta hai
#matlb ki camera on hogya
#loop statement use kiya hai tabh tak true reega jab tak break nhi hota
while True:
    ret, frame = cap.read()
    #cap.read() ye camera ke ek frame ko cpture karta hai phir 2 cheezien return karta hai 1)ret2)frame
    #1)ret-camera true hai ya false (on/off)
    #2)frame-imag data(current pic)

    if not ret:
        break
#camera frame capture nhi kr paya tabh ye ayegaaur break-loop stop
    gray_Scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Scale", gray_Scale)
    cv2.imshow("Original", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

#summary
#1️⃣ Camera ON
# 2️⃣ Frame read
# 3️⃣ Error check
# 4️⃣ Convert to grayscale
# 5️⃣ Show 2 windows
# 6️⃣ Keyboard check
# 7️⃣ ESC press → exit
# 8️⃣ Camera release
# 9️⃣ Windows close

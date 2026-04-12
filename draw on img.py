import cv2
import numpy as np
image=cv2.imread("test.jpg")


shape=image.shape
print(shape)
#colors define karta hai
voilet=(180,0,180)
blue=(255,0,0)
red=(0,0,255)
green=(0,255,0)

#line draw karegga
#syntax=cv2.line(image, start_point, end_point, color, thickness)
cv2.line(image,(50,30),(450,35),blue,thickness=5)

#circle draw karega
#syntax=cv2.circle(image, center, radius, color, thickness)
cv2.circle(image,(250,250),26,red,-1)

#rectangle draw karega
#syntax=cv2.rectangle(image, top_left, bottom_right, color)
cv2.rectangle(image,(50,60),(450,400),green)

#ellipse draw karega
#syntax=cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
cv2.ellipse(image,(250,250),(100,50),0,0,180,voilet,thickness=3)   

#polygon points banana
#corrdinate ka array
point=np.array([[[140,230],[380,230],[320,250],[250,280]]],np.int32)
font=cv2.FONT_HERSHEY_SIMPLEX    #font define kiya

#polygon draw karega
#syntax=cv2.polylines(image, points, isClosed, color, thickness)
cv2.polylines(image,[point],True,green,thickness=4)

#text add karega
#syntax=cv2.putText(image, text, position, font, fontScale, color)
cv2.putText(image,"test.jpg",(20,100),font,4,red)

#img sow karega
cv2.imshow("test.jpg",image)

#wait and close
cv2.waitKey(0)

#sab windoes close hoga
cv2.destroyAllWindows()
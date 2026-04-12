import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Canvas for drawing
canvas = None

# Previous point
prev_x = 0
prev_y = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Draw line
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 0), 5)

            prev_x, prev_y = cx, cy

    # Merge drawing with camera
    frame = cv2.add(frame, canvas)

    cv2.imshow("Virtual Drawing App", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
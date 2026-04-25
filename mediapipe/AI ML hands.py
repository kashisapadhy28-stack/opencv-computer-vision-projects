import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(1)

gesture_label = "one_finger"   # CHANGE THIS manually while recording

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            # 👉 NORMALIZATION (VERY IMPORTANT)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)
                row.append(lm.z)

            row.append(gesture_label)

            # SAVE DATA
            with open("data.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)

            cv2.putText(frame, "Recording: " + gesture_label, (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
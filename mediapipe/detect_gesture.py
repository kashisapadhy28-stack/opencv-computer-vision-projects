import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load model
model = joblib.load("gesture_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Webcam
cap = cv2.VideoCapture(1)

# Smoothing
history = deque(maxlen=10)

gesture_to_command = {
    "Fist": "STAY",
    "open_palm": "STOP",
    "ok": "WALK",
    "peace": "BACKWARD",
    "thumbs_down": "SIT_DOWN",
    "thumbs_up": "STAND",
    "namasate": "NAMASTE",
    "one_finger": "FORWARD"
}

last_command = ""

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Normalize landmarks
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)
                row.append(lm.z)

            # Predict
            probs = model.predict_proba([row])
            confidence = np.max(probs)

            if confidence > 0.8:
                pred = model.predict([row])[0]
                history.append(pred)

                # Smooth output
                gesture = max(set(history), key=history.count)
            else:
                gesture = "Unknown"
# Display se pehle add karo
            if gesture in gesture_to_command:
               command = gesture_to_command[gesture]

               if command != last_command:
                print("Robot Command:", command)
                last_command = command
    # Display
    cv2.putText(frame, f"{gesture}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("AI Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
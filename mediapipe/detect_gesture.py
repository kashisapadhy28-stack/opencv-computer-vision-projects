import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# ==========================
# Load Model
# ==========================
model = joblib.load("gesture_model.pkl")

# ==========================
# MediaPipe Setup
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(1)   # Change to 0 if needed

# ==========================
# Smoothing
# ==========================
history = deque(maxlen=10)

# ==========================
# Gesture Mapping
# ==========================
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

# ==========================
# Variables
# ==========================
stable_gesture = None
gesture_start_time = None

HOLD_TIME = 2.0
CONFIDENCE_THRESH = 0.9

gesture = "No Hand"
confidence = 0.0

# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"
    confidence = 0.0

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:

        # Find nearest hand (largest bounding box)
        largest_area = 0
        selected_hand = None

        for hand_landmarks in result.multi_hand_landmarks:

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min = min(x_list)
            x_max = max(x_list)
            y_min = min(y_list)
            y_max = max(y_list)

            area = (x_max - x_min) * (y_max - y_min)

            if area > largest_area:
                largest_area = area
                selected_hand = hand_landmarks

        # Use only nearest hand
        if selected_hand is not None:
            hand_landmarks = selected_hand

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min = min(x_list)
            x_max = max(x_list)
            y_min = min(y_list)
            y_max = max(y_list)

            padding = 20

            cv2.rectangle(
                frame,
                (x_min - padding, y_min - padding),
                (x_max + padding, y_max + padding),
                (0, 255, 0),
                2
            )

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            row = []

            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)
                row.append(lm.z)

            probs = model.predict_proba([row])[0]
            confidence = np.max(probs)

            if confidence > CONFIDENCE_THRESH:

                pred = model.predict([row])[0]

                history.append(pred)

                gesture = max(set(history), key=history.count)

            else:
                gesture = "Unknown"

            cv2.putText(
                frame,
                gesture,
                (x_min - padding, y_min - padding - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (x_min - padding, y_max + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if gesture in gesture_to_command:

                if gesture != stable_gesture:
                    stable_gesture = gesture
                    gesture_start_time = time.time()

                elif (
                    gesture_start_time is not None
                    and time.time() - gesture_start_time >= HOLD_TIME
                ):

                    command = gesture_to_command[gesture]

                    print("Robot Command:", command)

                    gesture_start_time = time.time()

            else:
                stable_gesture = None
                gesture_start_time = None

    # ==========================
    # Hold Countdown
    # ==========================
    hold_text = ""

    if stable_gesture and gesture_start_time:

        elapsed = time.time() - gesture_start_time
        remaining = max(0, HOLD_TIME - elapsed)

        hold_text = f"Hold: {remaining:.1f}s"

    # ==========================
    # Global Display
    # ==========================
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        hold_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence*100:.1f}%",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("AI Gesture Control", frame)

    # ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()
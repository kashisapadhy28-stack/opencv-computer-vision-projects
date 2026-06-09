import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

model   = joblib.load("gesture_model.pkl")
CLASSES = model.classes_

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands            = 2,     # detect up to 2, but only use closest
    min_detection_confidence = 0.7,
    min_tracking_confidence  = 0.7,
)
cap = cv2.VideoCapture(1)

GESTURE_TO_COMMAND = {
    "Fist":        "STAY",
    "open_palm":   "STOP",
    "ok":          "WALK",
    "peace":       "BACKWARD",
    "thumbs_down": "SIT_DOWN",
    "thumbs_up":   "STAND",
    "one_finger":  "FORWARD",
}

# ── Tuning ───────────────────────────────────────────────────────────────────
LM_ALPHA      = 0.50   # landmark EMA  — high = responsive, low = smoother source
PROB_ALPHA    = 0.10   # probability EMA — low = very stable output
CONFIRM_THRESH = 0.50  # min smoothed probability to accept a gesture
HOLD_TIME      = 5.0   # seconds to hold gesture before command fires
# ─────────────────────────────────────────────────────────────────────────────

smoothed_lm    = None   # shape (21, 3) — EMA over landmark x,y,z
smoothed_probs = None   # shape (n_classes,) — EMA over model probabilities
stable_gesture     = None
gesture_start_time = None


def select_closest_hand(multi_landmarks, w, h):
    """Return the hand landmark with the largest bounding box (closest to camera)."""
    best, best_area = None, 0
    for lm_set in multi_landmarks:
        xs = [lm.x * w for lm in lm_set.landmark]
        ys = [lm.y * h for lm in lm_set.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best      = lm_set
    return best


def fix_prediction(pred, lm):
    """Geometry override: open_palm vs peace — check if ring+pinky are extended."""
    if pred in ("open_palm", "peace"):
        wrist     = np.array([lm[0].x,  lm[0].y])
        ring_tip  = np.array([lm[16].x, lm[16].y])
        ring_pip  = np.array([lm[14].x, lm[14].y])
        pinky_tip = np.array([lm[20].x, lm[20].y])
        pinky_pip = np.array([lm[18].x, lm[18].y])
        ring_ext  = np.linalg.norm(ring_tip  - wrist) > np.linalg.norm(ring_pip  - wrist)
        pinky_ext = np.linalg.norm(pinky_tip - wrist) > np.linalg.norm(pinky_pip - wrist)
        return "open_palm" if (ring_ext and pinky_ext) else "peace"
    return pred


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = hands.process(rgb)

    gesture     = "No Hand"
    raw_conf    = 0.0
    smooth_conf = 0.0

    if result.multi_hand_landmarks:

        # ── Step 1: Pick closest hand ─────────────────────────────────────
        hand_lm = select_closest_hand(result.multi_hand_landmarks, w, h)
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        lm = hand_lm.landmark

        # ── Step 2: Landmark EMA (source smoothing) ───────────────────────
        raw_lm = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)  # (21, 3)

        if smoothed_lm is None:
            smoothed_lm = raw_lm.copy()
        else:
            smoothed_lm = LM_ALPHA * raw_lm + (1 - LM_ALPHA) * smoothed_lm

        # Build feature row from smoothed landmarks
        base_x = smoothed_lm[0, 0]
        base_y = smoothed_lm[0, 1]
        row = []
        for x, y, z in smoothed_lm:
            row.append(x - base_x)
            row.append(y - base_y)
            row.append(z)

        # ── Step 3: Model prediction ──────────────────────────────────────
        raw_probs = model.predict_proba([row])[0]
        raw_conf  = float(np.max(raw_probs))

        # ── Step 4: Probability EMA (output smoothing) ────────────────────
        if smoothed_probs is None:
            smoothed_probs = raw_probs.copy()
        else:
            smoothed_probs = PROB_ALPHA * raw_probs + (1 - PROB_ALPHA) * smoothed_probs

        smooth_conf = float(np.max(smoothed_probs))
        pred        = CLASSES[int(np.argmax(smoothed_probs))]
        pred        = fix_prediction(pred, lm)

        gesture = pred if smooth_conf >= CONFIRM_THRESH else "Low conf"

        # ── Bounding box + label ──────────────────────────────────────────
        xs  = [int(p.x * w) for p in lm]
        ys  = [int(p.y * h) for p in lm]
        pad = 20
        cv2.rectangle(frame,
                      (min(xs)-pad, min(ys)-pad),
                      (max(xs)+pad, max(ys)+pad),
                      (0, 255, 0), 2)
        cv2.putText(frame, gesture,
                    (min(xs)-pad, min(ys)-pad-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show other hands as inactive (grey box)
        for other_lm in result.multi_hand_landmarks:
            if other_lm is hand_lm:
                continue
            oxs = [int(p.x * w) for p in other_lm.landmark]
            oys = [int(p.y * h) for p in other_lm.landmark]
            cv2.rectangle(frame,
                          (min(oxs)-pad, min(oys)-pad),
                          (max(oxs)+pad, max(oys)+pad),
                          (100, 100, 100), 1)
            cv2.putText(frame, "ignored",
                        (min(oxs)-pad, min(oys)-pad-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # ── Hold timer ────────────────────────────────────────────────────
        if gesture in GESTURE_TO_COMMAND:
            if gesture != stable_gesture:
                stable_gesture     = gesture
                gesture_start_time = time.time()
            elif gesture_start_time and time.time() - gesture_start_time >= HOLD_TIME:
                print("Robot Command:", GESTURE_TO_COMMAND[gesture])
                gesture_start_time = time.time()
        else:
            stable_gesture     = None
            gesture_start_time = None

    else:
        # Hand left frame — reset all smoothing state
        smoothed_lm        = None
        smoothed_probs     = None
        stable_gesture     = None
        gesture_start_time = None

    # ── Hold countdown bar ────────────────────────────────────────────────
    hold_text = ""
    if stable_gesture and gesture_start_time:
        elapsed   = time.time() - gesture_start_time
        remaining = max(0.0, HOLD_TIME - elapsed)
        hold_text = f"Hold: {remaining:.1f}s"

        # Progress bar for hold
        progress = min(elapsed / HOLD_TIME, 1.0)
        cv2.rectangle(frame, (10, 130), (10 + int(300 * progress), 148),
                      (0, 255, 100), -1)
        cv2.rectangle(frame, (10, 130), (310, 148), (100, 100, 100), 1)

    # ── HUD ───────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Gesture: {gesture}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, hold_text,
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, f"Raw:{raw_conf*100:.0f}%  Smooth:{smooth_conf*100:.0f}%",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    cv2.imshow("Robot Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

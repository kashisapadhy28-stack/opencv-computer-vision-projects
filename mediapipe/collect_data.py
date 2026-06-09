import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap      = cv2.VideoCapture(1)

# Each gesture has 5 phases x 100 samples = 500 total
# Each phase gives a specific angle/position instruction
GESTURES = [
    ("Fist", [
        ("1/5  STRAIGHT",   "Fist facing camera directly. Knuckles toward lens."),
        ("2/5  TILT LEFT",  "Tilt fist 20deg LEFT. Pinky side goes up."),
        ("3/5  TILT RIGHT", "Tilt fist 20deg RIGHT. Thumb side goes up."),
        ("4/5  THUMB IN",   "Thumb TIGHTLY wrapped over fingers. No thumb sticking out."),
        ("5/5  MOVE",       "Keep fist. Slowly move hand up/down/left/right while holding pose."),
    ]),
    ("thumbs_up", [
        ("1/5  STRAIGHT",   "Thumb pointing straight UP. All 4 fingers tightly curled."),
        ("2/5  TILT LEFT",  "Tilt hand 20deg LEFT. Thumb still points mostly upward."),
        ("3/5  TILT RIGHT", "Tilt hand 20deg RIGHT. Thumb still points mostly upward."),
        ("4/5  ANGLED",     "Rotate wrist slightly inward. Thumb at about 70deg angle."),
        ("5/5  MOVE",       "Keep thumbs up shape. Move hand closer then farther slowly."),
    ]),
    ("thumbs_down", [
        ("1/5  STRAIGHT",   "Thumb pointing straight DOWN. All 4 fingers tightly curled."),
        ("2/5  TILT LEFT",  "Tilt hand 20deg LEFT. Thumb still points mostly downward."),
        ("3/5  TILT RIGHT", "Tilt hand 20deg RIGHT. Thumb still points mostly downward."),
        ("4/5  ANGLED",     "Rotate wrist slightly. Thumb at about 110deg angle downward."),
        ("5/5  MOVE",       "Keep thumbs down shape. Move hand slowly in small circles."),
    ]),
    ("one_finger", [
        ("1/5  STRAIGHT",   "ONLY index finger pointing straight UP. All others tightly curled."),
        ("2/5  TILT LEFT",  "Tilt hand 20deg LEFT. Index still points mostly up."),
        ("3/5  TILT RIGHT", "Tilt hand 20deg RIGHT. Index still points mostly up."),
        ("4/5  FORWARD",    "Tilt index finger slightly toward camera. Thumb tucked tight."),
        ("5/5  MOVE",       "Keep one finger up. Slowly move hand up/down while holding pose."),
    ]),
    ("peace", [
        ("1/5  STRAIGHT",   "Index + Middle fingers in V shape pointing UP. Ring, pinky, thumb curled."),
        ("2/5  TILT LEFT",  "Tilt hand 20deg LEFT. Keep V shape tight."),
        ("3/5  TILT RIGHT", "Tilt hand 20deg RIGHT. Keep V shape tight."),
        ("4/5  SPREAD",     "Spread the V fingers WIDER apart. Others still tightly curled."),
        ("5/5  MOVE",       "Keep peace sign. Slowly move hand left/right while holding V shape."),
    ]),
    ("open_palm", [
        ("1/5  FLAT",       "All 5 fingers OPEN and flat. Palm facing camera directly."),
        ("2/5  TILT LEFT",  "Tilt open palm 20deg LEFT. All fingers still relaxed open."),
        ("3/5  TILT RIGHT", "Tilt open palm 20deg RIGHT. All fingers still relaxed open."),
        ("4/5  SPREAD",     "Spread all 5 fingers as WIDE apart as possible."),
        ("5/5  MOVE",       "Keep open palm. Move hand closer then farther from camera slowly."),
    ]),
    ("ok", [
        ("1/5  STRAIGHT",   "Index + Thumb make a circle. Other 3 fingers pointing UP."),
        ("2/5  TILT LEFT",  "Tilt hand 20deg LEFT. Keep index+thumb circle tight."),
        ("3/5  TILT RIGHT", "Tilt hand 20deg RIGHT. Keep index+thumb circle tight."),
        ("4/5  ROTATE",     "Rotate wrist slightly. Circle still visible to camera."),
        ("5/5  MOVE",       "Keep OK sign. Move hand slowly up/down while holding pose."),
    ]),
]

SAMPLES_PER_PHASE = 100
COUNTDOWN_SEC     = 4

gesture_idx   = 0
phase_idx     = 0
recording     = False
sample_count  = 0
countdown_start = None
status_msg    = "Press SPACE to start"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_name, phases    = GESTURES[gesture_idx]
    phase_label, phase_hint = phases[phase_idx]

    total_done = phase_idx * SAMPLES_PER_PHASE + sample_count

    # ── Top panel ──────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 155), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture_name}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

    cv2.putText(frame, f"Phase {phase_label}",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    cv2.putText(frame, phase_hint,
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    # Phase progress bar (cyan)
    phase_progress = sample_count / SAMPLES_PER_PHASE
    cv2.rectangle(frame, (15, 98),  (w - 15, 112), (40, 40, 40), -1)
    cv2.rectangle(frame, (15, 98),  (15 + int((w-30) * phase_progress), 112), (0, 220, 220), -1)
    cv2.putText(frame, f"Phase: {sample_count}/{SAMPLES_PER_PHASE}",
                (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 1)

    # Overall progress bar (green)
    total_progress = total_done / (SAMPLES_PER_PHASE * 5)
    cv2.rectangle(frame, (15, 135), (w - 15, 148), (40, 40, 40), -1)
    cv2.rectangle(frame, (15, 135), (15 + int((w-30) * total_progress), 148), (0, 200, 80), -1)
    cv2.putText(frame, f"Total:  {total_done}/{SAMPLES_PER_PHASE * 5}",
                (w//2, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 80), 1)

    # ── Gesture list (right side) ──────────────────────────────────────────
    for i, (gname, gphases) in enumerate(GESTURES):
        if i < gesture_idx:
            color = (60, 60, 60)
            label = f"{i+1}. {gname} DONE"
        elif i == gesture_idx:
            color = (0, 255, 100)
            label = f"{i+1}. {gname} <--"
        else:
            color = (80, 80, 80)
            label = f"{i+1}. {gname}"
        cv2.putText(frame, label,
                    (w - 230, 180 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # ── Countdown ─────────────────────────────────────────────────────────
    if countdown_start is not None and not recording:
        elapsed   = time.time() - countdown_start
        remaining = COUNTDOWN_SEC - elapsed
        if remaining > 0:
            cv2.putText(frame, f"Get ready... {int(remaining)+1}",
                        (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 3)
        else:
            recording       = True
            countdown_start = None
            status_msg      = "RECORDING..."

    # ── Recording ─────────────────────────────────────────────────────────
    if recording and result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        base_x = hand_lm.landmark[0].x
        base_y = hand_lm.landmark[0].y
        row = []
        for lm in hand_lm.landmark:
            row.append(lm.x - base_x)
            row.append(lm.y - base_y)
            row.append(lm.z)
        row.append(gesture_name)

        with open("data.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)

        sample_count += 1
        cv2.rectangle(frame, (3, 3), (w-3, h-3), (0, 255, 0), 4)

        if sample_count >= SAMPLES_PER_PHASE:
            recording    = False
            sample_count = 0
            phase_idx   += 1

            if phase_idx >= 5:
                phase_idx   = 0
                gesture_idx += 1

                if gesture_idx >= len(GESTURES):
                    cv2.putText(frame, "ALL DONE! Press Q.",
                                (w//2 - 150, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow("Data Collection", frame)
                    cv2.waitKey(3000)
                    break

                status_msg = f"Gesture done! Next: {GESTURES[gesture_idx][0]}. Press SPACE."
            else:
                status_msg = f"Phase done! Next phase: {phases[phase_idx][0]}. Press SPACE."

    elif result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    # ── Status bar ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h-42), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, status_msg,
                (15, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 255) if recording else (200, 200, 200), 2)
    cv2.putText(frame, "SPACE=start phase   Q=quit   S=skip phase",
                (w - 370, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not recording:
        countdown_start = time.time()
        status_msg      = "Get ready..."
    elif key == ord('s') and not recording:
        sample_count = 0
        phase_idx   += 1
        if phase_idx >= 5:
            phase_idx   = 0
            gesture_idx += 1
            if gesture_idx >= len(GESTURES):
                break
            status_msg = f"Skipped to: {GESTURES[gesture_idx][0]} Phase 1. Press SPACE."
        else:
            status_msg = f"Skipped to phase: {GESTURES[gesture_idx][1][phase_idx][0]}. Press SPACE."

cap.release()
cv2.destroyAllWindows()
print("Done! Delete old data.csv first, then run: python train_model.py")

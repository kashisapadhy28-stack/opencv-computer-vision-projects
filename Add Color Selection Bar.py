import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe hands setup ──────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

# ── Color palette (BGR) ───────────────────────────────────────────────────────
COLORS = {
    "Blue"  : (255,   0,   0),
    "Green" : (  0, 255,   0),
    "Red"   : (  0,   0, 255),
    "Yellow": (  0, 255, 255),
    "Eraser": (  0,   0,   0),
}
COLOR_NAMES = list(COLORS.keys())
NUM_COLORS  = len(COLOR_NAMES)

# Each color button occupies an equal horizontal slice across 640 px
BTN_H = 70          # button height in pixels
BTN_W = 640 // NUM_COLORS

def get_button_rect(idx):
    """Return (x1, y1, x2, y2) for colour button at index idx."""
    x1 = idx * BTN_W
    x2 = x1 + BTN_W
    return x1, 0, x2, BTN_H

def draw_color_buttons(frame, selected_name):
    for i, name in enumerate(COLOR_NAMES):
        x1, y1, x2, y2 = get_button_rect(i)
        bgr = COLORS[name]
        fill = bgr if name != "Eraser" else (60, 60, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)

        # Highlight selected button
        border_col = (255, 255, 255) if name == selected_name else (0, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 3)

        # Label
        label_col = (0, 0, 0) if name != "Eraser" else (200, 200, 200)
        cv2.putText(frame, name, (x1 + 8, y2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_col, 2)

def hit_button(cx, cy):
    """Return color name if (cx,cy) is inside any button, else None."""
    if cy < BTN_H:
        idx = cx // BTN_W
        if 0 <= idx < NUM_COLORS:
            return COLOR_NAMES[idx]
    return None

# ── Blue-object (bottle) tracker ──────────────────────────────────────────────
def find_blue_object(frame):
    """Return (cx, cy) of the largest blue contour, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([ 90,  50,  50])
    upper = np.array([130, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)
    mask  = cv2.erode (mask, None, iterations=2)
    mask  = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            return x + w // 2, y + h // 2, mask
    return None, None, mask

# ── Selection helper ──────────────────────────────────────────────────────────
HOVER_FRAMES_NEEDED = 15   # hold finger over button for this many frames to select
hover_counter = 0
hover_target  = None       # button name currently being hovered

# ── Main ──────────────────────────────────────────────────────────────────────
cap       = cv2.VideoCapture(0)
canvas    = None
prev_pt   = None
sel_color = "Blue"         # currently selected colour name
draw_pen  = COLORS["Blue"]

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    # ── 1. Hand / finger detection for colour selection ────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_pt = None   # index-fingertip position in pixels

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # Landmark 8 = index fingertip
        tip    = lm.landmark[8]
        fx, fy = int(tip.x * w), int(tip.y * h)
        finger_pt = (fx, fy)

        # Visual dot on fingertip
        cv2.circle(frame, finger_pt, 12, (0, 255, 255), -1)
        cv2.circle(frame, finger_pt, 12, (0, 0, 0), 2)

        # Check if fingertip hovers over a colour button
        hovered = hit_button(fx, fy)
        if hovered:
            if hovered == hover_target:
                hover_counter += 1
            else:
                hover_target  = hovered
                hover_counter = 1

            # Show dwell progress bar inside the button
            idx       = COLOR_NAMES.index(hovered)
            bx1, _, bx2, _ = get_button_rect(idx)
            fill_w = int((hover_counter / HOVER_FRAMES_NEEDED) * (bx2 - bx1))
            cv2.rectangle(frame, (bx1, BTN_H - 8), (bx1 + fill_w, BTN_H),
                          (255, 255, 255), -1)

            if hover_counter >= HOVER_FRAMES_NEEDED:
                sel_color   = hovered
                draw_pen    = COLORS[hovered]
                hover_counter = 0
                hover_target  = None
        else:
            hover_counter = 0
            hover_target  = None

    # ── 2. Blue-object tracking for drawing ───────────────────────────────
    cx, cy, mask = find_blue_object(frame)

    if cx is not None:
        # Don't draw while inside the button strip
        if cy > BTN_H + 5:
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy),  3, (0,   0,   0),   -1)

            if prev_pt is None:
                prev_pt = (cx, cy)

            thickness = 30 if sel_color == "Eraser" else 5
            cv2.line(canvas, prev_pt, (cx, cy), draw_pen, thickness)
            prev_pt = (cx, cy)
        else:
            prev_pt = None   # lift pen while in button zone
    else:
        prev_pt = None       # lift pen when object not visible

    # ── 3. Draw colour buttons on top ─────────────────────────────────────
    draw_color_buttons(frame, sel_color)

    # ── 4. Blend canvas onto frame ────────────────────────────────────────
    frame = cv2.add(frame, canvas)

    # ── 5. Status bar ─────────────────────────────────────────────────────
    status = f"Color: {sel_color}  |  Point finger at a button & hold to select  |  ESC to quit"
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, status, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── 6. Show windows ───────────────────────────────────────────────────
    cv2.imshow("Virtual Painter  [finger=select | bottle=draw]", frame)
    cv2.imshow("Blue Object Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
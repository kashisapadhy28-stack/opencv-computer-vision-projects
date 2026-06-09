import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import math

# ── Model & MediaPipe ─────────────────────────────────────────────────────────
model   = joblib.load("gesture_model.pkl")
CLASSES = model.classes_

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
cap = cv2.VideoCapture(1)

GESTURE_TO_COMMAND = {
    "Fist":        "STAY",
    "open_palm":   "FORWARD",
    "ok":          "WALK",
    "peace":       "BACKWARD",
    "thumbs_down": "SIT_DOWN",
    "thumbs_up":   "STAND",
}

# ── Tuning ────────────────────────────────────────────────────────────────────
LM_ALPHA       = 0.50   # landmark EMA  (0 = frozen, 1 = no smoothing)
PROB_ALPHA     = 0.10   # probability EMA
CONFIRM_THRESH = 0.50   # min smoothed prob to accept a gesture
HOLD_TIME      = 5.0    # seconds to hold gesture before command fires

DEBUG = True            # show per-finger state overlay; set False to hide

# ── EMA state ─────────────────────────────────────────────────────────────────
smoothed_lm        = None
smoothed_probs     = None
stable_gesture     = None
gesture_start_time = None


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
#
#  MediaPipe hand landmark indices:
#
#   0  Wrist
#   1  Thumb CMC    2  Thumb MCP    3  Thumb IP     4  Thumb TIP
#   5  Index MCP    6  Index PIP    7  Index DIP    8  Index TIP
#   9  Mid   MCP   10  Mid   PIP   11  Mid   DIP   12  Mid   TIP
#  13  Ring  MCP   14  Ring  PIP   15  Ring  DIP   16  Ring  TIP
#  17  Pinky MCP   18  Pinky PIP   19  Pinky DIP   20  Pinky TIP
#
#  Coordinate system (normalised 0-1, OpenCV layout):
#    x : left → right
#    y : top  → bottom   ← "screen-up" means DECREASING y
#    z : depth (less reliable, mostly unused here)
# =============================================================================

def _pt(lm, i):
    """
    XY of landmark i as a float32 (2,) array.
    Accepts both a MediaPipe NormalizedLandmarkList and a (21, 3) numpy array.
    """
    if isinstance(lm, np.ndarray):
        return lm[i, :2].astype(np.float32)
    return np.array([lm[i].x, lm[i].y], dtype=np.float32)


def _angle_deg(a, b, c):
    """
    Angle at vertex b formed by rays b→a and b→c, in degrees.
    A perfectly straight segment returns 180°.
    A fully folded joint returns ~0-90°.
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_v = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_v)))


# ── Finger extension (index / middle / ring / pinky) ─────────────────────────

def _finger_extended(lm, mcp, pip, dip, tip, thr=155.0):
    """
    True when the finger is open/extended.  Uses two complementary methods.

    Method 1 – 2D joint angles (PIP and DIP):
      Works for hands held flat/sideways to the camera.
      FAILS when the finger points toward or away from the camera lens:
      the 2D projection collapses the finger, making all joint angles
      appear small even though the finger is fully straight
      (foreshortening artifact — this is why an open palm can read as
      four_closed=True when the hand faces the camera directly).

    Method 2 – 3D wrist-to-landmark distance ratio (foreshortening fix):
      Uses MediaPipe's Z coordinate (wrist-relative depth, same scale as X/Y).
      Extended finger → tip is farther from wrist in 3D than PIP.
      Folded   finger → tip curls back past MCP toward the palm,
                        landing CLOSER to the wrist than the PIP in 3D.
      Threshold 1.15 sits between the folded ratio (~0.65-0.85)
      and the extended ratio (~1.40-1.70) with comfortable margin.
    """
    # ── Method 1: 2D joint angles ─────────────────────────────────────────────
    pip_ang = _angle_deg(_pt(lm, mcp), _pt(lm, pip), _pt(lm, dip))
    dip_ang = _angle_deg(_pt(lm, pip), _pt(lm, dip), _pt(lm, tip))
    if (pip_ang > thr) and (dip_ang > thr):
        return True

    # ── Method 2: 3D distance ratio (camera-facing foreshortening fix) ────────
    if isinstance(lm, np.ndarray):
        w3 = lm[0,   :3]
        t3 = lm[tip, :3]
        p3 = lm[pip, :3]
    else:
        w3 = np.array([lm[0].x,   lm[0].y,   lm[0].z],   dtype=np.float32)
        t3 = np.array([lm[tip].x, lm[tip].y, lm[tip].z], dtype=np.float32)
        p3 = np.array([lm[pip].x, lm[pip].y, lm[pip].z], dtype=np.float32)
    return np.linalg.norm(t3 - w3) > np.linalg.norm(p3 - w3) * 1.15


# ── Thumb state ───────────────────────────────────────────────────────────────

def _thumb_extended(lm, thr_ip=140.0, thr_mcp=115.0):
    """
    True when the thumb is NOT wrapped against the palm (i.e. open/abducted).

    WHY two joints:
      In a Fist the thumb wraps OVER the index/middle fingers:
        IP  joint bends sharply  (~80-110°)
        MCP joint also bends     (~70-100°)
      For Thumbs-Up or a sideways hitchhiker thumb both joints stay straight:
        IP  ~160-175°
        MCP ~130-165°
      Checking IP alone is usually enough, but adding MCP eliminates
      near-threshold poses (thumb barely touching the index knuckle) that
      fool a single-joint check.

    NOTE: The thumb moves in a DIFFERENT PLANE than the other fingers.
      It adducts/abducts laterally, so a "tip above MCP in Y" rule is
      unreliable for any hand that is rotated or tilted sideways.
    """
    ip_ang  = _angle_deg(_pt(lm, 2), _pt(lm, 3), _pt(lm, 4))  # MCP → IP → TIP
    mcp_ang = _angle_deg(_pt(lm, 1), _pt(lm, 2), _pt(lm, 3))  # CMC → MCP → IP
    return (ip_ang > thr_ip) and (mcp_ang > thr_mcp)


def _thumb_away_from_fingers(lm, ratio_thr=0.30):
    """
    True when the thumb tip is clearly SEPARATED from the finger region.

    WHY this check is needed:
      In a Fist the thumb wraps over/against the index+middle fingers.
      The thumb IP joint does NOT fully flex in this pose — it often reads
      ~145-160°, above the _thumb_extended() threshold.  Simultaneously,
      the MCP→TIP vector points upward on screen (tip resting on the index
      knuckle), so _thumb_screen_direction() returns 'up'.
      The result: a Fist is misclassified as thumbs_up.

      The spatial fix: in a Fist the thumb TIP is physically touching the
      index/middle PIP area (minimum distance is small).  In a true
      thumbs-up/down/sideways the tip is pulled well away from all fingers.

    Metric:
      min_dist  = distance from thumb tip to the nearest of
                  { index MCP, index PIP, middle MCP, middle PIP }
      Normalised by wrist-to-middle-MCP (hand scale).
      Threshold 0.30 separates fist (ratio ≈ 0.05-0.20) from
      extended thumb (ratio ≈ 0.50-1.20) with a comfortable margin.
    """
    thumb_tip  = _pt(lm, 4)
    check_pts  = [_pt(lm, 5),   # index  MCP
                  _pt(lm, 6),   # index  PIP  ← where fist-thumb rests
                  _pt(lm, 9),   # middle MCP
                  _pt(lm, 10)]  # middle PIP
    min_dist   = min(np.linalg.norm(thumb_tip - p) for p in check_pts)
    hand_scale = np.linalg.norm(_pt(lm, 9) - _pt(lm, 0)) + 1e-9
    return (min_dist / hand_scale) > ratio_thr


def _thumb_screen_direction(lm):
    """
    Returns which direction the thumb tip points in SCREEN SPACE.
    Result: 'up' | 'down' | 'sideways'

    Method:
      Compute the angle between the MCP→TIP vector and (0, -1),
      which is "screen-up" in OpenCV's y-down coordinate system.

    Thresholds (generous to tolerate normal wrist tilt):
      0°  – 55°   → 'up'        (classic thumbs-up pose)
      55° – 125°  → 'sideways'  (hitchhiker / side thumb — NOT thumbs-up)
      125° – 180° → 'down'      (thumbs-down pose)

    WHY screen-absolute instead of hand-relative:
      "Thumbs up" has a physical meaning — thumb toward the ceiling.
      A hand-relative approach ("thumb along finger axis") would classify
      a horizontal hitchhiker thumb as 'up' when the hand is held sideways,
      which is semantically wrong for robot commands.
    """
    vec = _pt(lm, 4) - _pt(lm, 2)   # MCP → TIP
    n   = np.linalg.norm(vec)
    if n < 1e-9:
        return 'unknown'
    angle = math.degrees(
        math.acos(np.clip(np.dot(vec / n, np.array([0.0, -1.0])), -1.0, 1.0))
    )
    if angle < 55:
        return 'up'
    if angle > 125:
        return 'down'
    return 'sideways'


# ── Public API ────────────────────────────────────────────────────────────────

def get_finger_states(lm):
    """
    Returns a dict[str, bool]: True = extended, False = folded.
    Keys: 'index', 'middle', 'ring', 'pinky', 'thumb'
    Works with both MediaPipe landmark lists and (21, 3) numpy arrays.
    """
    return {
        'index':  _finger_extended(lm,  5,  6,  7,  8),
        'middle': _finger_extended(lm,  9, 10, 11, 12),
        'ring':   _finger_extended(lm, 13, 14, 15, 16),
        'pinky':  _finger_extended(lm, 17, 18, 19, 20),
        'thumb':  _thumb_extended(lm),
    }


def geometry_override(ml_pred, lm):
    """
    Priority-ordered rule-based classifier applied AFTER the ML model.

    WHY layer rules on top of a 99.5%-accurate model:
      Training accuracy measures how well the model fits your 500-sample
      dataset, not how well it generalises.  Features built from raw (x, y)
      coordinates are neither scale-invariant (hand distance from camera)
      nor rotation-invariant (wrist twist).  The model likely memorised
      pose-specific patterns, so it fails on novel angles.
      Hard geometric rules are correct by construction: if the IP joint
      angle is provably < 140° the thumb IS bent, regardless of what the
      model thinks.

    Priority order (highest → lowest specificity):
      1. Fist          – all 5 digits closed
      2. thumbs_up     – 4 fingers closed + thumb up
      3. thumbs_down   – 4 fingers closed + thumb down
      4. Unknown       – 4 fingers closed + thumb sideways   (explicit non-match)
      5. (removed)     – one_finger no longer used
      6. peace         – index + middle extended
      7. ok            – middle+ring+pinky up + thumb-index pinch
      8. open_palm     – all 5 extended
      9. (fallback)    – trust ML model for ambiguous combinations

    Placing Fist at priority 1 means it can NEVER be misclassified as
    thumbs_up, even if the model assigns 100% confidence to thumbs_up.
    """
    s = get_finger_states(lm)
    four_closed = not (s['index'] or s['middle'] or s['ring'] or s['pinky'])
    all_open    = s['index'] and s['middle'] and s['ring'] and s['pinky']

    # ── 1. Fist ───────────────────────────────────────────────────────────────
    # Fires when 4 fingers are folded AND either:
    #   (a) thumb angle check says it is bent, OR
    #   (b) thumb tip is spatially pressed against the index/middle fingers.
    # Case (b) catches the "thumb wrapped over knuckles" fist where the IP
    # joint stays ~150° (above the bend threshold) but the tip physically
    # touches the index PIP — the classic false thumbs_up situation.
    thumb_pressed = not _thumb_away_from_fingers(lm)
    if four_closed and (not s['thumb'] or thumb_pressed):
        return 'Fist'

    # ── 2–4. Four fingers closed, thumb clearly away from fingers ─────────────
    # Only reaches here when the thumb tip is genuinely separated from the
    # palm — meaning it is actually extended in some direction.
    if four_closed and s['thumb']:
        direction = _thumb_screen_direction(lm)
        if direction == 'up':
            return 'thumbs_up'
        if direction == 'down':
            return 'thumbs_down'
        # Sideways thumb: ambiguous, not a defined command
        return 'Unknown'

    # ── 5. Peace / V Sign ─────────────────────────────────────────────────────
    if s['index'] and s['middle'] and not s['ring'] and not s['pinky']:
        return 'peace'

    # ── 7. OK Sign ────────────────────────────────────────────────────────────
    # Three fingers up AND thumb tip pinched to index tip.
    # Scale the pinch threshold by middle-finger length so it works
    # at any hand distance from the camera.
    if s['middle'] and s['ring'] and s['pinky']:
        mid_len = np.linalg.norm(_pt(lm, 12) - _pt(lm, 9))   # middle finger length
        pinch   = np.linalg.norm(_pt(lm,  4) - _pt(lm, 8))   # thumb tip → index tip
        if pinch < mid_len * 0.55:
            return 'ok'

    # ── 8. Open Palm ──────────────────────────────────────────────────────────
    if all_open:
        return 'open_palm'

    # ── 9. Fallback: trust ML model for any combination rules don't cover ─────
    return ml_pred


# =============================================================================
# UTILITY
# =============================================================================

def select_closest_hand(multi_landmarks, w, h):
    """Return the hand landmark set with the largest bounding box (closest to cam)."""
    best, best_area = None, 0
    for lm_set in multi_landmarks:
        xs = [lm.x * w for lm in lm_set.landmark]
        ys = [lm.y * h for lm in lm_set.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best      = lm_set
    return best


def _3d_ratio(lm, pip, tip):
    """Return dist(wrist,tip)/dist(wrist,pip) in 3D — used in debug overlay."""
    if isinstance(lm, np.ndarray):
        w3 = lm[0, :3]
        t3 = lm[tip, :3]
        p3 = lm[pip, :3]
    else:
        w3 = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float32)
        t3 = np.array([lm[tip].x, lm[tip].y, lm[tip].z], dtype=np.float32)
        p3 = np.array([lm[pip].x, lm[pip].y, lm[pip].z], dtype=np.float32)
    denom = np.linalg.norm(p3 - w3) + 1e-9
    return np.linalg.norm(t3 - w3) / denom


def draw_debug_overlay(frame, lm, w):
    """
    Top-right overlay showing per-finger extension state, thumb direction,
    and the 3D wrist-to-tip ratio for the index finger.

    Green O = extended, Red X = folded.
    3D ratio > 1.15 means the 3D fallback counts the finger as extended.
    Use this to tune the 1.15 threshold in _finger_extended if needed.
    """
    states    = get_finger_states(lm)
    thumb_dir = _thumb_screen_direction(lm)
    labels    = [('T', 'thumb'), ('I', 'index'), ('M', 'middle'),
                 ('R', 'ring'),  ('P', 'pinky')]
    for row_i, (label, key) in enumerate(labels):
        open_ = states[key]
        color = (0, 220, 0) if open_ else (0, 0, 220)
        cv2.putText(frame, f"{label}:{'O' if open_ else 'X'}",
                    (w - 75, 38 + row_i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    away  = _thumb_away_from_fingers(lm)
    ratio = _3d_ratio(lm, 6, 8)   # index finger PIP=6, TIP=8
    cv2.putText(frame, f"dir:{thumb_dir}",
                (w - 115, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1)
    cv2.putText(frame, f"away:{'Y' if away else 'N'}",
                (w - 115, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 0), 1)
    cv2.putText(frame, f"3D:{ratio:.2f}",
                (w - 115, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1)


# =============================================================================
# MAIN LOOP
# =============================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result   = hands.process(rgb)
    rgb.flags.writeable = True

    gesture     = "No Hand"
    raw_conf    = 0.0
    smooth_conf = 0.0

    if result.multi_hand_landmarks:

        # ── Step 1: Pick closest hand ─────────────────────────────────────────
        hand_lm = select_closest_hand(result.multi_hand_landmarks, w, h)
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        lm = hand_lm.landmark

        # ── Step 2: Landmark EMA (source smoothing) ───────────────────────────
        raw_lm = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)
        if smoothed_lm is None:
            smoothed_lm = raw_lm.copy()
        else:
            smoothed_lm = LM_ALPHA * raw_lm + (1 - LM_ALPHA) * smoothed_lm

        # ── Feature extraction — MUST match collect_data.py exactly ─────────────
        # Current format (original model, no retraining yet):
        #   wrist-relative XY + Z, no normalisation → 63 features
        #
        # After you recollect data with the updated collect_data.py and retrain,
        # replace this block with the RETRAINED FORMAT below:
        #
        #   base_x, base_y = smoothed_lm[0, 0], smoothed_lm[0, 1]
        #   rel = []
        #   for x, y, _ in smoothed_lm:          # _ drops Z
        #       rel.append(x - base_x)
        #       rel.append(y - base_y)
        #   max_val = max(abs(v) for v in rel) + 1e-9
        #   row = [v / max_val for v in rel]      # 42 features, normalised [-1,1]
        #
        # Until you retrain, keep the block below as-is.
        base_x, base_y = smoothed_lm[0, 0], smoothed_lm[0, 1]
        row = []
        for x, y, z in smoothed_lm:
            row.append(x - base_x)
            row.append(y - base_y)
            row.append(z)

        # ── Step 3: ML model prediction ───────────────────────────────────────
        raw_probs = model.predict_proba([row])[0]
        raw_conf  = float(np.max(raw_probs))

        # ── Step 4: Probability EMA (output smoothing) ────────────────────────
        if smoothed_probs is None:
            smoothed_probs = raw_probs.copy()
        else:
            smoothed_probs = PROB_ALPHA * raw_probs + (1 - PROB_ALPHA) * smoothed_probs

        smooth_conf = float(np.max(smoothed_probs))
        ml_pred     = CLASSES[int(np.argmax(smoothed_probs))]

        # ── Step 5: Geometry override ─────────────────────────────────────────
        # Rules run on the RAW (un-smoothed) landmarks for maximum responsiveness.
        # The ML model handles ambiguous multi-finger combinations the rules
        # don't explicitly cover (e.g. three-finger combinations).
        pred    = geometry_override(ml_pred, lm)
        gesture = pred if smooth_conf >= CONFIRM_THRESH else "Low conf"

        # ── Step 6: Debug overlay ─────────────────────────────────────────────
        if DEBUG:
            draw_debug_overlay(frame, lm, w)

        # ── Bounding box + label ──────────────────────────────────────────────
        xs  = [int(p.x * w) for p in lm]
        ys  = [int(p.y * h) for p in lm]
        pad = 20
        cv2.rectangle(frame,
                      (min(xs) - pad, min(ys) - pad),
                      (max(xs) + pad, max(ys) + pad),
                      (0, 255, 0), 2)
        cv2.putText(frame, gesture,
                    (min(xs) - pad, min(ys) - pad - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show inactive hands as grey
        for other_lm in result.multi_hand_landmarks:
            if other_lm is hand_lm:
                continue
            oxs = [int(p.x * w) for p in other_lm.landmark]
            oys = [int(p.y * h) for p in other_lm.landmark]
            cv2.rectangle(frame,
                          (min(oxs) - pad, min(oys) - pad),
                          (max(oxs) + pad, max(oys) + pad),
                          (100, 100, 100), 1)
            cv2.putText(frame, "ignored",
                        (min(oxs) - pad, min(oys) - pad - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # ── Hold timer ────────────────────────────────────────────────────────
        if gesture in GESTURE_TO_COMMAND:
            if gesture != stable_gesture:
                stable_gesture     = gesture
                gesture_start_time = time.time()
                print(f"Gesture: {gesture}  →  Command: {GESTURE_TO_COMMAND[gesture]}")
            elif gesture_start_time and time.time() - gesture_start_time >= HOLD_TIME:
                print(f"Robot Command CONFIRMED: {GESTURE_TO_COMMAND[gesture]}")
                gesture_start_time = time.time()
        else:
            stable_gesture     = None
            gesture_start_time = None

    else:
        smoothed_lm        = None
        smoothed_probs     = None
        stable_gesture     = None
        gesture_start_time = None

    # ── Hold countdown bar ────────────────────────────────────────────────────
    hold_text = ""
    if stable_gesture and gesture_start_time:
        elapsed   = time.time() - gesture_start_time
        remaining = max(0.0, HOLD_TIME - elapsed)
        hold_text = f"Hold: {remaining:.1f}s"
        progress  = min(elapsed / HOLD_TIME, 1.0)
        cv2.rectangle(frame, (10, 130), (10 + int(300 * progress), 148),
                      (0, 255, 100), -1)
        cv2.rectangle(frame, (10, 130), (310, 148), (100, 100, 100), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
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

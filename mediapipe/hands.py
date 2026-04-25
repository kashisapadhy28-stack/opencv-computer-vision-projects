# =============================================================================
#  CANIS — Real-Time Hand Gesture Control for Robotic Dog  v3
#  ─────────────────────────────────────────────────────────
#  10 Gestures  |  Enhanced Stability  |  State Machine  |  Cooldown
#  ANGLE-ROBUST: Uses hand-relative coordinate system (palm basis vectors)
#                so gestures work from any viewing angle or hand orientation.
#
#  Gesture → Command Map
#  ─────────────────────────────────────────────────────────
#  ✋  Open Palm        [1 1 1 1 1]  spread low, no shake  → CANIS_STOP
#  👍  Thumbs Up        [1 0 0 0 0]  tip Y < wrist Y       → CANIS_WALK
#  👎  Thumbs Down      [1 0 0 0 0]  tip Y > wrist Y       → CANIS_SIT
#  ✊  Fist             [0 0 0 0 0]  all curled             → CANIS_STAY
#  ☝️  One Finger Up    [0 1 0 0 0]  tip X ≈ wrist X       → CANIS_STAND
#  👉  Point Right      [0 1 0 0 0]  tip X > wrist X+0.10  → CANIS_TURN_RIGHT
#  👈  Point Left       [0 1 0 0 0]  tip X < wrist X-0.10  → CANIS_TURN_LEFT
#  ✌️  Two Fingers      [0 1 1 0 0]  unique pattern        → CANIS_MOVE_MODE
#  🤚  Spread Fingers   [1 1 1 1 1]  spread score > 0.10   → CANIS_ATTENTION
#  ✋🔀 Palm + Shake    [1 1 1 1 1]  X-axis osc ≥5/1.2s    → CANIS_EXCITED
#  ─────────────────────────────────────────────────────────
#
#  Install:   pip install opencv-python mediapipe numpy
#  Run:       python canis_gesture_control_v3.py
#  Keys:      Q=quit  R=reset  S=screenshot
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np
import time
import collections

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class Config:
    """All tuneable parameters in one place. Edit these to adjust behaviour."""

    # ── Webcam ─────────────────────────────────────────────────────────────
    WEBCAM_INDEX        = 0          # 0 = default camera
    FRAME_WIDTH         = 1280
    FRAME_HEIGHT        = 720

    # ── MediaPipe ──────────────────────────────────────────────────────────
    MAX_HANDS           = 1          # only track one hand
    DETECTION_CONF      = 0.80       # initial detection confidence
    TRACKING_CONF       = 0.70       # tracking confidence

    # ── Stability (Enhanced) ───────────────────────────────────────────────
    # A gesture must appear consistently for this many frames before firing.
    STABILITY_REQUIRED  = 8          # frames
    STABILITY_TOLERANCE = 0.875      # fraction of frames that must agree

    # ── Cooldown ───────────────────────────────────────────────────────────
    COOLDOWN_SECONDS    = 1.2        # seconds before same state fires again

    # ── Gesture thresholds ─────────────────────────────────────────────────
    # All thresholds are now in *palm-relative* normalised units (0–1).
    # They are scale-invariant: results are the same whether the hand is
    # close to or far from the camera.

    FINGER_CURL_THRESH       = 0.40  # ratio of projected extension / palm length
                                     # above this → extended, below → curled
    THUMB_UP_DOWN_DEADZONE   = 0.12  # palm-axis dead-zone for thumb direction
    POINT_HORIZONTAL_THRESH  = 0.25  # palm-axis X offset to detect left/right point
    SPREAD_SCORE_THRESH      = 0.10  # mean fingertip spacing (screen-space, unchanged)

    # ── Shake detection ────────────────────────────────────────────────────
    SHAKE_WINDOW_SEC         = 1.2
    SHAKE_MIN_REVERSALS      = 5
    SHAKE_AMPLITUDE_THRESH   = 0.06

    # ── HUD ────────────────────────────────────────────────────────────────
    FONT                = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_ALPHA         = 0.60
    WINDOW_TITLE        = "CANIS — Gesture Control v3"


# =============================================================================
# SECTION 2 — GESTURE & STATE DEFINITIONS
# =============================================================================

G_OPEN_PALM    = "OPEN_PALM"
G_THUMBS_UP    = "THUMBS_UP"
G_THUMBS_DOWN  = "THUMBS_DOWN"
G_FIST         = "FIST"
G_ONE_FINGER   = "ONE_FINGER"
G_POINT_RIGHT  = "POINT_RIGHT"
G_POINT_LEFT   = "POINT_LEFT"
G_TWO_FINGERS  = "TWO_FINGERS"
G_SPREAD       = "SPREAD_FINGERS"
G_EXCITED      = "PALM_SHAKE"
G_UNKNOWN      = "UNKNOWN"

S_STOP         = "CANIS_STOP"
S_WALK         = "CANIS_WALK"
S_SIT          = "CANIS_SIT"
S_STAY         = "CANIS_STAY"
S_STAND        = "CANIS_STAND"
S_TURN_RIGHT   = "CANIS_TURN_RIGHT"
S_TURN_LEFT    = "CANIS_TURN_LEFT"
S_MOVE_MODE    = "CANIS_MOVE_MODE"
S_ATTENTION    = "CANIS_ATTENTION"
S_EXCITED      = "CANIS_EXCITED"
S_NONE         = "CANIS_NONE"

GESTURE_TO_STATE = {
    G_OPEN_PALM:   S_STOP,
    G_THUMBS_UP:   S_WALK,
    G_THUMBS_DOWN: S_SIT,
    G_FIST:        S_STAY,
    G_ONE_FINGER:  S_STAND,
    G_POINT_RIGHT: S_TURN_RIGHT,
    G_POINT_LEFT:  S_TURN_LEFT,
    G_TWO_FINGERS: S_MOVE_MODE,
    G_SPREAD:      S_ATTENTION,
    G_EXCITED:     S_EXCITED,
}

GESTURE_COLOR = {
    G_OPEN_PALM:   (50,  230, 80),
    G_THUMBS_UP:   (50,  220, 255),
    G_THUMBS_DOWN: (50,  80,  220),
    G_FIST:        (60,  60,  210),
    G_ONE_FINGER:  (200, 200, 50),
    G_POINT_RIGHT: (0,   160, 255),
    G_POINT_LEFT:  (255, 100, 50),
    G_TWO_FINGERS: (200, 50,  255),
    G_SPREAD:      (0,   255, 200),
    G_EXCITED:     (0,   200, 255),
    G_UNKNOWN:     (100, 100, 100),
}


# =============================================================================
# SECTION 3 — MEDIAPIPE LANDMARK INDICES
# =============================================================================

LM = mp.solutions.hands.HandLandmark

WRIST           = LM.WRIST
THUMB_TIP       = LM.THUMB_TIP
THUMB_IP        = LM.THUMB_IP
THUMB_MCP       = LM.THUMB_MCP
INDEX_TIP       = LM.INDEX_FINGER_TIP
INDEX_PIP       = LM.INDEX_FINGER_PIP
INDEX_MCP       = LM.INDEX_FINGER_MCP
MIDDLE_TIP      = LM.MIDDLE_FINGER_TIP
MIDDLE_PIP      = LM.MIDDLE_FINGER_PIP
MIDDLE_MCP      = LM.MIDDLE_FINGER_MCP
RING_TIP        = LM.RING_FINGER_TIP
RING_PIP        = LM.RING_FINGER_PIP
RING_MCP        = LM.RING_FINGER_MCP
PINKY_TIP       = LM.PINKY_TIP
PINKY_PIP       = LM.PINKY_PIP
PINKY_MCP       = LM.PINKY_MCP

ALL_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


# =============================================================================
# SECTION 4 — HAND BASIS (ANGLE-ROBUST COORDINATE SYSTEM)
# =============================================================================

def build_hand_basis(lm):
    """
    Construct a normalised orthonormal basis (palm frame) from hand landmarks.

    Why this matters:
        Comparing raw Y coordinates only works when the hand is upright.
        If the hand is tilted 45° or even sideways, tip.y < pip.y can
        completely reverse, causing misclassifications.

        Instead, we build a *local* coordinate system aligned with the
        palm itself, then project every landmark into that frame.
        Gestures evaluated in this frame are invariant to hand rotation,
        tilt, and distance from the camera.

    Basis vectors (all in normalised image space):
        palm_up  : direction from wrist toward middle-finger MCP
                   ("up" relative to the hand, regardless of hand tilt)
        palm_right : perpendicular, pointing toward the pinky side

    Palm length (scalar):
        Distance from wrist to middle MCP, used to normalise all
        projected distances so they are scale-invariant.

    Returns:
        palm_up    : np.ndarray shape (2,) — unit vector "up" in palm frame
        palm_right : np.ndarray shape (2,) — unit vector "right" in palm frame
        palm_len   : float — length of wrist→middle-MCP (normalisation factor)
        wrist_pt   : np.ndarray shape (2,) — wrist position in screen space
    """
    def pt(idx):
        return np.array([lm[idx].x, lm[idx].y], dtype=float)

    wrist_pt  = pt(WRIST)
    mid_mcp   = pt(MIDDLE_MCP)

    raw_up    = mid_mcp - wrist_pt
    palm_len  = float(np.linalg.norm(raw_up))

    if palm_len < 1e-6:
        # Degenerate (hand collapsed to point) — return identity basis
        return np.array([0., -1.]), np.array([1., 0.]), 0.01, wrist_pt

    palm_up   = raw_up / palm_len            # unit vector wrist→mid-MCP
    # Rotate 90° CCW to get the "right" direction (toward pinky side)
    palm_right = np.array([-palm_up[1], palm_up[0]])

    return palm_up, palm_right, palm_len, wrist_pt


def project_palm(pt_idx, lm, palm_up, palm_right, palm_len, wrist_pt):
    """
    Project a landmark into the palm-relative coordinate system.

    Returns (along, lateral) where:
        along   : component along palm_up (positive = toward fingertips)
        lateral : component along palm_right (positive = toward pinky)
    Both are divided by palm_len so values are scale-independent.

    Args:
        pt_idx  : MediaPipe landmark index
        lm      : landmark list
        palm_up, palm_right, palm_len, wrist_pt : from build_hand_basis()
    """
    p     = np.array([lm[pt_idx].x, lm[pt_idx].y], dtype=float)
    delta = p - wrist_pt
    along   = float(np.dot(delta, palm_up))    / palm_len
    lateral = float(np.dot(delta, palm_right)) / palm_len
    return along, lateral


# =============================================================================
# SECTION 5 — FINGER STATE HELPERS (ANGLE-ROBUST)
# =============================================================================

def is_finger_extended_robust(lm, tip_id, pip_id,
                               palm_up, palm_right, palm_len, wrist_pt):
    """
    Angle-robust finger extension test.

    Projects both the tip and PIP joint into the palm frame, then checks
    whether the tip is further along the palm's "up" axis than the PIP.

    This works regardless of whether the hand is:
        • upright (classic portrait)
        • tilted 45°
        • horizontal (pointing left / right)
        • held at oblique angles to the camera

    A finger is extended when:
        tip_along > pip_along + FINGER_CURL_THRESH

    The threshold prevents noise near the boundary from toggling state.
    """
    tip_along, _ = project_palm(tip_id,  lm, palm_up, palm_right, palm_len, wrist_pt)
    pip_along, _ = project_palm(pip_id,  lm, palm_up, palm_right, palm_len, wrist_pt)
    return (tip_along - pip_along) > Config.FINGER_CURL_THRESH


def is_thumb_extended_robust(lm, handedness,
                              palm_up, palm_right, palm_len, wrist_pt):
    """
    Angle-robust thumb extension test.

    The thumb folds *sideways* relative to the palm, so we test its
    projection along the palm_right axis rather than palm_up.

    For a right hand the thumb extends toward *negative* lateral
    (toward the index finger side in screen space, after mirroring).
    For a left hand it extends toward *positive* lateral.

    A shorter palm-axis comparison (THUMB_MCP→TIP) is used because the
    thumb metacarpal does not follow the same "up" geometry as fingers.
    """
    tip_lat  = project_palm(THUMB_TIP,  lm, palm_up, palm_right, palm_len, wrist_pt)[1]
    mcp_lat  = project_palm(THUMB_MCP,  lm, palm_up, palm_right, palm_len, wrist_pt)[1]
    delta    = tip_lat - mcp_lat   # positive = tip moved toward pinky side

    if handedness == "Right":
        return delta < -0.15   # thumb extends away from pinky (negative lateral)
    else:
        return delta >  0.15


def get_finger_states(lm, handedness, palm_up, palm_right, palm_len, wrist_pt):
    """
    Return the extension state of all five fingers using the palm-relative basis.

    Returns:
        dict — keys: thumb, index, middle, ring, pinky  |  values: bool
    """
    return {
        "thumb":  is_thumb_extended_robust(
                      lm, handedness, palm_up, palm_right, palm_len, wrist_pt),
        "index":  is_finger_extended_robust(
                      lm, INDEX_TIP,  INDEX_PIP,
                      palm_up, palm_right, palm_len, wrist_pt),
        "middle": is_finger_extended_robust(
                      lm, MIDDLE_TIP, MIDDLE_PIP,
                      palm_up, palm_right, palm_len, wrist_pt),
        "ring":   is_finger_extended_robust(
                      lm, RING_TIP,   RING_PIP,
                      palm_up, palm_right, palm_len, wrist_pt),
        "pinky":  is_finger_extended_robust(
                      lm, PINKY_TIP,  PINKY_PIP,
                      palm_up, palm_right, palm_len, wrist_pt),
    }


# =============================================================================
# SECTION 6 — SPATIAL DISAMBIGUATION HELPERS (ANGLE-ROBUST)
# =============================================================================

def thumb_direction(lm, palm_up, palm_right, palm_len, wrist_pt):
    """
    Determine if the thumb points UP or DOWN relative to the palm frame.

    Uses the palm-up axis projection so this works even when the whole
    hand is tilted sideways.

    Returns:
        "up"   — thumb clearly above wrist along palm axis
        "down" — thumb clearly below wrist along palm axis
        None   — ambiguous (within dead-zone)
    """
    along, _ = project_palm(THUMB_TIP, lm, palm_up, palm_right, palm_len, wrist_pt)
    # `along` > 0 means tip is on the "finger" side of the wrist.
    # For thumbs-up the tip is above wrist in palm space (along > deadzone).
    # For thumbs-down the tip is below wrist (along < -deadzone).
    if along >  Config.THUMB_UP_DOWN_DEADZONE:
        return "up"
    if along < -Config.THUMB_UP_DOWN_DEADZONE:
        return "down"
    return None


def index_direction(lm, palm_up, palm_right, palm_len, wrist_pt):
    """
    Determine if the index finger points RIGHT, LEFT, or UP (vertically)
    relative to the palm's lateral axis.

    Comparing the index-tip lateral projection against a threshold lets
    us distinguish a pointing gesture from a "stand" gesture regardless
    of overall hand tilt.

    Returns:
        "right" — index tip clearly to the right of the wrist in palm frame
        "left"  — index tip clearly to the left
        "up"    — tip is mostly along the palm_up axis (vertical stand)
    """
    _, lateral = project_palm(INDEX_TIP, lm, palm_up, palm_right, palm_len, wrist_pt)
    if lateral >  Config.POINT_HORIZONTAL_THRESH:
        return "right"
    if lateral < -Config.POINT_HORIZONTAL_THRESH:
        return "left"
    return "up"


def spread_score(lm):
    """
    Measure how widely the fingertips are spread apart (screen-space).

    Uses raw screen coordinates (not palm-relative) because we care about
    the absolute separation of fingertips as seen by the camera, which is
    a reliable discriminator between Open Palm and Spread Fingers even when
    the hand is angled.

    Returns:
        float — higher = more spread, lower = fingers together
    """
    coords = np.array([[lm[t].x, lm[t].y] for t in ALL_TIPS])
    gaps   = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(4)]
    return float(np.mean(gaps))


# =============================================================================
# SECTION 7 — SHAKE DETECTOR
# =============================================================================

class ShakeDetector:
    """
    Detects rapid hand shaking by counting X-axis direction reversals
    in a rolling time window.

    The shake detector deliberately uses raw screen-space X (wrist position)
    rather than palm-relative coordinates, because shaking is a gross motor
    movement of the entire hand across the frame, not a fine articulation
    relative to the palm.
    """

    def __init__(self):
        self._buf = collections.deque(maxlen=150)

    def update(self, wrist_x: float, timestamp: float) -> bool:
        self._buf.append((timestamp, wrist_x))

        cutoff = timestamp - Config.SHAKE_WINDOW_SEC
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 8:
            return False

        xs = [p[1] for p in self._buf]
        velocities = [xs[i+1] - xs[i] for i in range(len(xs) - 1)]

        reversals   = 0
        amplitudes  = []
        last_dir    = 0
        seg_min     = xs[0]
        seg_max     = xs[0]

        for i, v in enumerate(velocities):
            if abs(v) < 0.003:
                continue
            direction = 1 if v > 0 else -1
            seg_min = min(seg_min, xs[i])
            seg_max = max(seg_max, xs[i])
            if last_dir != 0 and direction != last_dir:
                reversals += 1
                amplitudes.append(seg_max - seg_min)
                seg_min = xs[i]
                seg_max = xs[i]
            last_dir = direction

        if reversals < Config.SHAKE_MIN_REVERSALS:
            return False
        avg_amp = float(np.mean(amplitudes)) if amplitudes else 0.0
        return avg_amp >= Config.SHAKE_AMPLITUDE_THRESH

    def reset(self):
        self._buf.clear()


# =============================================================================
# SECTION 8 — GESTURE CLASSIFIER (ANGLE-ROBUST)
# =============================================================================

def classify_gesture(lm, handedness, is_shaking: bool) -> str:
    """
    Classify the current hand pose into one of the ten CANIS gestures.

    All finger-extension and directional tests are now performed in the
    *palm-relative coordinate frame* (see build_hand_basis / project_palm),
    making recognition robust to:
        • Hand tilt (0–360° roll)
        • Lateral / oblique viewing angles
        • Distance variation (scale-normalised)

    Priority / disambiguation order:
        1.  [1 1 1 1 1] + shaking         → PALM_SHAKE
        2.  [1 1 1 1 1] + high spread     → SPREAD_FINGERS
        3.  [1 1 1 1 1] + low  spread     → OPEN_PALM
        4.  [0 0 0 0 0]                   → FIST
        5.  [1 0 0 0 0] + thumb up        → THUMBS_UP
        6.  [1 0 0 0 0] + thumb down      → THUMBS_DOWN
        7.  [0 1 0 0 0] + index right     → POINT_RIGHT
        8.  [0 1 0 0 0] + index left      → POINT_LEFT
        9.  [0 1 0 0 0] + index up        → ONE_FINGER
        10. [0 1 1 0 0]                   → TWO_FINGERS
        *   anything else                 → UNKNOWN
    """
    # ── Build the palm coordinate frame ───────────────────────────────────
    palm_up, palm_right, palm_len, wrist_pt = build_hand_basis(lm)

    # ── Get all five finger states in the palm frame ───────────────────────
    fs = get_finger_states(lm, handedness, palm_up, palm_right, palm_len, wrist_pt)
    t  = fs["thumb"];  i  = fs["index"];  m  = fs["middle"]
    r  = fs["ring"];   p  = fs["pinky"]

    pattern = [int(t), int(i), int(m), int(r), int(p)]

    # ── All five extended ─────────────────────────────────────────────────
    if pattern == [1, 1, 1, 1, 1]:
        if is_shaking:
            return G_EXCITED
        score = spread_score(lm)
        return G_SPREAD if score > Config.SPREAD_SCORE_THRESH else G_OPEN_PALM

    # ── All curled ────────────────────────────────────────────────────────
    if pattern == [0, 0, 0, 0, 0]:
        return G_FIST

    # ── Only thumb ────────────────────────────────────────────────────────
    if pattern == [1, 0, 0, 0, 0]:
        d = thumb_direction(lm, palm_up, palm_right, palm_len, wrist_pt)
        if d == "up":   return G_THUMBS_UP
        if d == "down": return G_THUMBS_DOWN
        return G_UNKNOWN

    # ── Only index ────────────────────────────────────────────────────────
    if pattern == [0, 1, 0, 0, 0]:
        d = index_direction(lm, palm_up, palm_right, palm_len, wrist_pt)
        if d == "right": return G_POINT_RIGHT
        if d == "left":  return G_POINT_LEFT
        return G_ONE_FINGER

    # ── Index + middle ────────────────────────────────────────────────────
    if pattern == [0, 1, 1, 0, 0]:
        return G_TWO_FINGERS

    return G_UNKNOWN


# =============================================================================
# SECTION 9 — ENHANCED STABILITY BUFFER
# =============================================================================

class StabilityBuffer:
    """
    Enhanced stability system with tolerance support.

    Keeps a sliding window of the last N frames and confirms a gesture
    only when the dominant gesture appears in at least
    (STABILITY_TOLERANCE × N) of those frames.
    """

    def __init__(self,
                 required: int   = Config.STABILITY_REQUIRED,
                 tolerance: float = Config.STABILITY_TOLERANCE):
        self.required  = required
        self.tolerance = tolerance
        self._window   = collections.deque(maxlen=required)

    @property
    def count(self):
        if not self._window:
            return 0
        return sum(1 for g in self._window if g == self._dominant())

    def _dominant(self):
        counts = {}
        for g in self._window:
            if g != G_UNKNOWN:
                counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get) if counts else G_UNKNOWN

    def update(self, raw_gesture: str):
        self._window.append(raw_gesture)
        if len(self._window) < self.required:
            return None

        dominant = self._dominant()
        if dominant == G_UNKNOWN:
            return None

        match_count = sum(1 for g in self._window if g == dominant)
        if match_count >= self.required * self.tolerance:
            return dominant
        return None

    def reset(self):
        self._window.clear()


# =============================================================================
# SECTION 10 — COOLDOWN SYSTEM
# =============================================================================

class CooldownTimer:
    """
    Prevents the same CANIS state from being emitted more than once
    within the cooldown window.
    """

    def __init__(self, cooldown_seconds: float = Config.COOLDOWN_SECONDS):
        self.cooldown    = cooldown_seconds
        self._last_state = None
        self._last_time  = 0.0

    def can_send(self, state: str) -> bool:
        now             = time.time()
        new_state       = (state != self._last_state)
        cooldown_passed = (now - self._last_time) >= self.cooldown
        if new_state or cooldown_passed:
            self._last_state = state
            self._last_time  = now
            return True
        return False

    def reset(self):
        self._last_state = None
        self._last_time  = 0.0


# =============================================================================
# SECTION 11 — STATE MACHINE
# =============================================================================

class CanisStateMachine:
    """
    Tracks the active CANIS robot state.
    State only changes when a confirmed gesture maps to a different state.
    """

    def __init__(self):
        self.current_state   = S_NONE
        self.current_gesture = G_UNKNOWN
        self._frame_counter  = 0

    def set_frame(self, n: int):
        self._frame_counter = n

    def transition(self, confirmed_gesture: str) -> bool:
        new_state = GESTURE_TO_STATE.get(confirmed_gesture, S_NONE)
        if new_state == S_NONE or new_state == self.current_state:
            return False
        self.current_state   = new_state
        self.current_gesture = confirmed_gesture
        return True

    def log(self, frame_no: int):
        print(
            f"[FRAME {frame_no:>5}]  "
            f"GESTURE: {self.current_gesture:<16}  →  "
            f"STATE: {self.current_state}"
        )

    def reset(self):
        self.current_state   = S_NONE
        self.current_gesture = G_UNKNOWN


# =============================================================================
# SECTION 12 — COMMAND OUTPUT
# =============================================================================

def send_serial_command(state: str, port: str = None, baudrate: int = 115200):
    ts = time.strftime("%H:%M:%S")
    print(f"    [{ts}] [SERIAL MOCK] → {state}")
    # import serial
    # with serial.Serial(port, baudrate, timeout=1) as ser:
    #     ser.write((state + '\n').encode('utf-8'))


def send_wifi_command(state: str, ip: str = "192.168.1.100", udp_port: int = 5005):
    ts = time.strftime("%H:%M:%S")
    print(f"    [{ts}] [WIFI MOCK]   → {state}  (target {ip}:{udp_port})")
    # import socket
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.sendto((state + '\n').encode('utf-8'), (ip, udp_port))
    # sock.close()


# =============================================================================
# SECTION 13 — COMMAND LOG
# =============================================================================

class CommandLog:
    """Rolling history of sent commands rendered as a faded sidebar."""

    def __init__(self, max_entries: int = 6):
        self._entries = collections.deque(maxlen=max_entries)

    def add(self, gesture: str, state: str):
        ts    = time.strftime("%H:%M:%S")
        color = GESTURE_COLOR.get(gesture, GESTURE_COLOR[G_UNKNOWN])
        self._entries.appendleft((ts, gesture, state, color))

    def draw(self, frame):
        if not self._entries:
            return
        h, w    = frame.shape[:2]
        n       = len(self._entries)
        panel_w = 310
        panel_h = n * 34 + 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 8),
                               (w - 8, 8 + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for idx, (ts, gesture, state, color) in enumerate(self._entries):
            y     = 34 + idx * 34
            alpha = max(0.25, 1.0 - idx * 0.16)
            faded = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, f"{gesture}  →  {state}",
                        (w - panel_w, y), Config.FONT, 0.50, faded, 1, cv2.LINE_AA)
            cv2.putText(frame, ts,
                        (w - 75, y), Config.FONT, 0.38, (90, 90, 90), 1, cv2.LINE_AA)


# =============================================================================
# SECTION 14 — HUD DRAWING
# =============================================================================

def draw_hand_landmarks(frame, hand_lm, mp_draw, mp_hands):
    mp_draw.draw_landmarks(
        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(230, 230, 230), thickness=2),
    )


def draw_palm_axes(frame, lm, palm_up, palm_right, palm_len, wrist_pt, w, h):
    """
    Visualise the palm coordinate frame as two coloured arrows on screen.

    GREEN arrow  = palm_up  (finger direction)
    BLUE  arrow  = palm_right (lateral direction)

    This makes it easy to verify the basis is tracking correctly as the
    hand rotates, helping with threshold tuning.
    """
    # Scale the arrows to a visible size on screen
    scale = palm_len * min(w, h) * 0.6

    origin = (int(wrist_pt[0] * w), int(wrist_pt[1] * h))

    # palm_up arrow (green)
    tip_up = (
        int((wrist_pt[0] + palm_up[0] * palm_len * 1.2) * w),
        int((wrist_pt[1] + palm_up[1] * palm_len * 1.2) * h),
    )
    cv2.arrowedLine(frame, origin, tip_up, (0, 220, 80), 2,
                    tipLength=0.25, line_type=cv2.LINE_AA)

    # palm_right arrow (blue)
    tip_right = (
        int((wrist_pt[0] + palm_right[0] * palm_len * 0.8) * w),
        int((wrist_pt[1] + palm_right[1] * palm_len * 0.8) * h),
    )
    cv2.arrowedLine(frame, origin, tip_right, (255, 100, 50), 2,
                    tipLength=0.25, line_type=cv2.LINE_AA)


def draw_finger_bar(frame, lm, handedness, palm_up, palm_right, palm_len, wrist_pt):
    """
    Bottom strip: five coloured circles (T I M R P) showing finger states.
    Now uses the angle-robust palm-basis extension checks.
    """
    h, w = frame.shape[:2]
    fs     = get_finger_states(lm, handedness, palm_up, palm_right, palm_len, wrist_pt)
    labels = ["T", "I", "M", "R", "P"]
    states = [fs["thumb"], fs["index"], fs["middle"], fs["ring"], fs["pinky"]]

    for idx, (label, ext) in enumerate(zip(labels, states)):
        color = (0, 210, 80) if ext else (45, 45, 45)
        cx = 32 + idx * 50
        cy = h - 32
        cv2.circle(frame, (cx, cy), 20, color, -1)
        cv2.circle(frame, (cx, cy), 20, (180, 180, 180), 1)
        cv2.putText(frame, label, (cx - 8, cy + 6),
                    Config.FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_stability_bar(frame, buf: StabilityBuffer):
    filled   = buf.count
    total    = buf.required
    fraction = min(filled / max(total, 1), 1.0)
    bx, by   = 16, 178
    bw, bh   = 220, 13
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill_w = int(bw * fraction)
    if fill_w > 0:
        bar_color = (0, 200, 70) if fraction >= 1.0 else (0, 130, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bh), bar_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
    cv2.putText(frame, f"Stability  {filled}/{total}",
                (bx, by - 4), Config.FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


def draw_hud(frame, raw_gesture, confirmed_gesture,
             canis_state, stability_buf, frame_no, fps,
             is_shaking, shake_det_running):
    h, w    = frame.shape[:2]
    display = confirmed_gesture if confirmed_gesture else raw_gesture
    color   = GESTURE_COLOR.get(display, GESTURE_COLOR[G_UNKNOWN])

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (258, 210), (18, 18, 18), -1)
    cv2.addWeighted(overlay, Config.PANEL_ALPHA, frame,
                    1 - Config.PANEL_ALPHA, 0, frame)

    cv2.putText(frame, f"raw: {raw_gesture}", (16, 30),
                Config.FONT, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(frame, "Gesture:", (16, 58),
                Config.FONT, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, display, (16, 90),
                Config.FONT, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "State:", (16, 118),
                Config.FONT, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, canis_state, (16, 152),
                Config.FONT, 0.82, (50, 220, 255), 2, cv2.LINE_AA)

    draw_stability_bar(frame, stability_buf)

    if shake_det_running and is_shaking:
        cv2.putText(frame, "~ SHAKE DETECTED ~", (16, 206),
                    Config.FONT, 0.44, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS {fps:>4.0f}   Frame {frame_no}",
                (w - 220, 26), Config.FONT, 0.52, (140, 140, 140), 1, cv2.LINE_AA)

    # Mode indicator (angle-robust badge)
    cv2.putText(frame, "[ANGLE-ROBUST]",
                (w - 160, h - 26), Config.FONT, 0.38, (0, 180, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q = quit   R = reset   S = screenshot",
                (w - 320, h - 12), Config.FONT, 0.40, (80, 80, 80), 1, cv2.LINE_AA)


# =============================================================================
# SECTION 15 — MAIN LOOP
# =============================================================================

def main():
    """
    Entry point.

    Frame-by-frame processing:
        1.  Capture + flip frame
        2.  Run MediaPipe inference
        3.  Build palm coordinate basis (angle-robust)
        4.  Update ShakeDetector with wrist X
        5.  Classify raw gesture using palm-relative coordinates
        6.  Feed into StabilityBuffer
        7.  If confirmed → CanisStateMachine
        8.  If state changed → CooldownTimer → send_serial_command()
        9.  Draw skeleton + palm axes + HUD + command log
        10. Handle keyboard input
    """
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = Config.MAX_HANDS,
        min_detection_confidence = Config.DETECTION_CONF,
        min_tracking_confidence  = Config.TRACKING_CONF,
    )

    cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (index {Config.WEBCAM_INDEX}). "
            "Check your camera connection or adjust WEBCAM_INDEX in Config."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    stability_buf = StabilityBuffer()
    cooldown      = CooldownTimer()
    state_machine = CanisStateMachine()
    shake_det     = ShakeDetector()
    cmd_log       = CommandLog(max_entries=6)

    frame_no     = 0
    prev_time    = time.time()
    raw_gesture  = G_UNKNOWN
    is_shaking   = False
    hand_visible = False

    print("=" * 65)
    print("  CANIS Gesture Control  v3  [ANGLE-ROBUST]")
    print(f"  Stability : {Config.STABILITY_REQUIRED} frames "
          f"({int(Config.STABILITY_TOLERANCE*100)}% tolerance)")
    print(f"  Cooldown  : {Config.COOLDOWN_SECONDS}s")
    print("  Basis     : Palm-relative coordinate frame (scale+angle invariant)")
    print("  Gestures  : 10 gestures, works from multiple viewing angles")
    print("  Keys      : Q=quit  R=reset  S=screenshot")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed — exiting.")
            break

        frame_no += 1
        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        confirmed_gesture = None
        hand_visible      = bool(results.multi_hand_landmarks)

        if hand_visible:
            hand_lm    = results.multi_hand_landmarks[0]
            lm         = hand_lm.landmark
            handedness = results.multi_handedness[0].classification[0].label

            # ── Build angle-robust palm basis ─────────────────────────────
            palm_up, palm_right, palm_len, wrist_pt = build_hand_basis(lm)

            # ── Draw skeleton ──────────────────────────────────────────────
            draw_hand_landmarks(frame, hand_lm, mp_draw, mp_hands)

            # ── Draw palm coordinate axes (green=up, blue=right) ──────────
            draw_palm_axes(frame, lm, palm_up, palm_right, palm_len, wrist_pt, w, h)

            # ── Draw finger indicators ─────────────────────────────────────
            draw_finger_bar(frame, lm, handedness,
                            palm_up, palm_right, palm_len, wrist_pt)

            # ── Shake detector (screen-space wrist X) ─────────────────────
            is_shaking = shake_det.update(lm[WRIST].x, now)

            # ── Classify using palm-relative coordinates ───────────────────
            raw_gesture = classify_gesture(lm, handedness, is_shaking)

            # ── Stability buffer ───────────────────────────────────────────
            confirmed_gesture = stability_buf.update(raw_gesture)

            # ── State machine ──────────────────────────────────────────────
            if confirmed_gesture is not None:
                state_machine.set_frame(frame_no)
                changed = state_machine.transition(confirmed_gesture)
                if changed:
                    new_state = state_machine.current_state
                    if cooldown.can_send(new_state):
                        state_machine.log(frame_no)
                        cmd_log.add(confirmed_gesture, new_state)
                        send_serial_command(new_state)
                        # send_wifi_command(new_state)

        else:
            stability_buf.reset()
            shake_det.reset()
            is_shaking  = False
            raw_gesture = G_UNKNOWN
            cooldown._last_state = None

        draw_hud(
            frame             = frame,
            raw_gesture       = raw_gesture,
            confirmed_gesture = confirmed_gesture,
            canis_state       = state_machine.current_state,
            stability_buf     = stability_buf,
            frame_no          = frame_no,
            fps               = fps,
            is_shaking        = is_shaking,
            shake_det_running = hand_visible,
        )

        cmd_log.draw(frame)
        cv2.imshow(Config.WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[EXIT] Quit key pressed.")
            break
        elif key == ord('r'):
            stability_buf.reset()
            cooldown.reset()
            state_machine.reset()
            shake_det.reset()
            raw_gesture = G_UNKNOWN
            is_shaking  = False
            print("[RESET] All components cleared.")
        elif key == ord('s'):
            fname = f"canis_v3_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[SCREENSHOT] → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("CANIS v3 stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
# =============================================================================
#  CANIS — Real-Time Hand Gesture Control for Robotic Dog  v3
#  ─────────────────────────────────────────────────────────
#  10 Gestures  |  Enhanced Stability  |  State Machine  |  Cooldown
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
    WEBCAM_INDEX        = 1          # 0 = default camera
    FRAME_WIDTH         = 1280
    FRAME_HEIGHT        = 720

    # ── MediaPipe ──────────────────────────────────────────────────────────
    MAX_HANDS           = 1          # only track one hand
    DETECTION_CONF      = 0.80       # initial detection confidence
    TRACKING_CONF       = 0.70       # tracking confidence

    # ── Stability (Enhanced) ───────────────────────────────────────────────
    # A gesture must appear consistently for this many frames before firing.
    # Raising this value makes detection more robust but slightly slower.
    STABILITY_REQUIRED  = 8          # frames (was 5 — now stricter)

    # Minimum fraction of STABILITY_REQUIRED frames that must agree.
    # e.g. 0.875 means 7/8 frames must match (allows 1 noisy frame).
    STABILITY_TOLERANCE = 0.875      # fraction of frames that must agree

    # ── Cooldown ───────────────────────────────────────────────────────────
    COOLDOWN_SECONDS    = 1.2        # seconds before same state fires again

    # ── Gesture thresholds ─────────────────────────────────────────────────
    THUMB_UP_DOWN_DEADZONE   = 0.06  # normalised Y dead-zone for thumb direction
    POINT_HORIZONTAL_THRESH  = 0.10  # normalised X offset to detect left/right point
    SPREAD_SCORE_THRESH      = 0.10  # mean fingertip spacing to classify as spread

    # ── Shake detection ────────────────────────────────────────────────────
    SHAKE_WINDOW_SEC         = 1.2   # rolling window to measure oscillations
    SHAKE_MIN_REVERSALS      = 5     # X-direction reversals in window
    SHAKE_AMPLITUDE_THRESH   = 0.06  # min normalised X per reversal

    # ── HUD ────────────────────────────────────────────────────────────────
    FONT                = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_ALPHA         = 0.60       # overlay panel transparency
    WINDOW_TITLE        = "CANIS — Gesture Control v3"


# =============================================================================
# SECTION 2 — GESTURE & STATE DEFINITIONS
# =============================================================================

# ── Raw gesture labels ────────────────────────────────────────────────────
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

# ── CANIS robot state labels ──────────────────────────────────────────────
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

# ── Gesture → State mapping ───────────────────────────────────────────────
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

# ── BGR colours for HUD labels ────────────────────────────────────────────
GESTURE_COLOR = {
    G_OPEN_PALM:   (50,  230, 80),    # green
    G_THUMBS_UP:   (50,  220, 255),   # yellow
    G_THUMBS_DOWN: (50,  80,  220),   # blue
    G_FIST:        (60,  60,  210),   # red-blue
    G_ONE_FINGER:  (200, 200, 50),    # cyan-yellow
    G_POINT_RIGHT: (0,   160, 255),   # orange
    G_POINT_LEFT:  (255, 100, 50),    # deep orange
    G_TWO_FINGERS: (200, 50,  255),   # purple
    G_SPREAD:      (0,   255, 200),   # teal
    G_EXCITED:     (0,   200, 255),   # gold
    G_UNKNOWN:     (100, 100, 100),   # grey
}


# =============================================================================
# SECTION 3 — MEDIAPIPE LANDMARK INDICES
# =============================================================================
#
#  MediaPipe provides 21 landmarks per hand (0–20).
#  Coordinate system: X ∈ [0,1] left→right,  Y ∈ [0,1] top→bottom.
#
#  Finger extension rule:
#      tip.y < pip.y  →  finger is EXTENDED (tip is higher on screen)
#      tip.y > pip.y  →  finger is CURLED
#
#  Thumb exception:
#      Thumb folds sideways, so we compare X coordinates instead.

LM = mp.solutions.hands.HandLandmark   # shorthand enum

WRIST           = LM.WRIST

THUMB_TIP       = LM.THUMB_TIP
THUMB_IP        = LM.THUMB_IP
THUMB_MCP       = LM.THUMB_MCP

INDEX_TIP       = LM.INDEX_FINGER_TIP
INDEX_PIP       = LM.INDEX_FINGER_PIP
INDEX_MCP       = LM.INDEX_FINGER_MCP

MIDDLE_TIP      = LM.MIDDLE_FINGER_TIP
MIDDLE_PIP      = LM.MIDDLE_FINGER_PIP

RING_TIP        = LM.RING_FINGER_TIP
RING_PIP        = LM.RING_FINGER_PIP

PINKY_TIP       = LM.PINKY_TIP
PINKY_PIP       = LM.PINKY_PIP

# Ordered list used for spread_score calculation
ALL_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


# =============================================================================
# SECTION 4 — FINGER STATE HELPERS
# =============================================================================

def is_finger_extended(lm, tip_id, pip_id):
    """
    Return True if a non-thumb finger is extended.

    A finger is extended when its tip is ABOVE its PIP joint on screen,
    which means tip.y < pip.y (Y increases downward in image coordinates).

    Args:
        lm     : list of 21 NormalizedLandmark from MediaPipe
        tip_id : landmark index of the fingertip
        pip_id : landmark index of the PIP joint (first major knuckle)

    Returns:
        bool — True = extended, False = curled
    """
    return lm[tip_id].y < lm[pip_id].y


def is_thumb_extended(lm, handedness):
    """
    Return True if the thumb is extended (open, pointing away from palm).

    The thumb folds horizontally so we compare X coordinates:
        Right hand — thumb extends toward smaller X (leftward on mirrored frame)
        Left  hand — thumb extends toward larger  X (rightward on mirrored frame)

    Args:
        lm          : list of 21 NormalizedLandmark
        handedness  : "Right" or "Left" (corrected for mirrored frame)

    Returns:
        bool — True = thumb extended, False = thumb folded in
    """
    tip_x = lm[THUMB_TIP].x
    ip_x  = lm[THUMB_IP].x
    if handedness == "Right":
        return tip_x < ip_x
    return tip_x > ip_x


def get_finger_states(lm, handedness):
    """
    Return the extension state of all five fingers as a plain dict.

    Args:
        lm          : MediaPipe landmark list
        handedness  : "Right" or "Left"

    Returns:
        dict — keys: 'thumb','index','middle','ring','pinky'  values: bool
    """
    return {
        "thumb":  is_thumb_extended(lm, handedness),
        "index":  is_finger_extended(lm, INDEX_TIP,  INDEX_PIP),
        "middle": is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        "ring":   is_finger_extended(lm, RING_TIP,   RING_PIP),
        "pinky":  is_finger_extended(lm, PINKY_TIP,  PINKY_PIP),
    }


# =============================================================================
# SECTION 5 — SPATIAL DISAMBIGUATION HELPERS
# =============================================================================

def thumb_direction(lm):
    """
    Determine if the thumb (only digit extended) points UP or DOWN.

    Compares thumb tip Y against wrist Y with a dead-zone to avoid
    misclassification when the hand is horizontal.

    Returns:
        "up"   — thumb tip clearly above wrist
        "down" — thumb tip clearly below wrist
        None   — ambiguous (within dead-zone)
    """
    diff = lm[WRIST].y - lm[THUMB_TIP].y   # positive = tip above wrist
    if diff >  Config.THUMB_UP_DOWN_DEADZONE:
        return "up"
    if diff < -Config.THUMB_UP_DOWN_DEADZONE:
        return "down"
    return None


def index_direction(lm):
    """
    Determine if the index finger (only digit extended) is pointing
    horizontally LEFT, RIGHT, or vertically UP.

    Compares index tip X against wrist X.
    After mirroring, right on screen = dog's right.

    Returns:
        "right" — tip is significantly to the right of the wrist
        "left"  — tip is significantly to the left  of the wrist
        "up"    — tip is roughly above the wrist (vertical)
    """
    diff = lm[INDEX_TIP].x - lm[WRIST].x   # positive = tip to the right
    if diff >  Config.POINT_HORIZONTAL_THRESH:
        return "right"
    if diff < -Config.POINT_HORIZONTAL_THRESH:
        return "left"
    return "up"


def spread_score(lm):
    """
    Measure how widely the fingertips are spread apart.

    Computes the mean Euclidean distance between consecutive fingertip
    positions [thumb→index→middle→ring→pinky].

    Returns:
        float — higher = more spread (Spread Fingers gesture)
                 lower  = fingers together (Open Palm, Stop)
    """
    coords = np.array([[lm[t].x, lm[t].y] for t in ALL_TIPS])
    gaps   = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(4)]
    return float(np.mean(gaps))


# =============================================================================
# SECTION 6 — SHAKE DETECTOR
# =============================================================================

class ShakeDetector:
    """
    Detects rapid hand shaking by counting X-axis direction reversals
    in a rolling time window.

    Algorithm:
        1. Record (timestamp, wrist_x) for every frame.
        2. Drop readings older than SHAKE_WINDOW_SEC.
        3. Compute frame-to-frame X velocities.
        4. Count sign flips (reversals) in velocity signal.
        5. A shake is detected when reversals ≥ SHAKE_MIN_REVERSALS
           AND average reversal amplitude ≥ SHAKE_AMPLITUDE_THRESH.

    This is robust to normal hand motion because the amplitude threshold
    filters out small tremors.
    """

    def __init__(self):
        # Rolling deque stores (timestamp, wrist_x) tuples
        self._buf = collections.deque(maxlen=150)

    def update(self, wrist_x: float, timestamp: float) -> bool:
        """
        Feed the current wrist X position and return whether shaking.

        Args:
            wrist_x   : normalised wrist X (0–1) from MediaPipe
            timestamp : current time.time() value

        Returns:
            True if a shake is currently detected.
        """
        self._buf.append((timestamp, wrist_x))

        # Remove stale readings
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
                continue   # ignore micro-jitter
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
# SECTION 7 — GESTURE CLASSIFIER
# =============================================================================

def classify_gesture(lm, handedness, is_shaking: bool) -> str:
    """
    Classify the current hand pose into one of the ten CANIS gestures.

    Priority / disambiguation order:
        1.  [1 1 1 1 1] + shaking         → PALM_SHAKE   (Excited Mode)
        2.  [1 1 1 1 1] + high spread     → SPREAD_FINGERS (Attention)
        3.  [1 1 1 1 1] + low  spread     → OPEN_PALM      (Stop)
        4.  [0 0 0 0 0]                   → FIST           (Stay)
        5.  [1 0 0 0 0] + thumb up        → THUMBS_UP      (Walk)
        6.  [1 0 0 0 0] + thumb down      → THUMBS_DOWN    (Sit)
        7.  [0 1 0 0 0] + index right     → POINT_RIGHT    (Turn Right)
        8.  [0 1 0 0 0] + index left      → POINT_LEFT     (Turn Left)
        9.  [0 1 0 0 0] + index up        → ONE_FINGER     (Stand)
        10. [0 1 1 0 0]                   → TWO_FINGERS    (Move Mode)
        *   anything else                 → UNKNOWN

    Args:
        lm          : MediaPipe landmark list (21 NormalizedLandmark)
        handedness  : "Right" or "Left"
        is_shaking  : output from ShakeDetector.update()

    Returns:
        One of the G_* gesture constants.
    """
    fs = get_finger_states(lm, handedness)
    t  = fs["thumb"]
    i  = fs["index"]
    m  = fs["middle"]
    r  = fs["ring"]
    p  = fs["pinky"]

    # Shorthand list for readability
    pattern = [int(t), int(i), int(m), int(r), int(p)]

    # ── All five extended [1 1 1 1 1] ─────────────────────────────────────
    if pattern == [1, 1, 1, 1, 1]:
        if is_shaking:
            return G_EXCITED            # palm + shake → excited mode

        score = spread_score(lm)
        if score > Config.SPREAD_SCORE_THRESH:
            return G_SPREAD             # fingers fanned wide → attention
        return G_OPEN_PALM              # fingers together   → stop

    # ── All curled [0 0 0 0 0] ────────────────────────────────────────────
    if pattern == [0, 0, 0, 0, 0]:
        return G_FIST                   # → stay

    # ── Only thumb [1 0 0 0 0] ────────────────────────────────────────────
    if pattern == [1, 0, 0, 0, 0]:
        d = thumb_direction(lm)
        if d == "up":
            return G_THUMBS_UP          # → walk
        if d == "down":
            return G_THUMBS_DOWN        # → sit
        return G_UNKNOWN                # thumb horizontal → ignore

    # ── Only index [0 1 0 0 0] ────────────────────────────────────────────
    if pattern == [0, 1, 0, 0, 0]:
        d = index_direction(lm)
        if d == "right":
            return G_POINT_RIGHT        # → turn right
        if d == "left":
            return G_POINT_LEFT         # → turn left
        return G_ONE_FINGER             # vertical → stand

    # ── Index + middle [0 1 1 0 0] ────────────────────────────────────────
    if pattern == [0, 1, 1, 0, 0]:
        return G_TWO_FINGERS            # → move mode

    return G_UNKNOWN


# =============================================================================
# SECTION 8 — ENHANCED STABILITY BUFFER
# =============================================================================

class StabilityBuffer:
    """
    Enhanced stability system with tolerance support.

    Instead of requiring N *perfectly* consecutive identical frames,
    this buffer keeps a sliding window of the last N frames and confirms
    a gesture only when the dominant gesture appears in at least
    (STABILITY_TOLERANCE × N) of those frames.

    This allows up to one noisy/transitional frame per window without
    resetting the counter — making it noticeably more robust while still
    filtering genuine gesture changes.

    Example (N=8, tolerance=0.875 → needs 7/8 frames to agree):
        Window: [UP UP UP UP UP UP ?? UP]  →  UP confirmed  ✓
        Window: [UP UP DOWN UP UP UP UP UP] →  UP confirmed  ✓
        Window: [UP DOWN UP DOWN UP DOWN]  →  not confirmed  ✗
    """

    def __init__(self,
                 required: int   = Config.STABILITY_REQUIRED,
                 tolerance: float = Config.STABILITY_TOLERANCE):
        self.required  = required
        self.tolerance = tolerance
        self._window   = collections.deque(maxlen=required)

    @property
    def count(self):
        """How many frames in the window match the dominant gesture."""
        if not self._window:
            return 0
        dominant = self._dominant()
        return sum(1 for g in self._window if g == dominant)

    def _dominant(self):
        """Return the most common non-UNKNOWN gesture in the window."""
        counts = {}
        for g in self._window:
            if g != G_UNKNOWN:
                counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get) if counts else G_UNKNOWN

    def update(self, raw_gesture: str):
        """
        Add the latest raw gesture and check if a stable gesture is confirmed.

        Args:
            raw_gesture : G_* gesture string from classify_gesture()

        Returns:
            Confirmed gesture string if stable threshold met, else None.
        """
        self._window.append(raw_gesture)

        if len(self._window) < self.required:
            return None   # window not yet full

        dominant = self._dominant()
        if dominant == G_UNKNOWN:
            return None

        # Check whether dominant gesture appears in enough frames
        match_count = sum(1 for g in self._window if g == dominant)
        threshold   = self.required * self.tolerance

        if match_count >= threshold:
            return dominant

        return None

    def reset(self):
        self._window.clear()


# =============================================================================
# SECTION 9 — COOLDOWN SYSTEM
# =============================================================================

class CooldownTimer:
    """
    Prevents the same CANIS state from being emitted more than once
    within the cooldown window.

    A new emission is allowed when:
        • The state is DIFFERENT from the last emitted state, OR
        • The cooldown period has elapsed since the last emission.
    """

    def __init__(self, cooldown_seconds: float = Config.COOLDOWN_SECONDS):
        self.cooldown   = cooldown_seconds
        self._last_state = None
        self._last_time  = 0.0

    def can_send(self, state: str) -> bool:
        """
        Return True if the state should be emitted right now.

        Always updates internal tracking when returning True.
        """
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
# SECTION 10 — STATE MACHINE
# =============================================================================

class CanisStateMachine:
    """
    Tracks the active CANIS robot state.

    Rules:
        • State only changes when a new confirmed gesture maps to a
          DIFFERENT state than the current one.
        • UNKNOWN gestures have no effect.
        • Logs every valid transition to the console.
    """

    def __init__(self):
        self.current_state   = S_NONE
        self.current_gesture = G_UNKNOWN
        self._frame_counter  = 0       # used in log output

    def set_frame(self, n: int):
        self._frame_counter = n

    def transition(self, confirmed_gesture: str) -> bool:
        """
        Attempt a state transition based on a confirmed gesture.

        Returns:
            True  if the state actually changed (command should be sent)
            False if no change (same state or unmapped gesture)
        """
        new_state = GESTURE_TO_STATE.get(confirmed_gesture, S_NONE)
        if new_state == S_NONE:
            return False
        if new_state == self.current_state:
            return False

        self.current_state   = new_state
        self.current_gesture = confirmed_gesture
        return True

    def log(self, frame_no: int):
        """Print the structured [FRAME X] log line."""
        print(
            f"[FRAME {frame_no:>5}]  "
            f"GESTURE: {self.current_gesture:<16}  →  "
            f"STATE: {self.current_state}"
        )

    def reset(self):
        self.current_state   = S_NONE
        self.current_gesture = G_UNKNOWN


# =============================================================================
# SECTION 11 — COMMAND OUTPUT  (Serial / WiFi — mocked by default)
# =============================================================================

def send_serial_command(state: str, port: str = None, baudrate: int = 115200):
    """
    Transmit a CANIS state string to an ESP32 via USB Serial.

    MOCKED by default — prints to console.
    To enable real serial transmission:
        1. pip install pyserial
        2. Set `port` to your device path:
               Windows → 'COM3'
               Linux   → '/dev/ttyUSB0'
               macOS   → '/dev/cu.usbserial-0001'
        3. Uncomment the serial block below.

    The ESP32 Arduino sketch should read:
        String cmd = Serial.readStringUntil('\\n');
        cmd.trim();
        if (cmd == "CANIS_WALK") { ... }
    """
    ts = time.strftime("%H:%M:%S")
    print(f"    [{ts}] [SERIAL MOCK] → {state}")

    # ── Real serial (uncomment to activate) ───────────────────────────────
    # import serial
    # try:
    #     with serial.Serial(port, baudrate, timeout=1) as ser:
    #         ser.write((state + '\n').encode('utf-8'))
    # except Exception as e:
    #     print(f"    [SERIAL ERROR] {e}")


def send_wifi_command(state: str, ip: str = "192.168.1.100", udp_port: int = 5005):
    """
    Transmit a CANIS state string to an ESP32 via UDP over WiFi.

    MOCKED by default.  To enable:
        1. Set `ip` to the ESP32's IP address on your LAN.
        2. Uncomment the socket block below.

    The ESP32 should listen on the same UDP port and parse the packet body.
    """
    ts = time.strftime("%H:%M:%S")
    print(f"    [{ts}] [WIFI MOCK]   → {state}  (target {ip}:{udp_port})")

    # ── Real UDP (uncomment to activate) ──────────────────────────────────
    # import socket
    # try:
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     sock.sendto((state + '\n').encode('utf-8'), (ip, udp_port))
    #     sock.close()
    # except Exception as e:
    #     print(f"    [WIFI ERROR] {e}")


# =============================================================================
# SECTION 12 — COMMAND LOG (on-screen history)
# =============================================================================

class CommandLog:
    """
    Maintains a short rolling history of sent commands and renders them
    as a faded sidebar on the right edge of the frame.
    """

    def __init__(self, max_entries: int = 6):
        self._entries = collections.deque(maxlen=max_entries)

    def add(self, gesture: str, state: str):
        ts = time.strftime("%H:%M:%S")
        color = GESTURE_COLOR.get(gesture, GESTURE_COLOR[G_UNKNOWN])
        self._entries.appendleft((ts, gesture, state, color))

    def draw(self, frame):
        if not self._entries:
            return

        h, w = frame.shape[:2]
        n     = len(self._entries)
        panel_w = 310
        panel_h = n * 34 + 12

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 8),
                               (w - 8, 8 + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for idx, (ts, gesture, state, color) in enumerate(self._entries):
            y      = 34 + idx * 34
            alpha  = max(0.25, 1.0 - idx * 0.16)
            faded  = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, f"{gesture}  →  {state}",
                        (w - panel_w, y),
                        Config.FONT, 0.50, faded, 1, cv2.LINE_AA)
            cv2.putText(frame, ts,
                        (w - 75, y),
                        Config.FONT, 0.38, (90, 90, 90), 1, cv2.LINE_AA)


# =============================================================================
# SECTION 13 — HUD DRAWING
# =============================================================================

def draw_hand_landmarks(frame, hand_lm, mp_draw, mp_hands):
    """Render the 21-point skeleton and joint circles onto the frame."""
    mp_draw.draw_landmarks(
        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(230, 230, 230), thickness=2),
    )


def draw_finger_bar(frame, lm, handedness):
    """
    Bottom strip: five coloured circles (T I M R P) showing finger states.
    Green = extended, dark grey = curled.
    """
    h, w = frame.shape[:2]
    fs     = get_finger_states(lm, handedness)
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
    """
    Progress bar (top-left, below state label) showing stability progress.
    Turns green when the threshold is met.
    """
    filled   = buf.count
    total    = buf.required
    fraction = min(filled / max(total, 1), 1.0)

    bx, by   = 16, 178
    bw, bh   = 220, 13

    # Background
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)

    # Fill
    fill_w = int(bw * fraction)
    if fill_w > 0:
        bar_color = (0, 200, 70) if fraction >= 1.0 else (0, 130, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bh), bar_color, -1)

    # Border
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)

    # Label above bar
    cv2.putText(frame, f"Stability  {filled}/{total}",
                (bx, by - 4),
                Config.FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


def draw_hud(frame, raw_gesture, confirmed_gesture,
             canis_state, stability_buf, frame_no, fps,
             is_shaking, shake_det_running):
    """
    Render the full HUD:
        • Dark overlay panel (top-left)
        • Raw gesture (grey, small)
        • Confirmed/current gesture (coloured, large)
        • CANIS state
        • Stability bar
        • Shake indicator
        • FPS + frame counter (top-right)
        • Key hints (bottom-right)

    Args:
        frame             : OpenCV BGR frame (modified in place)
        raw_gesture       : latest raw detection (may be noisy)
        confirmed_gesture : stable confirmed gesture (or None)
        canis_state       : current CANIS_* state string
        stability_buf     : StabilityBuffer instance
        frame_no          : integer frame counter
        fps               : float frames-per-second
        is_shaking        : bool — shake currently detected
        shake_det_running : bool — hand is visible (shake detector active)
    """
    h, w   = frame.shape[:2]
    display = confirmed_gesture if confirmed_gesture else raw_gesture
    color   = GESTURE_COLOR.get(display, GESTURE_COLOR[G_UNKNOWN])

    # ── Semi-transparent panel ─────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (258, 210), (18, 18, 18), -1)
    cv2.addWeighted(overlay, Config.PANEL_ALPHA, frame,
                    1 - Config.PANEL_ALPHA, 0, frame)

    # ── Raw gesture (small, grey) ──────────────────────────────────────────
    cv2.putText(frame, f"raw: {raw_gesture}", (16, 30),
                Config.FONT, 0.42, (110, 110, 110), 1, cv2.LINE_AA)

    # ── Confirmed gesture (large, coloured) ───────────────────────────────
    cv2.putText(frame, "Gesture:", (16, 58),
                Config.FONT, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, display, (16, 90),
                Config.FONT, 0.85, color, 2, cv2.LINE_AA)

    # ── CANIS state ───────────────────────────────────────────────────────
    cv2.putText(frame, "State:", (16, 118),
                Config.FONT, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, canis_state, (16, 152),
                Config.FONT, 0.82, (50, 220, 255), 2, cv2.LINE_AA)

    # ── Stability bar ─────────────────────────────────────────────────────
    draw_stability_bar(frame, stability_buf)

    # ── Shake indicator ───────────────────────────────────────────────────
    if shake_det_running and is_shaking:
        cv2.putText(frame, "~ SHAKE DETECTED ~", (16, 206),
                    Config.FONT, 0.44, (0, 200, 255), 1, cv2.LINE_AA)

    # ── FPS + frame counter (top-right) ───────────────────────────────────
    cv2.putText(frame, f"FPS {fps:>4.0f}   Frame {frame_no}",
                (w - 220, 26),
                Config.FONT, 0.52, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Key hints (bottom-right) ──────────────────────────────────────────
    cv2.putText(frame, "Q = quit   R = reset   S = screenshot",
                (w - 320, h - 12),
                Config.FONT, 0.40, (80, 80, 80), 1, cv2.LINE_AA)


# =============================================================================
# SECTION 14 — MAIN LOOP
# =============================================================================

def main():
    """
    Entry point.

    Frame-by-frame processing:
        1.  Capture + flip frame
        2.  Run MediaPipe inference
        3.  Extract landmark list and handedness string
        4.  Update ShakeDetector with wrist X
        5.  Classify raw gesture
        6.  Feed into StabilityBuffer (enhanced tolerance window)
        7.  If confirmed → run through CanisStateMachine
        8.  If state changed → CooldownTimer → send_serial_command()
        9.  Draw skeleton + HUD + command log
        10. Handle keyboard input
    """

    # ── MediaPipe setup ───────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = Config.MAX_HANDS,
        min_detection_confidence = Config.DETECTION_CONF,
        min_tracking_confidence  = Config.TRACKING_CONF,
    )

    # ── Webcam setup ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (index {Config.WEBCAM_INDEX}). "
            "Check your camera connection or adjust WEBCAM_INDEX in Config."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    # ── Component initialisation ──────────────────────────────────────────
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

    # ── Startup banner ────────────────────────────────────────────────────
    print("=" * 65)
    print("  CANIS Gesture Control  v3")
    print(f"  Stability : {Config.STABILITY_REQUIRED} frames "
          f"({int(Config.STABILITY_TOLERANCE*100)}% tolerance)")
    print(f"  Cooldown  : {Config.COOLDOWN_SECONDS}s")
    print("  Gestures  : 10 gestures mapped to CANIS states")
    print("  Keys      : Q=quit  R=reset  S=screenshot")
    print("=" * 65)

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed — exiting.")
            break

        frame_no += 1
        frame     = cv2.flip(frame, 1)          # mirror for natural feel
        h, w      = frame.shape[:2]

        # FPS
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── MediaPipe inference ───────────────────────────────────────────
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

            # ── Step 1: Draw skeleton ──────────────────────────────────────
            draw_hand_landmarks(frame, hand_lm, mp_draw, mp_hands)

            # ── Step 2: Draw finger indicators at bottom ───────────────────
            draw_finger_bar(frame, lm, handedness)

            # ── Step 3: Update shake detector ──────────────────────────────
            is_shaking = shake_det.update(lm[WRIST].x, now)

            # ── Step 4: Classify raw gesture ───────────────────────────────
            raw_gesture = classify_gesture(lm, handedness, is_shaking)

            # ── Step 5: Stability buffer ───────────────────────────────────
            confirmed_gesture = stability_buf.update(raw_gesture)

            # ── Step 6: State machine transition ───────────────────────────
            if confirmed_gesture is not None:
                state_machine.set_frame(frame_no)
                changed = state_machine.transition(confirmed_gesture)

                # ── Step 7: Cooldown + command output ──────────────────────
                if changed:
                    new_state = state_machine.current_state
                    if cooldown.can_send(new_state):
                        state_machine.log(frame_no)
                        cmd_log.add(confirmed_gesture, new_state)
                        send_serial_command(new_state)
                        # send_wifi_command(new_state)  # uncomment for WiFi

        else:
            # Hand lost — reset all accumulators
            stability_buf.reset()
            shake_det.reset()
            is_shaking  = False
            raw_gesture = G_UNKNOWN
            cooldown._last_state = None   # allow immediate re-trigger on return

        # ── Step 8: HUD ───────────────────────────────────────────────────
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

        # ── Step 9: Command log sidebar ───────────────────────────────────
        cmd_log.draw(frame)

        # ── Step 10: Show frame ───────────────────────────────────────────
        cv2.imshow(Config.WINDOW_TITLE, frame)

        # ── Step 11: Keyboard handling ────────────────────────────────────
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

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("CANIS v3 stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
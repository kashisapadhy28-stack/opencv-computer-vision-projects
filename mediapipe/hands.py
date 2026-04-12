import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Legacy Solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 2. Open the default webcam (0)
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'ESC' to exit.")

# 3. Setup the Hands model
with mp_hands.Hands(
    model_complexity=0,              # 0 is faster, 1 is more accurate
    min_detection_confidence=0.5,    # Minimum confidence to detect a hand
    min_tracking_confidence=0.5      # Minimum confidence to keep tracking it
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 4. Prepare the image
        # MediaPipe expects RGB images, but OpenCV captures in BGR.
        # We also mark the image as not writeable for a slight performance boost.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 5. Process the image to find hands
        results = hands.process(image)

        # 6. Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
        # 7. Display the image
        # We flip it horizontally so it acts like a mirror (selfie-view)
        cv2.imshow('MediaPipe Hand Tracking', cv2.flip(image, 1))
        
        # 8. Exit condition: Press the 'ESC' key
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 9. Clean up resources
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"

# Initialize Pose Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Counter variables
counter = 0
stage = None  # "up" or "down"

def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to MP Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Run pose detection
    result = pose_landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # First person detected

        # Extract normalized coordinates
        shoulder = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER
        elbow = [landmarks[13].x, landmarks[13].y]    # LEFT_ELBOW
        wrist = [landmarks[15].x, landmarks[15].y]    # LEFT_WRIST

        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Display angle
        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Push‑up counter logic
        if angle > 160:
            stage = "up"
        if angle < 120 and stage == "up":
            stage = "down"
            counter += 1
            print(f"Rep Count: {counter}")

    # Draw counter box
    cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(frame, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Pushup Checker", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
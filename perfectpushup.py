import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = "pose_landmarker_heavy.task"

# Initialize Pose Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Global variables
counter = 0
stage = None  # "up" or "down"

def calculate_angle(a, b, c):
    """Calculates the angle between three points (Shoulder, Elbow, Wrist)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_skeleton(frame, landmarks):
    """Draws lines between joints for debugging visibility."""
    connections = [
        (11, 13), (13, 15), # Left Arm
        (12, 14), (14, 16), # Right Arm
        (11, 23), (12, 24), # Shoulders to Hips
        (23, 25), (25, 27), # Left Leg
        (24, 26), (26, 28), # Right Leg
        (11, 12), (23, 24)  # Torso
    ]
    h, w, _ = frame.shape
    for connection in connections:
        start_idx, end_idx = connection
        start_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
    
    for i in [11, 12, 13, 14, 15, 16, 23, 24]:
        pt = (int(landmarks[i].x * w), int(landmarks[i].y * h))
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

# Start Video Capture
cap = cv2.VideoCapture(0)
# 1. Create a named window that can be resized
cv2.namedWindow("Perfect Pushup Tracker", cv2.WINDOW_NORMAL)

# 2. Set the window to a larger resolution (e.g., 1280x720 or 1920x1080)
cv2.resizeWindow("Perfect Pushup Tracker", 1920, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect pose landmarks
    result = pose_landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Process detections if they exist
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]
        
        # 1. Draw Debug Skeleton
        draw_skeleton(frame, landmarks)

        # 2. Get Landmark Coordinates
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow    = [landmarks[13].x, landmarks[13].y]
        wrist    = [landmarks[15].x, landmarks[15].y]
        hip      = [landmarks[23].x, landmarks[23].y]
        ankle    = [landmarks[27].x, landmarks[27].y]
        if landmarks[13].visibility > 0.65: 
            pushup_angle = calculate_angle(shoulder, elbow, wrist)
        else:
    # Stay in the previous state to prevent "ghost" reps
            pass

        # 3. Calculate Geometry
        pushup_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle   = calculate_angle(shoulder, hip, ankle)

        # 4. Display Angles on Screen
        h, w, _ = frame.shape
        elbow_px = (int(elbow[0] * w), int(elbow[1] * h))
        cv2.putText(frame, f"{int(pushup_angle)}deg", elbow_px, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Push-up Counting & Form Logic
        # Back Alignment Check
        if back_angle < 150:
            cv2.putText(frame, "FIX BACK!", (250, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "FORM OK", (250, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # State Machine for Rep Counting
        if pushup_angle > 160:
            stage = "up"
        
        if pushup_angle <= 90 and stage == "up":
            if back_angle >= 150:
                stage = "down"
                counter += 1
                print(f"Rep Added! Total: {counter}")
            else:
                print("Bad Form: Rep not counted.")

    # 6. UI: Rep Counter Box
    cv2.rectangle(frame, (0,0), (180,80), (245,117,16), -1)
    cv2.putText(frame, 'REPS', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.putText(frame, str(counter), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    # Show Output
    cv2.imshow("Perfect Pushup Tracker", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
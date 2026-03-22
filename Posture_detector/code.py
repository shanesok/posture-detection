
import cv2
import mediapipe as mp
import time
import numpy as np
import os
import argparse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- MediaPipe Components ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2

# --- Constants for Configuration (Using Relative Paths) ---
MODEL_PATH = "pose_landmarker_heavy.task"
GOOD_POSTURE_PATHS = [
    r"datasets/userS_20251126_224347/good_posture/landmarks_good.csv",
    r"datasets/user_20251201_122340/good_posture/landmarks_good.csv",
    r"datasets/userk_20251201_151834/good_posture/landmarks_good.csv",
    r"datasets/users_20251201_160317/good_posture/landmarks_good.csv"
]
BAD_POSTURE_PATHS = [
    r"datasets/userS_20251126_224347/bad_posture/landmarks_bad.csv",
    r"datasets/user_20251201_122340/bad_posture/landmarks_bad.csv",
    r"datasets/userk_20251201_151834/bad_posture/landmarks_bad.csv",
    r"datasets/users_20251201_160317/bad_posture/landmarks_bad.csv"
]
# Define the specific landmarks to use for classification
# 0: nose, 11-16: shoulders, elbows, wrists, 23-24: hips
USED_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]


def setup_arg_parser():
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Posture estimation live feed using MediaPipe Pose Landmarker.')
    parser.add_argument('-dis_fps', '--display_fps', action="store_false", help='Disable FPS display on the live feed.')
    parser.add_argument('-dis_landmarks', '--display_pose_landmarkers', action="store_false", help='Disable pose landmark display on the live feed.')
    return parser.parse_args()


def train_knn_model(good_posture_csvs, bad_posture_csvs, used_landmark_indices):
    """
    Loads data, splits it for training/testing, trains a KNN model,
    displays a confusion matrix, and returns the trained model and scaler.
    """
    good_dfs = []
    for csv_path in good_posture_csvs:
        if not os.path.exists(csv_path):
            print(f"ERROR: Good posture CSV file not found at '{csv_path}'. Please check the path.")
            return None, None
        good_dfs.append(pd.read_csv(csv_path))

    bad_dfs = []
    for csv_path in bad_posture_csvs:
        if not os.path.exists(csv_path):
            print(f"ERROR: Bad posture CSV file not found at '{csv_path}'. Please check the path.")
            return None, None
        bad_dfs.append(pd.read_csv(csv_path))

    good_df = pd.concat(good_dfs, ignore_index=True)
    bad_df = pd.concat(bad_dfs, ignore_index=True)
    print(f"Loaded {len(good_df)} good posture samples and {len(bad_df)} bad posture samples.")

    feature_cols = [f'landmark_{i}_{axis}' for i in used_landmark_indices for axis in ['x', 'y', 'z']]
    
    X = pd.concat([good_df[feature_cols], bad_df[feature_cols]], ignore_index=True)
    y = np.concatenate([np.ones(len(good_df)), np.zeros(len(bad_df))])

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    # Train model
    knn = KNeighborsClassifier(n_neighbors=6, metric='manhattan').fit(X_train_scaled, y_train)
    print(f"KNN model trained on {len(X_train)} samples.")

    # Evaluate and display confusion matrix
    print("Evaluating model performance on the test set...")
    y_pred = knn.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad Posture', 'Good Posture'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("KNN Model - Posture Classification Accuracy")
    print("Displaying confusion matrix. Close the plot window to start the live feed.")
    plt.show()

    return knn, scaler


def setup_landmarker(model_path):
    """Configures and creates the PoseLandmarker options."""
    if not os.path.exists(model_path):
        print(f"ERROR: The model file was not found at the path: '{model_path}'")
        return None
    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate='GPU'),
            running_mode=VisionRunningMode.VIDEO
        )
        return options
    except Exception as e:
        print(f"Error creating PoseLandmarkerOptions: {e}")
        return None


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the landmarks and connections on the image."""
    annotated_image = np.copy(rgb_image)
    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
    return annotated_image


def draw_info_on_image(image, fps, posture_class, show_fps):
    """Draws FPS and posture status text on the image."""
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

    status_text = "No Person Detected"
    color = (255, 255, 0) # Yellow for no detection

    if posture_class:
        status_text = f"Posture: {posture_class}"
        color = (0, 255, 0) if posture_class == "Good Posture" else (0, 0, 255)
    
    cv2.putText(image, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def process_frame(frame, landmarker, knn_model, scaler, args):
    """Processes a single frame for pose detection and classification."""
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp = int(time.time() * 1000)
    detection_result = landmarker.detect_for_video(mp_image, timestamp)

    posture_class = None
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        try:
            pose_row = [coord for i in USED_LANDMARK_INDICES for coord in [landmarks[i].x, landmarks[i].y, landmarks[i].z]]
            live_features = np.array(pose_row).reshape(1, -1)
            live_features_scaled = scaler.transform(live_features)
            prediction = knn_model.predict(live_features_scaled)
            posture_class = "Good Posture" if prediction[0] == 1 else "Bad Posture"
        except IndexError:
            posture_class = "Landmarks Not Visible"
    
    # Draw landmarks if requested
    annotated_image = frame
    if args.display_pose_landmarkers:
        rgb_frame_for_drawing = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image_rgb = draw_landmarks_on_image(rgb_frame_for_drawing, detection_result)
        annotated_image = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

    return annotated_image, posture_class


def run_live_feed(cap, landmarker, knn_model, scaler, args):
    """Initializes and runs the main video capture and processing loop."""
    prev_time = time.time()
    fps_smooth = 0
    alpha = 0.9

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        annotated_image, posture_class = process_frame(frame, landmarker, knn_model, scaler, args)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        fps_smooth = alpha * fps_smooth + (1 - alpha) * fps

        # Draw informational text
        draw_info_on_image(annotated_image, fps_smooth, posture_class, args.display_fps)
        
        cv2.imshow('Posture Buddy', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


def main():
    """Main orchestrator function."""
    args = setup_arg_parser()
    
    print("Training posture classification model...")
    knn_model, scaler = train_knn_model(GOOD_POSTURE_PATHS, BAD_POSTURE_PATHS, USED_LANDMARK_INDICES)
    if knn_model is None:
        return

    print("Setting up MediaPipe Pose Landmarker...")
    landmarker_options = setup_landmarker(MODEL_PATH)
    if landmarker_options is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Use a 'with' statement for resource management
    with PoseLandmarker.create_from_options(landmarker_options) as landmarker:
        print("Pose landmarker created. Starting camera feed...")
        print("Press 'q' to quit.")
        run_live_feed(cap, landmarker, knn_model, scaler, args)


if __name__ == "__main__":
    main()

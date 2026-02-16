import os
import time

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

HAND_MODEL_PATH = "./models/hand_landmarker.task"
POSE_MODEL_PATH = "./models/pose_landmarker_lite.task"


def draw_landmarks_on_frame(image, detection_results):
    pose_landmarks_list = detection_results["pose"].pose_landmarks
    hand_landmarks_list = detection_results["hand"].hand_landmarks
    annotated_image = image

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = drawing_styles.get_default_hand_connections_style()

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )
    for hand_landmarks in hand_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_style,
            connection_drawing_spec=hand_connection_style,
        )
    return annotated_image

def mediapipe_detection(image, timestamp_start, pose_model, hand_model):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image.flags.writeable = False
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    results = {}
    timestamp = int((time.perf_counter_ns() - timestamp_start) * 10^6)
    results["pose"] = pose_model.detect_for_video(image, timestamp)
    results["hand"] = hand_model.detect_for_video(image, timestamp)
    
#    image.flags.writeable = True
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def main():
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_options=HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        timestamp_start = time.perf_counter_ns()
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            results = mediapipe_detection(
                                                 frame,
                                                 timestamp_start=timestamp_start,
                                                 pose_model=pose_landmarker,
                                                 hand_model=hand_landmarker,
                                             )

            frame = draw_landmarks_on_frame(frame, results)
            
            frame = cv2.flip(frame, 1)
            cv2.imshow("OpenCV Feed", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

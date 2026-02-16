from json.encoder import ESCAPE
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_MODEL_PATH = "./models/hand_landmarker.task"
POSE_MODEL_PATH = "./models/pose_landmarker_lite.task"

def main():
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )
    
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode = VisionRunningMode.VIDEO
    )

    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow("OpenCV Feed", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

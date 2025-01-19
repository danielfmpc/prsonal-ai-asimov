import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2

file_name = 'IMG_2149.mov'
model_path = 'pose_landmarker_heavy.task'


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


cap = cv2.VideoCapture(file_name)

options = python.vision.PoseLandmarkerOptions(
    base_Options=python.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=python.vision.RunningMode.VIDEO,
)

with python.vision.PoseLandmarker.create_from_options(options) as landmarker:
    calc_ts = [0.0]

    while (cap.isOpened()):
        ret, frame = cap.read()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if ret:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # calc_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            calc_ts.append(int(calc_ts[-1] + 1000 / fps))

            dect_result = landmarker.detect_for_video(
                mp_image, calc_ts[-1], calc_ts
            )
            frame = draw_landmarks_on_image(frame, dect_result)
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()

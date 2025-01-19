import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import queue
import threading
import math




class PersonalAI:
	def __init__(self, file_name, model_path):
		self.file_name = file_name
		self.model_path = model_path
		self.image_q = queue.Queue()

		self.options = python.vision.PoseLandmarkerOptions(
			base_options=python.BaseOptions(model_asset_path=model_path),
			running_mode=python.vision.RunningMode.VIDEO
		)


	def find_angle(self, frame, landmarks, p1:int, p2:int, p3:int):
		land = landmarks.pose_landmarks[0]
		h, w, _ = frame.shape

		x1, y1 = (land[p1].x, land[p1].y)
		x2, y2 = (land[p2].x, land[p2].y)
		x3, y3 = (land[p3].x, land[p3].y)

		angle = math.degrees(
			math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
		)

		position = (int(x2 * w + 10), int(y2 * h + 10))
		frame = cv2.putText(frame, str(int(angle)), position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
		
		return frame, angle


	def draw_landmarks_on_image(self, rgb_image, detection_result):
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
				solutions.drawing_styles.get_default_pose_landmarks_style()
			)
		return annotated_image

	def process_video(self, display):
		with python.vision.PoseLandmarker.create_from_options(self.options) as landmarker:
			cap = cv2.VideoCapture(self.file_name)
			self.fps = cap.get(cv2.CAP_PROP_FPS)
			calc_ts = [0.0]			

			if (not cap.isOpened()): 
				print("Error opening video stream or file")

			while (cap.isOpened()):
				ret, frame = cap.read()
				if ret:
					mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
					# calc_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
					calc_ts.append(int(calc_ts[-1] + 1000 / self.fps))

					dect_result = landmarker.detect_for_video(mp_image, calc_ts[-1])
					annotated_image = self.draw_landmarks_on_image(frame, dect_result)
					annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
					
					if display:	
						cv2.imshow('frame', annotated_image)
						if cv2.waitKey(25) & 0xFF == ord('q'): break
					
					self.image_q.put((annotated_image, dect_result, calc_ts[-1] / 1000))
				else:
					break
		
		self.image_q.put((1, 1, "done"))
		cap.release()
		cv2.destroyAllWindows()

	def run(self, display=True):
		t1 = threading.Thread(target=self.process_video, args=(display, ))
		t1.start()

if __name__ == '__main__':
	file_name = 'IMG_2149.mov'
	model_path = 'pose_landmarker_full.task'
	personal_ai = PersonalAI(file_name, model_path)
	personal_ai.process_video(display=True)
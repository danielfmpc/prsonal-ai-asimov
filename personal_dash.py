import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from personal_ai import *

st.set_page_config(
	layout="wide",
)

file_name = 'IMG_2149.mov'
model_path = 'pose_landmarker_full.task'


personal_ai = PersonalAI(file_name, model_path)
personal_ai.run(display=False)

placeholder = st.empty()
st.sidebar.title("AI Personal Trainer")
display_charts = st.sidebar.checkbox('Display charts', value=True)
reset = st.sidebar.button("Reset")

count = 0
status = "relaxed"

while True:
  frame, landmarks, ts = personal_ai.image_q.get()
  if ts == "done": break
  # st.image(frame, use_column_width=True)

  if len(landmarks.pose_landmarks) > 0:
    frame, elbow_angle = personal_ai.find_angle(frame, landmarks, 12, 14, 16)
    frame, hip_angle = personal_ai.find_angle(frame, landmarks, 11, 23, 25)

            
    # Pushup Logic
    if elbow_angle > 150 and hip_angle > 170:
      status = "ready"
      dir = "down"
    if elbow_angle < 50 and hip_angle < -190:
      status = "relaxed"
      dir = "up"

    if status == "ready":
      if dir == "down" and elbow_angle < 60:
        dir = "up"
        count += 0.5
      if dir == "up" and elbow_angle > 100:
        dir = "down"
        count += 0.5
    with placeholder.container():
      col1, col2 = st.columns([0.4, 0.6])

      status_stylle = f":green[{status}]" if status == "ready" else f":red[{status}]"
      col2.markdown(f"### **Status:** {status_stylle}")
      col2.markdown(f"### **Pushup Count:** {int(count)}")
      col2.divider()
      col1.image(frame)

import streamlit as st
import cv2
import time

from camera import CameraStream
from pose_detector import PoseDetector

def main():
    st.set_page_config(page_title="Posture Perfect", layout="wide")

    st.title("Posture Perfect")
    st.write("Live camera feed and FPS display.")

    st.sidebar.title("Camera Settings")

    placeholder = st.empty()

    # Initialize camera
    try:
        cam = CameraStream(0)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Initialize pose detector
    pose_detector = PoseDetector(
        model_complexity=1,  # 0=fast, 1=balanced, 2=accurate
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Main loop
    try:
        while True:
            frame = cam.read_frame()
            
            # Detect pose landmarks
            annotated_frame, landmarks, results = pose_detector.detect(frame)
            # TODO: Add logic to check if the pose is correct 

            # Get specific landmarks
            # left_shoulder = pose_detector.get_landmark_by_name(landmarks, 'LEFT_SHOULDER')
            # right_shoulder = pose_detector.get_landmark_by_name(landmarks, 'RIGHT_SHOULDER')

            # Display pose detection status
            if landmarks:
                status_text = f"Pose detected: {len(landmarks)} landmarks"
                cv2.putText(annotated_frame, status_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                status_text = "No pose detected"
                cv2.putText(annotated_frame, status_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.01)

    except Exception as e:
        st.error(str(e))

    finally:
        cam.release()
        pose_detector.release()

if __name__ == "__main__":
    main()
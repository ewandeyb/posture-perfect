import streamlit as st
import cv2
import time

from camera import CameraStream

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

    # Main loop
    try:
        while True:
            frame = cam.read_frame()
            #TODO: Add IVP techniques

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.01)

    except Exception as e:
        st.error(str(e))

    finally:
        cam.release()

if __name__ == "__main__":
    main()
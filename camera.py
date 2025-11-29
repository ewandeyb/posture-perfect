import cv2
import time

class CameraStream:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

        # FPS vars
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame.")

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # FPS calculation
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        # Put FPS text
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
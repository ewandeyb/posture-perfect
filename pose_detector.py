import cv2
import mediapipe as mp

class PoseDetector:
    """
    MediaPipe pose detection wrapper for posture analysis.
    """
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 enable_segmentation=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            static_image_mode: If True, treat input images as static. If False, use video mode.
            model_complexity: 0, 1, or 2. Higher = more accurate but slower.
            enable_segmentation: Enable body segmentation.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame):
        """
        Detect pose landmarks in a frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            tuple: (annotated_frame, landmarks, results)
                - annotated_frame: Frame with pose drawn
                - landmarks: List of landmark coordinates (x, y, z, visibility)
                - results: MediaPipe pose results object
        """
        # Convert BGR to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        # Draw pose landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Extract landmark coordinates
        landmarks = []
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x * w,  # Convert normalized to pixel coordinates
                    'y': landmark.y * h,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        
        return annotated_frame, landmarks, results
    
    def get_landmark_by_name(self, landmarks, landmark_name):
        """
        Get a specific landmark by name.
        
        Args:
            landmarks: List of landmark dictionaries
            landmark_name: Name from mp.solutions.pose.PoseLandmark
            
        Returns:
            dict: Landmark coordinates or None if not found
        """
        if not landmarks:
            return None
        
        landmark_index = self.mp_pose.PoseLandmark[landmark_name].value
        if landmark_index < len(landmarks):
            return landmarks[landmark_index]
        return None
    
    def release(self):
        """Release resources."""
        if self.pose:
            self.pose.close()
"""
MediaPipe pose detection wrapper for rehabilitation exercises.
Integrates with the modular exercise system.
"""
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import numpy as np


class PoseDetector:
    """
    MediaPipe pose detection wrapper for posture analysis.
    Provides methods to extract landmarks in formats expected by exercises.
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
            min_detection_confidence: Minimum confidence for detection (0.0-1.0).
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0).
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
        
        # Cache for landmark name to index mapping
        self._landmark_indices = {
            member.name: member.value 
            for member in self.mp_pose.PoseLandmark
        }       
    
    def process(self, frame_rgb):
        """
        Process a frame and detect pose landmarks.
        
        Args:
            frame_rgb: RGB image frame (NOT BGR)
            
        Returns:
            MediaPipe results object with pose_landmarks
            
        Example:
            results = detector.process(frame_rgb)
            if results.pose_landmarks:
                # Landmarks detected
        """
        return self.pose.process(frame_rgb)
    
    def detect(self, frame):
        """
        Detect pose landmarks in a BGR frame (legacy method).
        
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
    
    def extract_landmarks(self, 
                         results, 
                         required_landmarks: List[str],
                         frame_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Tuple[float, float, float]]:
        """
        Extract specific landmarks required by an exercise.
        
        This is the KEY METHOD exercises use to get their landmark data.
        
        Args:
            results: MediaPipe results object from process()
            required_landmarks: List of landmark names (e.g., ['LEFT_SHOULDER', 'LEFT_HIP'])
            frame_shape: Optional (height, width) to convert to pixel coords. 
                        If None, returns normalized coords (0-1)
            
        Returns:
            Dict mapping landmark names to (x, y, z) tuples
            Returns empty dict if pose not detected
            
        Example:
            results = detector.process(frame_rgb)
            landmarks = detector.extract_landmarks(
                results, 
                ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'],
                frame_shape=(720, 1280)
            )
            # landmarks = {
            #     'LEFT_SHOULDER': (640.5, 200.3, -0.5),
            #     'LEFT_HIP': (630.2, 400.1, -0.3),
            #     'LEFT_KNEE': (625.8, 600.9, -0.2)
            # }
        """
        if not results.pose_landmarks:
            return {}
        
        landmarks_dict = {}
        
        for landmark_name in required_landmarks:
            if landmark_name not in self._landmark_indices:
                print(f"Warning: Unknown landmark '{landmark_name}', skipping")
                continue
            
            idx = self._landmark_indices[landmark_name]
            landmark = results.pose_landmarks.landmark[idx]
            
            # Convert to pixel coordinates if frame shape provided
            if frame_shape:
                h, w = frame_shape
                x = landmark.x * w
                y = landmark.y * h
            else:
                x = landmark.x
                y = landmark.y
            
            landmarks_dict[landmark_name] = (x, y, landmark.z)
        
        return landmarks_dict
    
    def check_visibility(self, 
                        results, 
                        required_landmarks: List[str],
                        min_visibility: float = 0.5) -> Tuple[bool, List[str]]:
        """
        Check if required landmarks are visible enough.
        
        Useful for giving feedback like "Move into frame" or "Turn to face camera".
        
        Args:
            results: MediaPipe results object
            required_landmarks: List of landmark names to check
            min_visibility: Minimum visibility threshold (0.0-1.0)
            
        Returns:
            Tuple of (all_visible: bool, missing_landmarks: List[str])
            
        Example:
            all_visible, missing = detector.check_visibility(
                results, 
                ['LEFT_SHOULDER', 'LEFT_HIP'],
                min_visibility=0.7
            )
            if not all_visible:
                print(f"Can't see: {missing}")
        """
        if not results.pose_landmarks:
            return False, required_landmarks
        
        missing = []
        
        for landmark_name in required_landmarks:
            if landmark_name not in self._landmark_indices:
                missing.append(landmark_name)
                continue
            
            idx = self._landmark_indices[landmark_name]
            landmark = results.pose_landmarks.landmark[idx]
            
            if landmark.visibility < min_visibility:
                missing.append(landmark_name)
        
        all_visible = len(missing) == 0
        return all_visible, missing
    
    def draw_landmarks(self, 
                      frame: np.ndarray,
                      results,
                      connections: Optional[List[Tuple[str, str]]] = None) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Can draw custom connections specified by exercise or default full skeleton.
        
        Args:
            frame: Image frame (BGR or RGB)
            results: MediaPipe results object
            connections: Optional list of (landmark1, landmark2) tuples for custom drawing.
                        If None, draws all default MediaPipe connections.
            
        Returns:
            Frame with landmarks drawn
            
        Example:
            # Draw full skeleton
            frame = detector.draw_landmarks(frame, results)
            
            # Draw only specific lines (from exercise)
            frame = detector.draw_landmarks(frame, results, [
                ('LEFT_SHOULDER', 'LEFT_HIP'),
                ('LEFT_HIP', 'LEFT_KNEE')
            ])
        """
        if not results.pose_landmarks:
            return frame
        
        frame_copy = frame.copy()
        
        if connections is None:
            # Draw all default connections
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            # Draw custom connections specified by exercise
            h, w = frame.shape[:2]
            
            # First draw all landmarks as points
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            
            # Then draw custom connections
            for landmark1_name, landmark2_name in connections:
                if landmark1_name not in self._landmark_indices or \
                   landmark2_name not in self._landmark_indices:
                    continue
                
                idx1 = self._landmark_indices[landmark1_name]
                idx2 = self._landmark_indices[landmark2_name]
                
                lm1 = results.pose_landmarks.landmark[idx1]
                lm2 = results.pose_landmarks.landmark[idx2]
                
                # Convert to pixel coordinates
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                
                # Draw line
                cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame_copy
    
    def draw_angle_annotation(self,
                             frame: np.ndarray,
                             point: Tuple[float, float],
                             angle: float,
                             label: str = "",
                             color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw angle value annotation on frame.
        
        Useful for debugging or showing angles to users.
        
        Args:
            frame: Image frame
            point: (x, y) position to draw annotation
            angle: Angle value in degrees
            label: Optional label (e.g., "Knee Angle")
            color: BGR color tuple
            
        Returns:
            Frame with annotation
        """
        frame_copy = frame.copy()
        
        x, y = int(point[0]), int(point[1])
        text = f"{label}: {angle:.1f}°" if label else f"{angle:.1f}°"
        
        # Draw background rectangle
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_copy, (x - 5, y - text_h - 10), 
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame_copy
    
    def get_landmark_by_name(self, landmarks, landmark_name):
        """
        Get a specific landmark by name (legacy method).
        
        Args:
            landmarks: List of landmark dictionaries
            landmark_name: Name from mp.solutions.pose.PoseLandmark
            
        Returns:
            dict: Landmark coordinates or None if not found
        """
        if not landmarks:
            return None
        
        if landmark_name not in self._landmark_indices:
            return None
        
        landmark_index = self._landmark_indices[landmark_name]
        if landmark_index < len(landmarks):
            return landmarks[landmark_index]
        return None
    
    def get_all_landmark_names(self) -> List[str]:
        """
        Get list of all available landmark names.
        
        Useful for debugging or displaying available landmarks.
        
        Returns:
            List of landmark name strings
        """
        return sorted(self._landmark_indices.keys())
    
    def is_pose_detected(self, results) -> bool:
        """
        Check if any pose was detected in results.
        
        Args:
            results: MediaPipe results object
            
        Returns:
            True if pose detected, False otherwise
        """
        return results.pose_landmarks is not None
    
    def release(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.release()

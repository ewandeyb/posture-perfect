"""
Reusable geometric calculations for pose estimation.
Provides common angle and distance calculations used across exercises.

All angles are returned in degrees for better readability.
All coordinates are assumed to be normalized (0-1 range) or pixel coordinates.
"""
import math
from typing import Tuple, List, Optional
import numpy as np


class AngleCalculator:
    """
    Static methods for calculating angles and distances from body landmarks.
    Used by exercise classes to compute metrics.
    """
    
    # ==================== BASIC ANGLE CALCULATIONS ====================
    
    @staticmethod
    def calculate_angle(p1: Tuple[float, float, float],
                       p2: Tuple[float, float, float],
                       p3: Tuple[float, float, float]) -> float:
        """
        Calculate angle formed by three points (p2 is the vertex).
        
        Uses the law of cosines to find the angle at p2.
        
        Args:
            p1: First point (x, y, z)
            p2: Vertex point (x, y, z) - the angle is measured here
            p3: Third point (x, y, z)
            
        Returns:
            Angle in degrees (0-180)
            
        Example:
            # Calculate elbow angle
            angle = AngleCalculator.calculate_angle(
                shoulder, elbow, wrist  # elbow is the vertex
            )
        """
        # Extract 2D coordinates (ignore z for simplicity)
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        # Vectors from vertex
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        
        # Clamp to valid range to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_angle_3d(p1: Tuple[float, float, float],
                          p2: Tuple[float, float, float],
                          p3: Tuple[float, float, float]) -> float:
        """
        Calculate 3D angle formed by three points.
        
        Similar to calculate_angle but uses all three dimensions.
        Useful when depth information is important.
        
        Args:
            p1, p2, p3: 3D points (x, y, z)
            
        Returns:
            Angle in degrees (0-180)
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def angle_from_vertical(p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> float:
        """
        Calculate angle from vertical (gravity direction).
        
        0° = perfectly vertical (p2 directly below p1)
        90° = horizontal
        
        Args:
            p1: Upper point (x, y, z)
            p2: Lower point (x, y, z)
            
        Returns:
            Angle from vertical in degrees (0-180)
            
        Example:
            # Check if torso is upright
            angle = AngleCalculator.angle_from_vertical(shoulder, hip)
            if angle > 15:
                print("Lean detected")
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # atan2 gives angle from horizontal, convert to vertical
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        
        return angle
    
    @staticmethod
    def angle_from_horizontal(p1: Tuple[float, float, float],
                             p2: Tuple[float, float, float]) -> float:
        """
        Calculate angle from horizontal.
        
        0° = perfectly horizontal
        90° = vertical
        
        Args:
            p1: Left/start point (x, y, z)
            p2: Right/end point (x, y, z)
            
        Returns:
            Angle from horizontal in degrees (-90 to 90)
            
        Example:
            # Check shoulder level
            angle = AngleCalculator.angle_from_horizontal(left_shoulder, right_shoulder)
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        
        return angle
    
    # ==================== DISTANCE CALCULATIONS ====================
    
    @staticmethod
    def euclidean_distance_2d(p1: Tuple[float, float, float],
                             p2: Tuple[float, float, float]) -> float:
        """
        Calculate 2D Euclidean distance between two points.
        
        Args:
            p1, p2: Points (x, y, z) - z is ignored
            
        Returns:
            Distance in the same units as input coordinates
        """
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def euclidean_distance_3d(p1: Tuple[float, float, float],
                             p2: Tuple[float, float, float]) -> float:
        """
        Calculate 3D Euclidean distance between two points.
        
        Args:
            p1, p2: 3D points (x, y, z)
            
        Returns:
            Distance in the same units as input coordinates
        """
        return math.sqrt(
            (p2[0] - p1[0])**2 + 
            (p2[1] - p1[1])**2 + 
            (p2[2] - p1[2])**2
        )
    
    @staticmethod
    def vertical_distance(p1: Tuple[float, float, float],
                         p2: Tuple[float, float, float]) -> float:
        """
        Calculate vertical (y-axis) distance between two points.
        
        Positive = p2 is below p1
        Negative = p2 is above p1
        
        Args:
            p1, p2: Points (x, y, z)
            
        Returns:
            Vertical distance (signed)
        """
        return p2[1] - p1[1]
    
    @staticmethod
    def horizontal_distance(p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> float:
        """
        Calculate horizontal (x-axis) distance between two points.
        
        Args:
            p1, p2: Points (x, y, z)
            
        Returns:
            Horizontal distance (absolute)
        """
        return abs(p2[0] - p1[0])
    
    # ==================== BODY-SPECIFIC CALCULATIONS ====================
    
    @staticmethod
    def calculate_spine_angle(shoulder_mid: Tuple[float, float, float],
                             hip_mid: Tuple[float, float, float],
                             reference_point: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate spine curvature angle.
        
        If reference_point provided, calculates 3-point angle (e.g., shoulder-hip-knee).
        Otherwise, calculates angle from vertical.
        
        Args:
            shoulder_mid: Midpoint between shoulders
            hip_mid: Midpoint between hips
            reference_point: Optional third point (e.g., knee midpoint)
            
        Returns:
            Spine angle in degrees
            
        Example:
            # Simple vertical angle
            angle = AngleCalculator.calculate_spine_angle(shoulder_mid, hip_mid)
            
            # 3-point curvature
            angle = AngleCalculator.calculate_spine_angle(
                shoulder_mid, hip_mid, knee_mid
            )
        """
        if reference_point:
            return AngleCalculator.calculate_angle(
                shoulder_mid, hip_mid, reference_point
            )
        else:
            return AngleCalculator.angle_from_vertical(shoulder_mid, hip_mid)
    
    @staticmethod
    def calculate_knee_angle(hip: Tuple[float, float, float],
                            knee: Tuple[float, float, float],
                            ankle: Tuple[float, float, float]) -> float:
        """
        Calculate knee flexion angle.
        
        180° = fully extended
        90° = right angle bend
        0° = fully flexed
        
        Args:
            hip, knee, ankle: Joint positions
            
        Returns:
            Knee angle in degrees (0-180)
        """
        return AngleCalculator.calculate_angle(hip, knee, ankle)
    
    @staticmethod
    def calculate_elbow_angle(shoulder: Tuple[float, float, float],
                             elbow: Tuple[float, float, float],
                             wrist: Tuple[float, float, float]) -> float:
        """
        Calculate elbow flexion angle.
        
        180° = fully extended
        90° = right angle bend
        0° = fully flexed
        
        Args:
            shoulder, elbow, wrist: Joint positions
            
        Returns:
            Elbow angle in degrees (0-180)
        """
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_hip_angle(shoulder: Tuple[float, float, float],
                           hip: Tuple[float, float, float],
                           knee: Tuple[float, float, float]) -> float:
        """
        Calculate hip flexion angle.
        
        Args:
            shoulder, hip, knee: Joint positions
            
        Returns:
            Hip angle in degrees (0-180)
        """
        return AngleCalculator.calculate_angle(shoulder, hip, knee)
    
    @staticmethod
    def calculate_shoulder_abduction(shoulder: Tuple[float, float, float],
                                    elbow: Tuple[float, float, float],
                                    hip: Tuple[float, float, float]) -> float:
        """
        Calculate shoulder abduction angle (arm away from body).
        
        0° = arm at side
        90° = arm horizontal
        180° = arm overhead
        
        Args:
            shoulder, elbow, hip: Joint positions
            
        Returns:
            Abduction angle in degrees
        """
        return AngleCalculator.calculate_angle(hip, shoulder, elbow)
    
    # ==================== HELPER FUNCTIONS ====================
    
    @staticmethod
    def midpoint_2d(p1: Tuple[float, float, float],
                    p2: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Calculate 2D midpoint between two points.
        
        Args:
            p1, p2: Points (x, y, z)
            
        Returns:
            Midpoint (x, y)
        """
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    @staticmethod
    def midpoint_3d(p1: Tuple[float, float, float],
                    p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Calculate 3D midpoint between two points.
        
        Args:
            p1, p2: 3D points
            
        Returns:
            Midpoint (x, y, z)
        """
        return (
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2,
            (p1[2] + p2[2]) / 2
        )
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to 0-360 range.
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Normalized angle (0-360)
        """
        return angle % 360
    
    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """
        Calculate smallest difference between two angles.
        
        Handles wraparound (e.g., difference between 350° and 10° is 20°, not 340°)
        
        Args:
            angle1, angle2: Angles in degrees
            
        Returns:
            Absolute angle difference (0-180)
        """
        diff = abs(angle1 - angle2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
    
    # ==================== BODY SYMMETRY ====================
    
    @staticmethod
    def calculate_symmetry_score(left_angle: float, right_angle: float,
                                tolerance: float = 5.0) -> Tuple[bool, float]:
        """
        Calculate symmetry between left and right body sides.
        
        Critical for scoliosis and asymmetry detection.
        
        Args:
            left_angle: Angle measurement from left side
            right_angle: Angle measurement from right side
            tolerance: Acceptable difference in degrees
            
        Returns:
            Tuple of (is_symmetric: bool, difference: float)
            
        Example:
            is_sym, diff = AngleCalculator.calculate_symmetry_score(
                left_shoulder_angle, right_shoulder_angle, tolerance=10
            )
            if not is_sym:
                print(f"Asymmetry detected: {diff:.1f}° difference")
        """
        difference = abs(left_angle - right_angle)
        is_symmetric = difference <= tolerance
        
        return is_symmetric, difference
    
    @staticmethod
    def calculate_shoulder_level(left_shoulder: Tuple[float, float, float],
                                right_shoulder: Tuple[float, float, float],
                                tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Check if shoulders are level (important for posture).
        
        Args:
            left_shoulder, right_shoulder: Shoulder positions
            tolerance: Acceptable vertical difference (normalized coords)
            
        Returns:
            Tuple of (is_level: bool, tilt_angle: float)
        """
        vertical_diff = abs(left_shoulder[1] - right_shoulder[1])
        tilt_angle = AngleCalculator.angle_from_horizontal(left_shoulder, right_shoulder)
        
        is_level = vertical_diff <= tolerance
        
        return is_level, tilt_angle
    
    @staticmethod
    def calculate_hip_level(left_hip: Tuple[float, float, float],
                           right_hip: Tuple[float, float, float],
                           tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Check if hips are level.
        
        Args:
            left_hip, right_hip: Hip positions
            tolerance: Acceptable vertical difference (normalized coords)
            
        Returns:
            Tuple of (is_level: bool, tilt_angle: float)
        """
        vertical_diff = abs(left_hip[1] - right_hip[1])
        tilt_angle = AngleCalculator.angle_from_horizontal(left_hip, right_hip)
        
        is_level = vertical_diff <= tolerance
        
        return is_level, tilt_angle
    
    # ==================== ADVANCED CALCULATIONS ====================
    
    @staticmethod
    def calculate_body_ratio(segment1_length: float, segment2_length: float) -> float:
        """
        Calculate ratio between body segments.
        
        Useful for checking proportional movement.
        
        Args:
            segment1_length: Length of first segment
            segment2_length: Length of second segment
            
        Returns:
            Ratio (segment1 / segment2)
        """
        if segment2_length == 0:
            return 0
        return segment1_length / segment2_length
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float],
                               line_p1: Tuple[float, float],
                               line_p2: Tuple[float, float]) -> float:
        """
        Calculate perpendicular distance from point to line.
        
        Useful for checking alignment (e.g., is knee aligned with hip-ankle line?).
        
        Args:
            point: Point to measure from (x, y)
            line_p1, line_p2: Two points defining the line (x, y)
            
        Returns:
            Perpendicular distance
        """
        # Line equation: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
        x0, y0 = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    @staticmethod
    def is_point_between(point: Tuple[float, float, float],
                        boundary1: Tuple[float, float, float],
                        boundary2: Tuple[float, float, float],
                        axis: str = 'x') -> bool:
        """
        Check if point is between two boundaries along specified axis.
        
        Args:
            point: Point to check
            boundary1, boundary2: Boundary points
            axis: 'x', 'y', or 'z'
            
        Returns:
            True if point is between boundaries
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_map[axis]
        
        min_val = min(boundary1[idx], boundary2[idx])
        max_val = max(boundary1[idx], boundary2[idx])
        
        return min_val <= point[idx] <= max_val

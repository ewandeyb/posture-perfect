"""
Abstract base class for all rehabilitation exercises.
Each exercise must implement its own validation logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of exercise validation"""
    is_correct: bool
    feedback_messages: List[str]
    angles: Dict[str, float]
    score: float  # 0-100
    details: Dict[str, any]  # Additional exercise-specific data


class BaseExercise(ABC):
    """
    Abstract base class for all exercises.
    Each exercise defines its own:
    - Required landmarks
    - Angle calculations
    - Validation thresholds
    - Feedback messages
    """
    
    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
        self.required_landmarks = self.get_required_landmarks()
        
    @abstractmethod
    def get_name(self) -> str:
        """Return the exercise name"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return exercise description for UI"""
        pass
    
    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """
        Return list of required MediaPipe landmarks.
        Example: ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE']
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate all relevant metrics (angles, distances) from landmarks.
        
        Args:
            landmarks: Dict mapping landmark names to (x, y, z) coordinates
            
        Returns:
            Dict of metric names to values (e.g., {'shoulder_angle': 45.0})
        """
        pass
    
    @abstractmethod
    def validate_form(self, metrics: Dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate exercise form based on calculated metrics.
        
        Args:
            metrics: Output from calculate_metrics()
            frame_count: Current frame number (for temporal validation)
            
        Returns:
            ValidationResult with feedback and correctness
        """
        pass
    
    def get_visualization_points(self) -> List[Tuple[str, str]]:
        """
        Return pairs of landmarks to draw as lines for visualization.
        Override if custom visualization needed.
        
        Returns:
            List of (landmark1, landmark2) tuples
        """
        return []
    
    def reset(self):
        """
        Reset exercise state (e.g., rep counters, temporal buffers).
        Override if exercise maintains state.
        """
        pass
    
    def get_instructions(self) -> List[str]:
        """
        Return step-by-step instructions for the exercise.
        Override to provide custom instructions.
        """
        return [
            "Position yourself in frame",
            "Follow the on-screen guidance",
            "Maintain proper form"
        ]
    
    def get_common_mistakes(self) -> List[str]:
        """
        Return common mistakes for this exercise.
        Override to provide exercise-specific warnings.
        """
        return []
"""
Scoliosis Cat-Cow Exercise (Marjaryasana-Bitilasana)
Therapeutic spine flexion and extension for scoliosis management.
"""
import math
from typing import Dict, List, Tuple
from collections import deque
from exercises.base_exercise import BaseExercise, ValidationResult
from exercises.exercise_factory import ExerciseFactory


@ExerciseFactory.register('scoliosis_cat_cow')
class CatCowPose(BaseExercise):
    """
    Cat-Cow pose for scoliosis rehabilitation.
    
    Validates:
    - Spinal flexion/extension range
    - Shoulder-hip alignment
    - Movement tempo and control
    - Proper hand/knee positioning
    
    Benefits for Scoliosis:
    - Improves spinal flexibility
    - Strengthens core muscles
    - Increases body awareness
    - Promotes symmetrical movement patterns
    """
    
    MIN_SPINE_FLEX = 10      # Minimum flexion angle (Cat pose)
    MAX_SPINE_FLEX = 35      # Maximum safe flexion
    MIN_SPINE_EXT = 10       # Minimum extension angle (Cow pose)
    MAX_SPINE_EXT = 30       # Maximum safe extension
    
    TEMPO_MIN = 2.0          # Minimum seconds per transition
    TEMPO_MAX = 5.0          # Maximum seconds per transition
    IDEAL_TEMPO = 3.0        # Ideal tempo
    
    SHOULDER_WRIST_TOLERANCE = 15  # Degrees from vertical
    HIP_KNEE_TOLERANCE = 15        # Degrees from vertical
    
    # Position states
    STATE_NEUTRAL = "neutral"
    STATE_CAT = "cat"
    STATE_COW = "cow"
    
    def __init__(self):
        super().__init__()
        
        # State tracking
        self.current_state = self.STATE_NEUTRAL
        self.last_state_change = 0
        self.rep_count = 0
        
        # Temporal tracking
        self.spine_angle_history = deque(maxlen=150)  # 5 seconds at 30fps
        self.transition_times = []
        
        # Movement quality tracking
        self.symmetry_violations = 0
        self.tempo_violations = 0
        
    def get_name(self) -> str:
        return "Cat-Cow Pose (Scoliosis)"
    
    def get_description(self) -> str:
        return (
            "Spinal flexion-extension exercise performed on hands and knees. "
            "Alternates between arching (Cow) and rounding (Cat) the spine "
            "to improve flexibility and core strength."
        )
    
    def get_required_landmarks(self) -> List[str]:
        return [
            # Upper body
            'NOSE',
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_ELBOW',
            'RIGHT_ELBOW',
            'LEFT_WRIST',
            'RIGHT_WRIST',
            
            # Core
            'LEFT_HIP',
            'RIGHT_HIP',
            
            # Lower body
            'LEFT_KNEE',
            'RIGHT_KNEE',
            'LEFT_ANKLE',
            'RIGHT_ANKLE'
        ]
    
    def calculate_metrics(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate all biomechanical metrics for Cat-Cow pose.
        
        Metrics:
        - spine_curvature: Overall spine angle (+ = extension/cow, - = flexion/cat)
        - shoulder_alignment: Shoulder-wrist vertical alignment
        - hip_alignment: Hip-knee vertical alignment
        - left_right_symmetry: Difference between left and right side curves
        - head_position: Neck alignment with spine
        """
        
        # Calculate midpoints
        shoulder_mid = self._midpoint(
            landmarks['LEFT_SHOULDER'],
            landmarks['RIGHT_SHOULDER']
        )
        hip_mid = self._midpoint(
            landmarks['LEFT_HIP'],
            landmarks['RIGHT_HIP']
        )
        
        # Spine curvature (shoulder-hip-knee angle)
        # Positive = extension (Cow), Negative = flexion (Cat)
        knee_mid = self._midpoint(
            landmarks['LEFT_KNEE'],
            landmarks['RIGHT_KNEE']
        )
        
        spine_angle = self._calculate_angle(
            shoulder_mid,
            hip_mid,
            knee_mid
        )
        
        # Adjust to make extension positive, flexion negative
        spine_curvature = 180 - spine_angle
        if hip_mid[1] < shoulder_mid[1]:  # If hips higher than shoulders
            spine_curvature = -spine_curvature
        
        # Shoulder-wrist alignment (should be vertical)
        left_shoulder_alignment = self._angle_from_vertical(
            landmarks['LEFT_SHOULDER'],
            landmarks['LEFT_WRIST']
        )
        right_shoulder_alignment = self._angle_from_vertical(
            landmarks['RIGHT_SHOULDER'],
            landmarks['RIGHT_WRIST']
        )
        shoulder_alignment = (abs(left_shoulder_alignment) + 
                             abs(right_shoulder_alignment)) / 2
        
        # Hip-knee alignment (should be vertical)
        left_hip_alignment = self._angle_from_vertical(
            landmarks['LEFT_HIP'],
            landmarks['LEFT_KNEE']
        )
        right_hip_alignment = self._angle_from_vertical(
            landmarks['RIGHT_HIP'],
            landmarks['RIGHT_KNEE']
        )
        hip_alignment = (abs(left_hip_alignment) + 
                        abs(right_hip_alignment)) / 2
        
        # Left-right symmetry (important for scoliosis)
        left_spine_curve = self._calculate_angle(
            landmarks['LEFT_SHOULDER'],
            landmarks['LEFT_HIP'],
            landmarks['LEFT_KNEE']
        )
        right_spine_curve = self._calculate_angle(
            landmarks['RIGHT_SHOULDER'],
            landmarks['RIGHT_HIP'],
            landmarks['RIGHT_KNEE']
        )
        symmetry_diff = abs(left_spine_curve - right_spine_curve)
        
        # Head position (nose should follow spine)
        head_spine_alignment = abs(
            landmarks['NOSE'][0] - shoulder_mid[0]
        ) * 100  # Scale to reasonable range
        
        return {
            'spine_curvature': spine_curvature,
            'shoulder_alignment': shoulder_alignment,
            'hip_alignment': hip_alignment,
            'symmetry_diff': symmetry_diff,
            'head_alignment': head_spine_alignment
        }
    
    def validate_form(self, metrics: Dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate Cat-Cow form with state machine and temporal analysis.
        """
        
        feedback = []
        is_correct = True
        score = 100.0
        
        spine_curve = metrics['spine_curvature']
        
        # Add to history
        self.spine_angle_history.append(spine_curve)
        
        # ========== STATE DETECTION ==========
        previous_state = self.current_state
        
        if spine_curve < -self.MIN_SPINE_FLEX:
            self.current_state = self.STATE_CAT
        elif spine_curve > self.MIN_SPINE_EXT:
            self.current_state = self.STATE_COW
        else:
            self.current_state = self.STATE_NEUTRAL
        
        # Detect state transitions (completed reps)
        if previous_state != self.current_state:
            transition_time = (frame_count - self.last_state_change) / 30.0
            
            # Only count Cat<->Cow transitions as reps
            if (previous_state == self.STATE_CAT and self.current_state == self.STATE_COW) or \
               (previous_state == self.STATE_COW and self.current_state == self.STATE_CAT):
                
                self.rep_count += 0.5  # Each full cycle = 2 transitions
                
                # Check transition tempo
                if transition_time < self.TEMPO_MIN:
                    feedback.append("⚠️ Slow down - move with control")
                    self.tempo_violations += 1
                    score -= 5
                elif transition_time > self.TEMPO_MAX:
                    feedback.append("⚠️ Speed up slightly - maintain flow")
                    self.tempo_violations += 1
                    score -= 3
                else:
                    feedback.append(f"✓ Good tempo ({transition_time:.1f}s)")
                
                self.transition_times.append(transition_time)
            
            self.last_state_change = frame_count
        
        # ========== POSITION VALIDATION ==========
        
        # 1. Check spine curvature range
        if self.current_state == self.STATE_CAT:
            if spine_curve > -self.MIN_SPINE_FLEX:
                feedback.append("⚠️ CAT: Round spine more (arch upward)")
                score -= 15
                is_correct = False
            elif spine_curve < -self.MAX_SPINE_FLEX:
                feedback.append("⚠️ CAT: Don't over-flex spine")
                score -= 20
                is_correct = False
            else:
                feedback.append("✓ CAT: Good spinal flexion")
        
        elif self.current_state == self.STATE_COW:
            if spine_curve < self.MIN_SPINE_EXT:
                feedback.append("⚠️ COW: Extend spine more (drop belly)")
                score -= 15
                is_correct = False
            elif spine_curve > self.MAX_SPINE_EXT:
                feedback.append("⚠️ COW: Don't over-extend spine")
                score -= 20
                is_correct = False
            else:
                feedback.append("✓ COW: Good spinal extension")
        
        else:
            feedback.append("ℹ️ Return to starting position")
        
        # 2. Check shoulder-wrist alignment
        if metrics['shoulder_alignment'] > self.SHOULDER_WRIST_TOLERANCE:
            feedback.append("⚠️ Hands should be directly under shoulders")
            score -= 10
            is_correct = False
        
        # 3. Check hip-knee alignment
        if metrics['hip_alignment'] > self.HIP_KNEE_TOLERANCE:
            feedback.append("⚠️ Knees should be directly under hips")
            score -= 10
            is_correct = False
        
        # 4. Check symmetry (critical for scoliosis)
        if metrics['symmetry_diff'] > 15:
            feedback.append("⚠️ IMPORTANT: Keep movement symmetrical on both sides")
            self.symmetry_violations += 1
            score -= 15
            is_correct = False
        else:
            feedback.append("✓ Good left-right symmetry")
        
        # 5. Check head alignment
        if metrics['head_alignment'] > 20:
            feedback.append("⚠️ Keep head aligned with spine")
            score -= 5
        
        # ========== PROGRESS TRACKING ==========
        if self.rep_count > 0:
            feedback.append(f"📊 Reps completed: {int(self.rep_count)}")
        
        # Average tempo
        if len(self.transition_times) > 0:
            avg_tempo = sum(self.transition_times) / len(self.transition_times)
            feedback.append(f"⏱️ Average tempo: {avg_tempo:.1f}s")
        
        score = max(0, score)
        
        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                'spine_curvature': spine_curve,
                'shoulder_alignment': metrics['shoulder_alignment'],
                'hip_alignment': metrics['hip_alignment'],
                'symmetry': metrics['symmetry_diff']
            },
            score=score,
            details={
                'state': self.current_state,
                'reps': int(self.rep_count),
                'symmetry_violations': self.symmetry_violations,
                'tempo_violations': self.tempo_violations,
                'avg_tempo': sum(self.transition_times) / len(self.transition_times) 
                            if self.transition_times else 0
            }
        )
    
    def get_visualization_points(self) -> List[Tuple[str, str]]:
        """Define skeleton connections to draw"""
        return [
            # Spine line
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            
            # Arms
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # Legs
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ]
    
    def reset(self):
        """Reset all exercise state"""
        self.current_state = self.STATE_NEUTRAL
        self.last_state_change = 0
        self.rep_count = 0
        self.spine_angle_history.clear()
        self.transition_times.clear()
        self.symmetry_violations = 0
        self.tempo_violations = 0
    
    def get_instructions(self) -> List[str]:
        return [
            "Start on hands and knees (tabletop position)",
            "Place hands directly under shoulders, knees under hips",
            "COW POSE: Inhale, drop belly, lift chest and tailbone (3 seconds)",
            "CAT POSE: Exhale, round spine upward, tuck chin and tailbone (3 seconds)",
            "Move slowly and with control between positions",
            "Focus on symmetrical movement on both sides",
            "Perform 5-10 complete cycles"
        ]
    
    def get_common_mistakes(self) -> List[str]:
        return [
            "Moving too quickly - should be slow and controlled",
            "Only moving upper back instead of entire spine",
            "Holding breath - remember to breathe with movement",
            "Hands not aligned under shoulders (puts strain on wrists)",
            "Asymmetrical movement (one side bending more than other)",
            "Over-extending neck instead of keeping it neutral",
            "Locking elbows instead of keeping slight bend"
        ]
    
    # ========== HELPER METHODS ==========
    
    def _midpoint(self, p1: Tuple[float, float, float], 
                  p2: Tuple[float, float, float]) -> Tuple[float, float]:
        """Calculate 2D midpoint between two 3D points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _angle_from_vertical(self, p1: Tuple, p2: Tuple) -> float:
        """
        Calculate angle from vertical (0° = straight up).
        Returns positive for deviation from vertical.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        return angle
    
    def _calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """
        Calculate angle formed by three points (p2 is vertex).
        Uses law of cosines.
        """
        # Calculate distances
        a = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        b = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
        c = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        
        # Avoid division by zero
        if a * b == 0:
            return 0
        
        # Law of cosines
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        return math.degrees(math.acos(cos_angle))
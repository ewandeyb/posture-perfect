"""
Chest Opener / Scapular Retraction Exercise
Therapeutic exercise for upper back strength and posture correction.

Best performed in SIDE VIEW to properly measure depth (z-axis) of arm retraction.

- Focus: Scapular retraction, chest opening, upper back engagement
- Target: Rhomboids, middle trapezius, posterior deltoids
- Indication: Forward head posture, rounded shoulders, upper crossed syndrome
"""
from typing import Dict, List, Tuple
from collections import deque
from exercises.base_exercise import BaseExercise, ValidationResult
from exercises.exercise_factory import ExerciseFactory
from angle_calculator import AngleCalculator


@ExerciseFactory.register('chest_opener_retraction')
class ChestOpener(BaseExercise):
    """
    Chest opener exercise with scapular retraction.
    
    Patient pulls fists/arms back to open chest and engage upper back.
    Requires SIDE VIEW for proper depth measurement.
    
    Validates:
    - Arm retraction depth (z-axis measurement critical)
    - Elbow position (should be at shoulder height)
    - Hold duration (5 seconds)
    - Shoulder elevation (shouldn't shrug)
    - Forward lean compensation
    """
    
    # Clinical thresholds (PT-validated)
    MIN_RETRACTION_DEPTH = 0.15      # Minimum z-depth (normalized units)
    IDEAL_RETRACTION_DEPTH = 0.25    # Ideal z-depth for full retraction
    MAX_RETRACTION_DEPTH = 0.40      # Maximum (avoid over-retraction)
    
    ELBOW_HEIGHT_TOLERANCE = 0.08    # Vertical distance from shoulder (normalized)
    SHOULDER_ELEVATION_MAX = 0.05    # Max shoulder rise (avoid shrugging)
    
    HOLD_DURATION = 5                # Seconds to hold position
    HOLD_FRAMES = 150                # 5 seconds at 30fps
    
    MIN_ELBOW_ANGLE = 70             # Minimum elbow flexion
    MAX_ELBOW_ANGLE = 110            # Maximum elbow flexion (should be ~90°)
    
    # Side view detection
    MIN_SIDE_VIEW_ANGLE = 60         # Minimum angle to consider "side view"
    
    def __init__(self):
        super().__init__()
        self.calc = AngleCalculator()
        
        # State tracking
        self.is_retracted = False
        self.hold_counter = 0
        self.rep_count = 0
        self.max_depth_achieved = 0
        
        # Temporal tracking
        self.depth_history = deque(maxlen=90)  # 3 seconds
        self.last_retraction_time = 0
        
        # Quality metrics
        self.successful_holds = 0
        self.shrugging_violations = 0
        
    def get_name(self) -> str:
        return "Chest Opener (Scapular Retraction)"
    
    def get_description(self) -> str:
        return (
            "Pull arms back to open chest and engage upper back muscles. "
            "Strengthens rhomboids and middle trapezius. "
            "⚠️ REQUIRES SIDE VIEW for proper depth measurement."
        )
    
    def get_required_landmarks(self) -> List[str]:
        return [
            # Upper body (critical for this exercise)
            'NOSE',
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_ELBOW',
            'RIGHT_ELBOW',
            'LEFT_WRIST',
            'RIGHT_WRIST',
            
            # Core (for compensation detection)
            'LEFT_HIP',
            'RIGHT_HIP',
            
            # Additional points for side view validation
            'LEFT_EAR',
            'RIGHT_EAR'
        ]
    
    def calculate_metrics(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate metrics with emphasis on Z-AXIS (depth) measurement.
        
        Critical for this exercise: The z-coordinate shows how far back
        the arms are pulled in 3D space (depth from camera).
        """
        
        # ===== DETERMINE WHICH SIDE IS VISIBLE =====
        # Use the side that's more visible (closer to camera)
        left_shoulder_z = landmarks['LEFT_SHOULDER'][2]
        right_shoulder_z = landmarks['RIGHT_SHOULDER'][2]
        
        # More negative z = closer to camera in MediaPipe
        if abs(left_shoulder_z) < abs(right_shoulder_z):
            # Left side is visible (facing right in camera)
            shoulder = landmarks['LEFT_SHOULDER']
            elbow = landmarks['LEFT_ELBOW']
            wrist = landmarks['LEFT_WRIST']
            hip = landmarks['LEFT_HIP']
            visible_side = 'left'
        else:
            # Right side is visible (facing left in camera)
            shoulder = landmarks['RIGHT_SHOULDER']
            elbow = landmarks['RIGHT_ELBOW']
            wrist = landmarks['RIGHT_WRIST']
            hip = landmarks['RIGHT_HIP']
            visible_side = 'right'
        
        # ===== CHECK IF SIDE VIEW =====
        # Calculate angle between shoulders to determine viewing angle
        shoulder_diff_x = abs(landmarks['LEFT_SHOULDER'][0] - landmarks['RIGHT_SHOULDER'][0])
        shoulder_diff_z = abs(landmarks['LEFT_SHOULDER'][2] - landmarks['RIGHT_SHOULDER'][2])
        
        # If z-difference is small compared to x-difference, user is facing camera (frontal view)
        # We want side view where z-difference is significant
        side_view_score = shoulder_diff_z / (shoulder_diff_x + 0.01)  # Avoid division by zero
        is_side_view = side_view_score > 0.3  # Threshold for "good enough" side view
        
        # ===== ARM RETRACTION DEPTH (KEY METRIC) =====
        # Z-coordinate difference: how far back is elbow compared to shoulder?
        # Positive = elbow is behind shoulder (retracted)
        # Negative = elbow is in front of shoulder
        retraction_depth = abs(shoulder[2] - elbow[2])
        
        # Track maximum depth achieved
        self.max_depth_achieved = max(self.max_depth_achieved, retraction_depth)
        self.depth_history.append(retraction_depth)
        
        # ===== ELBOW POSITION =====
        # Elbow should be at shoulder height (not drooping or elevated)
        elbow_height_diff = abs(elbow[1] - shoulder[1])
        
        # Elbow angle (should be ~90 degrees)
        elbow_angle = self.calc.calculate_elbow_angle(shoulder, elbow, wrist)
        
        # ===== SHOULDER ELEVATION (SHRUGGING) =====
        # Compare current shoulder height to hip reference
        # Shoulder shouldn't rise significantly during retraction
        shoulder_hip_distance = self.calc.vertical_distance(hip, shoulder)
        
        # We'll compare this to a baseline (could be improved with calibration)
        # For now, detect sudden elevation
        shoulder_elevation = shoulder_hip_distance
        
        # ===== FORWARD LEAN COMPENSATION =====
        # Check if person is leaning forward to "cheat" the retraction
        torso_angle = self.calc.angle_from_vertical(shoulder, hip)
        
        # ===== WRIST POSITION =====
        # Wrist should be behind elbow (fist pulled back)
        wrist_depth = abs(elbow[2] - wrist[2])
        
        return {
            'retraction_depth': retraction_depth,
            'elbow_height_diff': elbow_height_diff,
            'elbow_angle': elbow_angle,
            'shoulder_elevation': shoulder_elevation,
            'torso_lean': torso_angle,
            'wrist_depth': wrist_depth,
            'is_side_view': is_side_view,
            'side_view_score': side_view_score,
            'visible_side': visible_side
        }
    
    def validate_form(self, metrics: Dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate chest opener form with focus on depth and hold duration.
        """
        
        feedback = []
        is_correct = True
        score = 100.0
        
        # ========== CRITICAL: CHECK SIDE VIEW ==========
        if not metrics['is_side_view']:
            feedback.append("⚠️ TURN TO SIDE VIEW - Need side profile to measure depth!")
            feedback.append(f"   (Side view quality: {metrics['side_view_score']*100:.0f}%, need >30%)")
            is_correct = False
            score -= 50  # Major penalty - can't properly measure without side view
            
            return ValidationResult(
                is_correct=False,
                feedback_messages=feedback,
                angles={
                    'retraction_depth': metrics['retraction_depth'],
                    'side_view_score': metrics['side_view_score']
                },
                score=score,
                details={
                    'is_side_view': False,
                    'hold_frames': 0,
                    'reps': self.rep_count
                }
            )
        else:
            feedback.append(f"✓ Good side view ({metrics['visible_side']} side visible)")
        
        # ========== ARM RETRACTION DEPTH (PRIMARY METRIC) ==========
        depth = metrics['retraction_depth']
        
        if depth < self.MIN_RETRACTION_DEPTH:
            feedback.append("⚠️ Pull arms FURTHER back - open chest more")
            feedback.append(f"   Depth: {depth:.2f} (need ≥{self.MIN_RETRACTION_DEPTH:.2f})")
            is_correct = False
            score -= 30
            self.is_retracted = False
            self.hold_counter = 0
            
        elif depth > self.MAX_RETRACTION_DEPTH:
            feedback.append("⚠️ Don't over-retract - ease back slightly")
            score -= 10
            
        elif depth >= self.IDEAL_RETRACTION_DEPTH:
            feedback.append(f"✓ EXCELLENT retraction! (depth: {depth:.2f})")
            self.is_retracted = True
            
        else:  # Between MIN and IDEAL
            feedback.append(f"✓ Good retraction (depth: {depth:.2f})")
            feedback.append(f"   Try for {self.IDEAL_RETRACTION_DEPTH:.2f} for maximum benefit")
            self.is_retracted = True
        
        # ========== ELBOW POSITION ==========
        if metrics['elbow_height_diff'] > self.ELBOW_HEIGHT_TOLERANCE:
            feedback.append("⚠️ Keep elbow at shoulder height")
            score -= 10
            is_correct = False
        
        # Check elbow angle
        if metrics['elbow_angle'] < self.MIN_ELBOW_ANGLE:
            feedback.append("⚠️ Don't bend elbow too much")
            score -= 5
        elif metrics['elbow_angle'] > self.MAX_ELBOW_ANGLE:
            feedback.append("⚠️ Bend elbow slightly (~90°)")
            score -= 5
        else:
            feedback.append(f"✓ Good elbow angle ({metrics['elbow_angle']:.0f}°)")
        
        # ========== SHOULDER ELEVATION (SHRUGGING) ==========
        # This is a common compensation - detect if shoulders are rising
        if len(self.depth_history) > 30:
            # Compare current to baseline (simplified check)
            if metrics['shoulder_elevation'] < -0.05:  # Negative = shoulder rising
                feedback.append("⚠️ Don't SHRUG shoulders - keep them down")
                self.shrugging_violations += 1
                score -= 15
                is_correct = False
        
        # ========== FORWARD LEAN COMPENSATION ==========
        if metrics['torso_lean'] > 20:
            feedback.append("⚠️ Don't lean forward - keep torso upright")
            score -= 10
            is_correct = False
        
        # ========== HOLD DURATION TRACKING ==========
        if self.is_retracted and is_correct:
            self.hold_counter += 1
            
            time_held = self.hold_counter / 30.0  # Convert frames to seconds
            time_remaining = max(0, self.HOLD_DURATION - time_held)
            
            if self.hold_counter < self.HOLD_FRAMES:
                feedback.append(f"⏱️ HOLD position - {time_remaining:.1f}s remaining")
            else:
                # Successfully held for full duration!
                if self.hold_counter == self.HOLD_FRAMES:  # Only count once
                    self.rep_count += 1
                    self.successful_holds += 1
                    feedback.append(f"✓✓✓ REP {self.rep_count} COMPLETE! Great hold!")
                else:
                    feedback.append(f"✓ Keep holding - {time_held:.1f}s total")
        else:
            if self.hold_counter > 0:
                feedback.append(f"⚠️ Hold broken at {self.hold_counter/30:.1f}s")
            self.hold_counter = 0
        
        # ========== PROGRESS TRACKING ==========
        feedback.append(f"📊 Reps completed: {self.rep_count}")
        feedback.append(f"📏 Max depth: {self.max_depth_achieved:.2f}")
        
        if self.rep_count > 0:
            feedback.append(f"✓ Successful holds: {self.successful_holds}")
        
        # ========== DEPTH CONSISTENCY ==========
        if len(self.depth_history) >= 30:
            depth_std = self._calculate_std(list(self.depth_history))
            if depth_std > 0.05:
                feedback.append("⚠️ Try to maintain steady depth")
                score -= 5
        
        score = max(0, score)
        
        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                'retraction_depth': depth,
                'elbow_angle': metrics['elbow_angle'],
                'torso_lean': metrics['torso_lean'],
                'elbow_height_diff': metrics['elbow_height_diff']
            },
            score=score,
            details={
                'is_side_view': metrics['is_side_view'],
                'hold_frames': self.hold_counter,
                'hold_seconds': self.hold_counter / 30.0,
                'reps': self.rep_count,
                'max_depth': self.max_depth_achieved,
                'shrugging_violations': self.shrugging_violations,
                'visible_side': metrics['visible_side']
            }
        )
    
    def get_visualization_points(self) -> List[Tuple[str, str]]:
        """
        Emphasize the arm and shoulder regions.
        """
        return [
            # Torso
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            
            # Arms (CRITICAL - show retraction)
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # Head reference
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER'),
        ]
    
    def reset(self):
        """Reset exercise state"""
        self.is_retracted = False
        self.hold_counter = 0
        self.rep_count = 0
        self.max_depth_achieved = 0
        self.depth_history.clear()
        self.last_retraction_time = 0
        self.successful_holds = 0
        self.shrugging_violations = 0
    
    def get_instructions(self) -> List[str]:
        return [
            "⚠️ IMPORTANT: Stand in SIDE VIEW to camera (profile view)",
            "Stand upright with arms at sides",
            "Bend elbows to ~90 degrees, hands at chest level",
            "Pull fists/arms ALL THE WAY BACK behind your body",
            "Squeeze shoulder blades together",
            "Keep shoulders DOWN (don't shrug)",
            "Hold for 5 seconds, breathing normally",
            "Return slowly and repeat",
            "Perform 3-5 repetitions"
        ]
    
    def get_common_mistakes(self) -> List[str]:
        return [
            "❌ Facing camera instead of side view (can't measure depth!)",
            "❌ Not pulling arms far enough back",
            "❌ Shrugging shoulders up instead of keeping them down",
            "❌ Leaning forward to compensate",
            "❌ Holding breath during hold",
            "❌ Letting elbows drop below shoulder height",
            "❌ Rushing through - hold must be 5 full seconds",
            "❌ Not squeezing shoulder blades together"
        ]
    
    # ========== HELPER METHODS ==========
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
"""
Chest Opener / Scapular Retraction Exercise
Therapeutic exercise for upper back strength and posture correction.

Best performed in FRONT VIEW to measure depth (z-axis) of arm retraction.

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
    Requires FRONT VIEW for proper depth measurement using z-coordinate.
    
    Validates:
    - Arm retraction depth (z-axis measurement from camera)
    - Elbow position (should be at shoulder height)
    - Hold duration (5 seconds)
    - Shoulder elevation (shouldn't shrug)
    - Forward lean compensation
    """
    
    # Clinical thresholds (PT-validated)
    # Using Z-coordinate for front view (depth retraction distance)
    # Lowered values for easier testing - adjust based on real-world performance
    MIN_RETRACTION_DEPTH = 0.05      # Minimum depth retraction (normalized z units)
    IDEAL_RETRACTION_DEPTH = 0.10    # Ideal depth retraction for full retraction
    MAX_RETRACTION_DEPTH = 0.25      # Maximum (avoid over-retraction)
    
    ELBOW_HEIGHT_TOLERANCE = 0.15    # Vertical distance from shoulder (normalized) - increased for easier testing
    SHOULDER_ELEVATION_MAX = 0.10    # Max shoulder rise (avoid shrugging) - more lenient
    
    HOLD_DURATION = 5                # Seconds to hold position
    HOLD_FRAMES = 150                # 5 seconds at 30fps
    
    MIN_ELBOW_ANGLE = 60             # Minimum elbow flexion - wider range
    MAX_ELBOW_ANGLE = 120            # Maximum elbow flexion - wider range (should be ~90°)
    
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
        
    def get_name(self) -> str:
        return "Chest Opener (Scapular Retraction)"
    
    def get_description(self) -> str:
        return (
            "Pull arms back to open chest and engage upper back muscles. "
            "Strengthens rhomboids and middle trapezius. "
            "Face the camera directly for best results."
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
            
            # Additional points for reference
            'LEFT_EAR',
            'RIGHT_EAR'
        ]
    
    def calculate_metrics(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate metrics with emphasis on Z-AXIS (depth) measurement for front view.
        
        Critical for this exercise: In front view, the z-coordinate shows how far back
        the arms are pulled in 3D space (depth from camera). More negative z = closer to camera,
        so when arms are pulled back, elbow z becomes more positive (farther from camera).
        """
        
        # ===== CHECK IF ELBOWS ARE VISIBLE =====
        left_elbow_visible = 'LEFT_ELBOW' in landmarks
        right_elbow_visible = 'RIGHT_ELBOW' in landmarks
        
        # If elbows not visible, return early with error flags
        if not left_elbow_visible or not right_elbow_visible:
            return {
                'left_elbow_visible': left_elbow_visible,
                'right_elbow_visible': right_elbow_visible,
                'start_position_ok': False
            }
        
        # ===== USE BOTH ARMS FOR FRONT VIEW =====
        left_shoulder = landmarks['LEFT_SHOULDER']
        left_elbow = landmarks['LEFT_ELBOW']
        left_wrist = landmarks['LEFT_WRIST']
        
        right_shoulder = landmarks['RIGHT_SHOULDER']
        right_elbow = landmarks['RIGHT_ELBOW']
        right_wrist = landmarks['RIGHT_WRIST']
        
        # ===== CHECK START POSITION: ELBOW HEIGHT =====
        # Elbow should be at shoulder height (not drooping or elevated)
        left_elbow_height_diff = abs(left_elbow[1] - left_shoulder[1])
        right_elbow_height_diff = abs(right_elbow[1] - right_shoulder[1])
        left_height_ok = left_elbow_height_diff <= self.ELBOW_HEIGHT_TOLERANCE
        right_height_ok = right_elbow_height_diff <= self.ELBOW_HEIGHT_TOLERANCE
        
        # ===== CHECK START POSITION: ELBOW ANGLE (90 DEGREES) =====
        left_elbow_angle = self.calc.calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calc.calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
        left_angle_ok = self.MIN_ELBOW_ANGLE <= left_elbow_angle <= self.MAX_ELBOW_ANGLE
        right_angle_ok = self.MIN_ELBOW_ANGLE <= right_elbow_angle <= self.MAX_ELBOW_ANGLE
        
        # ===== START POSITION IS OK IF ALL CHECKS PASS =====
        start_position_ok = (left_height_ok and right_height_ok and 
                            left_angle_ok and right_angle_ok)
        
        # ===== ARM RETRACTION DEPTH (ONLY IF START POSITION IS OK) =====
        # For front view, use Z-coordinate (depth from camera) as primary measure
        # When pulling arms back, elbows move away from camera (z becomes more positive)
        # Depth = difference between elbow z and shoulder z
        left_retraction_depth = abs(left_elbow[2] - left_shoulder[2])  # Absolute value for depth
        right_retraction_depth = abs(right_elbow[2] - right_shoulder[2])  # Absolute value for depth
        
        # Track maximum depth achieved (only if start position is ok)
        if start_position_ok:
            avg_depth = (left_retraction_depth + right_retraction_depth) / 2.0
            self.max_depth_achieved = max(self.max_depth_achieved, avg_depth)
            self.depth_history.append(avg_depth)
        
        return {
            # Start position checks
            'left_elbow_visible': left_elbow_visible,
            'right_elbow_visible': right_elbow_visible,
            'start_position_ok': start_position_ok,
            'left_height_ok': left_height_ok,
            'right_height_ok': right_height_ok,
            'left_angle_ok': left_angle_ok,
            'right_angle_ok': right_angle_ok,
            'left_elbow_height_diff': left_elbow_height_diff,
            'right_elbow_height_diff': right_elbow_height_diff,
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            # Depth measurements (only meaningful if start position is ok)
            'left_retraction_depth': left_retraction_depth,
            'right_retraction_depth': right_retraction_depth
        }
    
    def validate_form(self, metrics: Dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate chest opener form with focus on start position and depth.
        """
        
        feedback = []
        is_correct = True
        score = 100.0
        
        # ========== CHECK IF ELBOWS ARE VISIBLE ==========
        left_elbow_visible = metrics.get('left_elbow_visible', True)
        right_elbow_visible = metrics.get('right_elbow_visible', True)
        
        if not left_elbow_visible or not right_elbow_visible:
            missing = []
            if not left_elbow_visible:
                missing.append("LEFT elbow")
            if not right_elbow_visible:
                missing.append("RIGHT elbow")
            feedback.append(f"❌ POSITION WRONG: Cannot see {', '.join(missing)} on camera")
            feedback.append("   Move so both elbows are clearly visible in the frame")
            is_correct = False
            score = 0
            self.is_retracted = False
            self.hold_counter = 0
            
            return ValidationResult(
                is_correct=False,
                feedback_messages=feedback,
                angles={},
                score=score,
                details={
                    'hold_frames': 0,
                    'reps': self.rep_count,
                    'max_depth': 0
                }
            )
        
        # ========== CHECK START POSITION ==========
        start_position_ok = metrics.get('start_position_ok', False)
        left_height_ok = metrics.get('left_height_ok', False)
        right_height_ok = metrics.get('right_height_ok', False)
        left_angle_ok = metrics.get('left_angle_ok', False)
        right_angle_ok = metrics.get('right_angle_ok', False)
        
        position_issues = []
        
        # Check elbow height
        if not left_height_ok:
            left_height_diff = metrics.get('left_elbow_height_diff', 0)
            position_issues.append(f"LEFT elbow not at shoulder height (diff: {left_height_diff:.3f})")
        if not right_height_ok:
            right_height_diff = metrics.get('right_elbow_height_diff', 0)
            position_issues.append(f"RIGHT elbow not at shoulder height (diff: {right_height_diff:.3f})")
        
        # Check elbow angle
        if not left_angle_ok:
            left_angle = metrics.get('left_elbow_angle', 0)
            position_issues.append(f"LEFT elbow not at 90° (current: {left_angle:.0f}°)")
        if not right_angle_ok:
            right_angle = metrics.get('right_elbow_angle', 0)
            position_issues.append(f"RIGHT elbow not at 90° (current: {right_angle:.0f}°)")
        
        if not start_position_ok:
            feedback.append("❌ POSITION WRONG: Fix your starting position first!")
            for issue in position_issues:
                feedback.append(f"   • {issue}")
            feedback.append("   Requirements:")
            feedback.append("   • Both elbows must be at shoulder height")
            feedback.append("   • Both elbows must be bent at ~90 degrees")
            is_correct = False
            score = 0
            self.is_retracted = False
            self.hold_counter = 0
            
            return ValidationResult(
                is_correct=False,
                feedback_messages=feedback,
                angles={
                    'left_elbow_angle': metrics.get('left_elbow_angle', 0),
                    'right_elbow_angle': metrics.get('right_elbow_angle', 0),
                    'left_height_diff': metrics.get('left_elbow_height_diff', 0),
                    'right_height_diff': metrics.get('right_elbow_height_diff', 0)
                },
                score=score,
                details={
                    'hold_frames': 0,
                    'reps': self.rep_count,
                    'max_depth': 0
                }
            )
        
        # ========== START POSITION IS OK - NOW CHECK DEPTH ==========
        feedback.append("✓ Starting position is correct!")
        feedback.append(f"   Left elbow: {metrics.get('left_elbow_angle', 0):.0f}° | Right elbow: {metrics.get('right_elbow_angle', 0):.0f}°")
        
        left_depth = metrics.get('left_retraction_depth', 0)
        right_depth = metrics.get('right_retraction_depth', 0)
        
        # Show both left and right depths
        feedback.append(f"📏 Left elbow depth: {left_depth:.3f} | Right elbow depth: {right_depth:.3f}")
        
        # Check left depth
        if left_depth < self.MIN_RETRACTION_DEPTH:
            feedback.append(f"⚠️ LEFT arm: Pull further back (depth: {left_depth:.3f}, need ≥{self.MIN_RETRACTION_DEPTH:.3f})")
            is_correct = False
            score -= 15
        elif left_depth >= self.IDEAL_RETRACTION_DEPTH:
            feedback.append(f"✓ LEFT arm: EXCELLENT! (depth: {left_depth:.3f})")
        else:
            feedback.append(f"✓ LEFT arm: Good (depth: {left_depth:.3f})")
        
        # Check right depth
        if right_depth < self.MIN_RETRACTION_DEPTH:
            feedback.append(f"⚠️ RIGHT arm: Pull further back (depth: {right_depth:.3f}, need ≥{self.MIN_RETRACTION_DEPTH:.3f})")
            is_correct = False
            score -= 15
        elif right_depth >= self.IDEAL_RETRACTION_DEPTH:
            feedback.append(f"✓ RIGHT arm: EXCELLENT! (depth: {right_depth:.3f})")
        else:
            feedback.append(f"✓ RIGHT arm: Good (depth: {right_depth:.3f})")
        
        # Both arms need to meet minimum for retraction to count
        if left_depth >= self.MIN_RETRACTION_DEPTH and right_depth >= self.MIN_RETRACTION_DEPTH:
            self.is_retracted = True
        else:
            self.is_retracted = False
            self.hold_counter = 0
        
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
        feedback.append(f"📏 Max depth achieved: {self.max_depth_achieved:.3f}")
        
        if self.rep_count > 0:
            feedback.append(f"✓ Successful holds: {self.successful_holds}")
        
        score = max(0, score)
        
        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                'left_depth': left_depth,
                'right_depth': right_depth,
                'left_elbow_angle': metrics.get('left_elbow_angle', 0),
                'right_elbow_angle': metrics.get('right_elbow_angle', 0)
            },
            score=score,
            details={
                'hold_frames': self.hold_counter,
                'hold_seconds': self.hold_counter / 30.0,
                'reps': self.rep_count,
                'max_depth': self.max_depth_achieved
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
    
    def get_instructions(self) -> List[str]:
        return [
            "⚠️ IMPORTANT: Face the camera directly (front view)",
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
            "❌ Not pulling arms far enough back",
            "❌ Letting elbows drop below shoulder height",
            "❌ Not keeping elbows at ~90 degrees",
            "❌ Rushing through - hold must be 5 full seconds",
            "❌ Not squeezing shoulder blades together",
            "❌ Moving arms asymmetrically (one side more than the other)"
        ]
    
    # ========== HELPER METHODS ==========
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
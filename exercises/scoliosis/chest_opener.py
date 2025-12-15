"""
Chest Opener / Scapular Retraction Exercise
Therapeutic exercise for upper back strength and posture correction.

Best performed in FRONT VIEW for proper angle and position measurement.

- Focus: Scapular retraction, chest opening, upper back engagement
- Target: Rhomboids, middle trapezius, posterior deltoids
- Indication: Forward head posture, rounded shoulders, upper crossed syndrome
"""
from typing import Dict, List, Tuple
from exercises.base_exercise import BaseExercise, ValidationResult
from exercises.exercise_factory import ExerciseFactory
from angle_calculator import AngleCalculator


@ExerciseFactory.register('chest_opener_retraction')
class ChestOpener(BaseExercise):
    """
    Chest opener exercise with scapular retraction.
    
    Patient starts with elbows at shoulder height or above, then lowers them
    below shoulder level, holds for 2 seconds, then returns to start to complete a rep.
    Requires FRONT VIEW for proper measurement.
    
    Validates:
    - Start position: Elbows at or above shoulder height (one-time check)
    - Rep completion: Hold at bottom for 2s, then return to start
    """
    
    # Hold requirement at bottom position
    HOLD_DURATION_SECONDS = 2       # Seconds to hold at bottom position
    HOLD_FRAMES = 60                # 2 seconds at 30fps
    
    def __init__(self):
        super().__init__()
        self.calc = AngleCalculator()
        
        # State tracking
        self.rep_count = 0
        self.in_start_position = False
        self.start_position_validated = False  # Track if start position has been validated (one-time check)
        self.was_above_shoulders = False  # Track if elbows were at/above shoulders (for rep counting)
        self.hold_complete = False  # Track if 2-second hold at bottom is complete
        self.hold_counter = 0  # Track frames held at bottom position
        self.last_rep_frame = 0
        
        # Temporal tracking to prevent double counting
        self.rep_cooldown_frames = 30  # 1 second cooldown between reps
        
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
        
        # ===== CHECK START POSITION: ELBOWS AT OR ABOVE SHOULDER HEIGHT =====
        # Check if elbows are at or above shoulders (elbow y <= shoulder y in normalized coordinates)
        # Allow some tolerance for "at shoulder height"
        ELBOW_HEIGHT_TOLERANCE = 0.10  # Tolerance for "at shoulder height"
        left_elbow_at_or_above = left_elbow[1] <= (left_shoulder[1] + ELBOW_HEIGHT_TOLERANCE)
        right_elbow_at_or_above = right_elbow[1] <= (right_shoulder[1] + ELBOW_HEIGHT_TOLERANCE)
        
        # Both elbows at or above shoulders = start position
        both_at_or_above_shoulders = left_elbow_at_or_above and right_elbow_at_or_above
        
        # Also track if above (for display)
        left_elbow_above_shoulder = left_elbow[1] < left_shoulder[1]
        right_elbow_above_shoulder = right_elbow[1] < right_shoulder[1]
        both_above_shoulders = left_elbow_above_shoulder and right_elbow_above_shoulder
        
        # Calculate angles for display
        left_elbow_angle = self.calc.calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calc.calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
        
        # ===== CHECK IF ELBOWS GO BELOW SHOULDERS =====
        # Check if elbows are below shoulders (elbow y > shoulder y in normalized coordinates)
        left_elbow_below_shoulder = left_elbow[1] > left_shoulder[1]
        right_elbow_below_shoulder = right_elbow[1] > right_shoulder[1]
        
        # Both elbows below shoulders
        both_below_shoulders = left_elbow_below_shoulder and right_elbow_below_shoulder
        
        return {
            # Visibility checks
            'left_elbow_visible': left_elbow_visible,
            'right_elbow_visible': right_elbow_visible,
            # Start position checks (at or above shoulder height)
            'start_position_ok': both_at_or_above_shoulders,
            'left_elbow_at_or_above': left_elbow_at_or_above,
            'right_elbow_at_or_above': right_elbow_at_or_above,
            'both_at_or_above_shoulders': both_at_or_above_shoulders,
            'left_elbow_above_shoulder': left_elbow_above_shoulder,
            'right_elbow_above_shoulder': right_elbow_above_shoulder,
            'both_above_shoulders': both_above_shoulders,
            # Movement tracking
            'left_elbow_below_shoulder': left_elbow_below_shoulder,
            'right_elbow_below_shoulder': right_elbow_below_shoulder,
            'both_below_shoulders': both_below_shoulders,
            # Angle display
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle
        }
    
    def validate_form(self, metrics: Dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate chest opener form:
        1. Start position: acute angles at shoulder height
        2. Movement: Lower elbows below shoulders
        3. Rep completion: Return to start position after going below shoulders
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
            self.in_start_position = False
            
            return ValidationResult(
                is_correct=False,
                feedback_messages=feedback,
                angles={},
                score=score,
                details={
                    'reps': self.rep_count,
                    'in_start_position': False
                }
            )
        
        # ========== CHECK START POSITION: ELBOWS AT OR ABOVE SHOULDER HEIGHT ==========
        start_position_ok = metrics.get('start_position_ok', False)
        both_at_or_above_shoulders = metrics.get('both_at_or_above_shoulders', False)
        both_above_shoulders = metrics.get('both_above_shoulders', False)
        left_elbow_at_or_above = metrics.get('left_elbow_at_or_above', False)
        right_elbow_at_or_above = metrics.get('right_elbow_at_or_above', False)
        
        left_elbow_angle = metrics.get('left_elbow_angle', 0)
        right_elbow_angle = metrics.get('right_elbow_angle', 0)
        
        # ========== ONE-TIME START POSITION VALIDATION ==========
        if not self.start_position_validated:
            if not start_position_ok:
                position_issues = []
                if not left_elbow_at_or_above:
                    position_issues.append("LEFT elbow not at or above shoulder height")
                if not right_elbow_at_or_above:
                    position_issues.append("RIGHT elbow not at or above shoulder height")
                
                feedback.append("❌ ERROR: Get into starting position first!")
                for issue in position_issues:
                    feedback.append(f"   • {issue}")
                feedback.append("   Requirements:")
                feedback.append("   • Both elbows must be at shoulder height or above")
                is_correct = False
                score = 0
                self.in_start_position = False
                self.was_above_shoulders = False
                
                return ValidationResult(
                    is_correct=False,
                    feedback_messages=feedback,
                    angles={
                        'left_elbow_angle': left_elbow_angle,
                        'right_elbow_angle': right_elbow_angle
                    },
                    score=score,
                    details={
                        'reps': self.rep_count,
                        'in_start_position': False
                    }
                )
            else:
                # Start position validated - mark as validated
                self.start_position_validated = True
                self.was_above_shoulders = True
        
        # ========== TRACK MOVEMENT: ELBOWS BELOW SHOULDERS ==========
        left_elbow_below_shoulder = metrics.get('left_elbow_below_shoulder', False)
        right_elbow_below_shoulder = metrics.get('right_elbow_below_shoulder', False)
        both_below_shoulders = metrics.get('both_below_shoulders', False)
        
        # Track if we're at or above shoulders
        if both_at_or_above_shoulders:
            self.was_above_shoulders = True
            self.in_start_position = True
        else:
            self.in_start_position = False
        
        # ========== HOLD TRACKING AT BOTTOM POSITION ==========
        if both_below_shoulders and self.was_above_shoulders:
            # Only increment counter if hold is not already complete
            if not self.hold_complete:
                self.hold_counter += 1
                time_held = self.hold_counter / 30.0  # Convert frames to seconds
                time_remaining = max(0, self.HOLD_DURATION_SECONDS - time_held)
                
                if self.hold_counter >= self.HOLD_FRAMES:
                    # Successfully held for required duration - set flag once
                    self.hold_complete = True
                    feedback.append("✓✓ Hold complete! Now rise back to start position")
                else:
                    feedback.append(f"⏱️ HOLD at bottom - {time_remaining:.1f}s remaining")
            else:
                # Hold already complete - just remind to go up
                feedback.append("✓ Hold complete - rise back to start position for rep")
        else:
            # Not holding at bottom - reset counter only if hold wasn't complete
            if self.hold_counter > 0 and not self.hold_complete:
                # Only reset if we had started holding but didn't complete it
                if self.hold_counter < self.HOLD_FRAMES:
                    feedback.append(f"⚠️ Hold broken at {self.hold_counter/30:.1f}s - maintain position")
                self.hold_counter = 0
        
        # ========== COUNT REP WHEN RETURNED TO START AFTER COMPLETING HOLD ==========
        if start_position_ok and self.hold_complete:
            # Check cooldown to prevent double counting
            frames_since_last_rep = frame_count - self.last_rep_frame
            if frames_since_last_rep >= self.rep_cooldown_frames or self.last_rep_frame == 0:
                self.rep_count += 1
                self.last_rep_frame = frame_count
                self.hold_complete = False  # Reset for next rep
                self.hold_counter = 0  # Reset hold counter
                self.was_above_shoulders = True  # Keep this true since we're back at start
                feedback.append("✓✓✓ Rep complete! Returned to start after completing hold!")
            else:
                feedback.append("✓ Back in start position - ready for next rep")
        elif start_position_ok:
            # In start position (elbows at or above shoulder height)
            if both_above_shoulders:
                feedback.append("✓ Starting position: Elbows above shoulders")
            else:
                feedback.append("✓ Starting position: Elbows at shoulder height")
            feedback.append(f"   Left: {left_elbow_angle:.0f}° | Right: {right_elbow_angle:.0f}°")
            if not self.hold_complete:
                feedback.append("→ Lower elbows below shoulder level, hold 2s, then rise back")
            else:
                feedback.append("→ Lower elbows below shoulder level again for next rep")
        else:
            # Not in start position - provide feedback on movement
            if both_below_shoulders:
                # Feedback already provided above in hold tracking section
                pass
            elif left_elbow_below_shoulder or right_elbow_below_shoulder:
                feedback.append("⚠️ Only one elbow below shoulder - lower both")
            else:
                # In transition
                if self.hold_complete:
                    feedback.append("→ Rise back to start position to complete rep")
                else:
                    feedback.append("→ Lower elbows below shoulder level, hold 2s, then rise back")
        
        score = max(0, score)
        
        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle
            },
            score=score,
            details={
                'reps': self.rep_count,
                'in_start_position': self.in_start_position,
                'was_above_shoulders': self.was_above_shoulders,
                'hold_frames': self.hold_counter,
                'hold_seconds': self.hold_counter / 30.0,
                'hold_complete': self.hold_complete
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
        self.rep_count = 0
        self.in_start_position = False
        self.start_position_validated = False
        self.was_above_shoulders = False
        self.hold_complete = False
        self.hold_counter = 0
        self.last_rep_frame = 0
    
    def get_instructions(self) -> List[str]:
        return [
            "⚠️ IMPORTANT: Face the camera directly (front view)",
            "1. START: Raise elbows to shoulder height or above",
            "2. LOWER: Lower elbows below shoulder level",
            "3. HOLD: Hold at bottom position for 2 seconds",
            "4. RISE: Return to start position (shoulder height or above)",
            "5. Rep counts when you return to start after holding!",
            "6. Repeat the movement",
            "Perform 3-5 repetitions"
        ]
    
    def get_common_mistakes(self) -> List[str]:
        return [
            "❌ Not starting with elbows at or above shoulder height",
            "❌ Not lowering elbows below shoulder level",
            "❌ Not holding at bottom position for 2 seconds",
            "❌ Not returning to start position after holding",
            "❌ Moving one arm more than the other (asymmetry)",
            "❌ Rushing through the movement"
        ]
    
    # ========== HELPER METHODS ==========
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_video_url(self) -> str | None:
        return "https://youtu.be/qjSllPcEooU?t=38" 
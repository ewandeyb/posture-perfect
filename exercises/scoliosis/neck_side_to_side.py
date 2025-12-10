"""
Neck Side to Side Exercise
Therapeutic exercise for neck lateral flexion and mobility.

Validates:
- Left shoulder and elbow crossing the vertical midpoint when tilting head left
- Right shoulder and elbow crossing the vertical midpoint when tilting head right
"""
from exercises.base_exercise import BaseExercise, ValidationResult
from exercises.exercise_factory import ExerciseFactory


@ExerciseFactory.register('neck_side_to_side')
class NeckSideToSide(BaseExercise):
    """
    Neck side to side exercise for lateral neck flexion.
    
    Both shoulders must cross the midpoint together in the same direction.
    - RIGHT tilt: Both shoulders cross to the right (x > 0.5)
    - LEFT tilt: Both shoulders cross to the left (x < 0.5)
    """
    
    # Vertical midpoint (normalized coordinates, 0-1)
    MIDPOINT_X = 0.5
    
    def __init__(self):
        super().__init__()
        self.rep_count = 0
        self.left_reps = 0
        self.right_reps = 0
        self._left_was_crossed = False
        self._right_was_crossed = False
        
    def get_name(self) -> str:
        return "Neck Side to Side"
    
    def get_description(self) -> str:
        return (
            "Lateral neck flexion exercise. Tilt your head to each side, "
            "ensuring all points (shoulders, elbows, wrists) cross the vertical midpoint together. "
            "Improves neck mobility and flexibility."
        )
    
    def get_required_landmarks(self) -> list[str]:
        return [
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_ELBOW',
            'RIGHT_ELBOW',
            'LEFT_WRIST',
            'RIGHT_WRIST',
        ]
    
    def calculate_metrics(self, landmarks: dict) -> dict[str, float]:
        """
        Calculate metrics for neck side to side exercise.
        
        All points (shoulders, elbows, wrists) must cross the midpoint together.
        - RIGHT tilt: All 6 points cross to right (x > 0.5)
        - LEFT tilt: All 6 points cross to left (x < 0.5)
        """
        # Check if all required landmarks are visible
        if not landmarks.get('_all_visible', True):
            missing_points = landmarks.get('_missing_points', [])
            return {
                'missing_points': missing_points,
                'all_visible': False
            }
        
        left_shoulder = landmarks['LEFT_SHOULDER']
        right_shoulder = landmarks['RIGHT_SHOULDER']
        left_elbow = landmarks['LEFT_ELBOW']
        right_elbow = landmarks['RIGHT_ELBOW']
        left_wrist = landmarks['LEFT_WRIST']
        right_wrist = landmarks['RIGHT_WRIST']
        
        # Extract x-coordinates (normalized 0-1)
        left_shoulder_x = left_shoulder[0]
        right_shoulder_x = right_shoulder[0]
        left_elbow_x = left_elbow[0]
        right_elbow_x = right_elbow[0]
        left_wrist_x = left_wrist[0]
        right_wrist_x = right_wrist[0]
        
        # Check if all points crossed to the right (RIGHT tilt)
        left_shoulder_crossed_right = left_shoulder_x > self.MIDPOINT_X
        right_shoulder_crossed_right = right_shoulder_x > self.MIDPOINT_X
        left_elbow_crossed_right = left_elbow_x > self.MIDPOINT_X
        right_elbow_crossed_right = right_elbow_x > self.MIDPOINT_X
        left_wrist_crossed_right = left_wrist_x > self.MIDPOINT_X
        right_wrist_crossed_right = right_wrist_x > self.MIDPOINT_X
        
        all_crossed_right = (left_shoulder_crossed_right and right_shoulder_crossed_right and
                            left_elbow_crossed_right and right_elbow_crossed_right and
                            left_wrist_crossed_right and right_wrist_crossed_right)
        
        # Check if all points crossed to the left (LEFT tilt)
        left_shoulder_crossed_left = left_shoulder_x < self.MIDPOINT_X
        right_shoulder_crossed_left = right_shoulder_x < self.MIDPOINT_X
        left_elbow_crossed_left = left_elbow_x < self.MIDPOINT_X
        right_elbow_crossed_left = right_elbow_x < self.MIDPOINT_X
        left_wrist_crossed_left = left_wrist_x < self.MIDPOINT_X
        right_wrist_crossed_left = right_wrist_x < self.MIDPOINT_X
        
        all_crossed_left = (left_shoulder_crossed_left and right_shoulder_crossed_left and
                           left_elbow_crossed_left and right_elbow_crossed_left and
                           left_wrist_crossed_left and right_wrist_crossed_left)
        
        return {
            'all_visible': True,
            'left_shoulder_x': left_shoulder_x,
            'right_shoulder_x': right_shoulder_x,
            'left_elbow_x': left_elbow_x,
            'right_elbow_x': right_elbow_x,
            'left_wrist_x': left_wrist_x,
            'right_wrist_x': right_wrist_x,
            # Right tilt: all 6 points crossed right
            'all_crossed_right': all_crossed_right,
            'left_shoulder_crossed_right': left_shoulder_crossed_right,
            'right_shoulder_crossed_right': right_shoulder_crossed_right,
            'left_elbow_crossed_right': left_elbow_crossed_right,
            'right_elbow_crossed_right': right_elbow_crossed_right,
            'left_wrist_crossed_right': left_wrist_crossed_right,
            'right_wrist_crossed_right': right_wrist_crossed_right,
            # Left tilt: all 6 points crossed left
            'all_crossed_left': all_crossed_left,
            'left_shoulder_crossed_left': left_shoulder_crossed_left,
            'right_shoulder_crossed_left': right_shoulder_crossed_left,
            'left_elbow_crossed_left': left_elbow_crossed_left,
            'right_elbow_crossed_left': right_elbow_crossed_left,
            'left_wrist_crossed_left': left_wrist_crossed_left,
            'right_wrist_crossed_left': right_wrist_crossed_left
        }
    
    def validate_form(self, metrics: dict[str, float], frame_count: int) -> ValidationResult:
        """
        Validate neck side to side exercise form.
        
        All 6 points (shoulders, elbows, wrists) must cross the midpoint together.
        """
        feedback = []
        is_correct = False
        score = 0.0
        
        # Check if all points are visible first
        if not metrics.get('all_visible', True):
            missing_points = metrics.get('missing_points', [])
            point_names = {
                'LEFT_SHOULDER': 'Left shoulder',
                'RIGHT_SHOULDER': 'Right shoulder',
                'LEFT_ELBOW': 'Left elbow',
                'RIGHT_ELBOW': 'Right elbow',
                'LEFT_WRIST': 'Left wrist',
                'RIGHT_WRIST': 'Right wrist'
            }
            missing_names = [point_names.get(p, p) for p in missing_points]
            feedback.append("❌ ERROR: Cannot see all required points!")
            feedback.append(f"   Missing: {', '.join(missing_names)}")
            feedback.append("   Please move into frame so all points are visible")
            feedback.append("   Make sure both arms are fully visible in the camera")
            
            return ValidationResult(
                is_correct=False,
                feedback_messages=feedback,
                angles={},
                score=0.0,
                details={
                    'missing_points': missing_points,
                    'all_visible': False
                }
            )
        
        all_crossed_right = metrics['all_crossed_right']
        all_crossed_left = metrics['all_crossed_left']
        
        # ========== RIGHT TILT CHECK ==========
        if all_crossed_right:
            feedback.append("✓✓✓ RIGHT TILT: All points (shoulders, elbows, wrists) crossed midpoint!")
            is_correct = True
            score = 100.0
        else:
            # Check which points are missing
            missing = []
            if not metrics['left_shoulder_crossed_right']:
                missing.append("Left shoulder")
            if not metrics['right_shoulder_crossed_right']:
                missing.append("Right shoulder")
            if not metrics['left_elbow_crossed_right']:
                missing.append("Left elbow")
            if not metrics['right_elbow_crossed_right']:
                missing.append("Right elbow")
            if not metrics['left_wrist_crossed_right']:
                missing.append("Left wrist")
            if not metrics['right_wrist_crossed_right']:
                missing.append("Right wrist")
            
            if missing:
                feedback.append(f"⚠️ RIGHT TILT: {', '.join(missing)} not crossing midpoint")
                feedback.append("   All 6 points must cross together (x > 0.5)")
                # Partial score based on how many crossed
                crossed_count = 6 - len(missing)
                score = (crossed_count / 6.0) * 100.0
            else:
                feedback.append("→ Tilt head RIGHT - all points should cross midpoint")
                score = 0.0
        
        # ========== LEFT TILT CHECK ==========
        if all_crossed_left:
            feedback.append("✓✓✓ LEFT TILT: All points (shoulders, elbows, wrists) crossed midpoint!")
            is_correct = True
            score = 100.0
        else:
            # Check which points are missing (only if not doing right tilt)
            if not all_crossed_right:
                missing = []
                if not metrics['left_shoulder_crossed_left']:
                    missing.append("Left shoulder")
                if not metrics['right_shoulder_crossed_left']:
                    missing.append("Right shoulder")
                if not metrics['left_elbow_crossed_left']:
                    missing.append("Left elbow")
                if not metrics['right_elbow_crossed_left']:
                    missing.append("Right elbow")
                if not metrics['left_wrist_crossed_left']:
                    missing.append("Left wrist")
                if not metrics['right_wrist_crossed_left']:
                    missing.append("Right wrist")
                
                if missing:
                    feedback.append(f"⚠️ LEFT TILT: {', '.join(missing)} not crossing midpoint")
                    feedback.append("   All 6 points must cross together (x < 0.5)")
                    # Partial score based on how many crossed
                    crossed_count = 6 - len(missing)
                    score = (crossed_count / 6.0) * 100.0
                else:
                    if score == 0.0:  # Only show if not already showing right tilt message
                        feedback.append("→ Tilt head LEFT - all points should cross midpoint")
                    score = 0.0
        
        # ========== PROGRESS TRACKING ==========
        # Rep count removed - now shown in main UI widget
        
        # Rep counting: increment when all points cross, then return to neutral
        if all_crossed_right and not self._right_was_crossed:
            self._right_was_crossed = True
        elif not all_crossed_right and self._right_was_crossed:
            self.right_reps += 1
            self.rep_count += 1
            self._right_was_crossed = False
            feedback.append(f"✓✓✓ RIGHT REP {self.right_reps} COMPLETE!")
        
        if all_crossed_left and not self._left_was_crossed:
            self._left_was_crossed = True
        elif not all_crossed_left and self._left_was_crossed:
            self.left_reps += 1
            self.rep_count += 1
            self._left_was_crossed = False
            feedback.append(f"✓✓✓ LEFT REP {self.left_reps} COMPLETE!")
        
        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                'left_shoulder_x': metrics['left_shoulder_x'] * 100,
                'right_shoulder_x': metrics['right_shoulder_x'] * 100,
                'left_elbow_x': metrics['left_elbow_x'] * 100,
                'right_elbow_x': metrics['right_elbow_x'] * 100,
                'left_wrist_x': metrics['left_wrist_x'] * 100,
                'right_wrist_x': metrics['right_wrist_x'] * 100
            },
            score=score,
            details={
                'all_crossed_right': all_crossed_right,
                'all_crossed_left': all_crossed_left,
                'reps': self.rep_count,
                'left_reps': self.left_reps,
                'right_reps': self.right_reps
            }
        )
    
    def get_visualization_points(self) -> list[tuple[str, str]]:
        """
        Show the upper body for neck side to side visualization.
        """
        return [
            # Shoulders
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            # Left arm
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            # Right arm
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ]
    
    def reset(self):
        """Reset exercise state"""
        self.rep_count = 0
        self.left_reps = 0
        self.right_reps = 0
        self._left_was_crossed = False
        self._right_was_crossed = False
    
    def get_instructions(self) -> list[str]:
      return [
          "Sit or stand upright facing the camera.",
          "Start with both hands on your head",
          "Slowly tilt your head to the RIGHT — bringing your ear toward your shoulder, and your hands should follow",
          "Pause for 1–2 seconds, then return to neutral.",
          "Slowly tilt your head to the LEFT in the same way.",
          "Pause for 1–2 seconds, then return to neutral.",
          "Repeat for 5–10 reps per side."
      ]
    def get_common_mistakes(self) -> list[str]:
        return [
            "❌ Not tilting head far enough - shoulder doesn't cross midpoint",
            "❌ Moving too quickly - need controlled, slow movement",
            "❌ Not holding the end position briefly",
            "❌ Tilting forward or backward instead of side to side",
            "❌ Not working both sides equally"
        ]
    
    def get_video_url(self) -> str | None:
        return "https://youtu.be/qjSllPcEooU?t=156"
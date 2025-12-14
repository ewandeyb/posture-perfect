"""
Wall Angels Exercise
Shoulder mobility and posture correction exercise.
"""

import math
from collections import deque
from typing import Dict, List, Tuple

from exercises.base_exercise import BaseExercise, ValidationResult
from exercises.exercise_factory import ExerciseFactory


@ExerciseFactory.register("wall_angels")
class WallAngels(BaseExercise):
    """
    Wall Angels exercise for shoulder mobility and posture.

    Validates:
    - Back flat against wall (torso alignment)
    - Arms raised overhead in a controlled motion
    - Elbows staying close to wall
    - Full range of motion (down to shoulder level, up overhead)
    - Symmetric arm movement

    Benefits:
    - Improves shoulder mobility
    - Strengthens upper back muscles
    - Corrects rounded shoulder posture
    - Increases thoracic spine extension
    """

    # Arm angle thresholds (measured from horizontal)
    MIN_ARM_ANGLE_DOWN = 70  # Arms down position (near shoulder level)
    MAX_ARM_ANGLE_DOWN = 110  # Allow some range
    MIN_ARM_ANGLE_UP = 160  # Arms raised overhead
    MAX_ARM_ANGLE_UP = 190  # Slightly past vertical

    # Symmetry thresholds
    MAX_ARM_ANGLE_DIFF = 20  # Max difference between left and right arm angles

    # Elbow position (should stay back near wall)
    MAX_ELBOW_FORWARD = 0.15  # Max forward distance from shoulders (normalized)

    # Position states
    STATE_NEUTRAL = "neutral"
    STATE_ARMS_DOWN = "arms_down"
    STATE_ARMS_UP = "arms_up"
    STATE_TRANSITION = "transition"

    def __init__(self):
        super().__init__()

        # State tracking
        self.current_state = self.STATE_NEUTRAL
        self.last_state = self.STATE_NEUTRAL
        self.rep_count = 0

        # Movement tracking
        self.arm_angle_history = deque(maxlen=90)  # 3 seconds at 30fps

        # Quality metrics
        self.symmetry_violations = 0
        self.elbow_violations = 0

        # Rep detection flags
        self._down_position_reached = False
        self._up_position_reached = False

    def get_name(self) -> str:
        return "Wall Angels"

    def get_description(self) -> str:
        return (
            "Shoulder mobility exercise performed with back against a wall. "
            "Slide arms up and down the wall in a 'snow angel' motion while "
            "keeping back flat and elbows close to the wall. Improves posture "
            "and shoulder mobility."
        )

    def get_required_landmarks(self) -> List[str]:
        return [
            # Head
            "NOSE",
            # Upper body
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            # Core (for posture check)
            "LEFT_HIP",
            "RIGHT_HIP",
        ]

    def calculate_metrics(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate all biomechanical metrics for wall angels.

        Metrics:
        - left_arm_angle: Angle of left arm from horizontal
        - right_arm_angle: Angle of right arm from horizontal
        - arm_symmetry: Difference between left and right arm angles
        - left_elbow_forward: How far left elbow is in front of shoulder
        - right_elbow_forward: How far right elbow is in front of shoulder
        - torso_alignment: Back flatness against wall
        """

        # Arm angles (shoulder-elbow-wrist angle from horizontal)
        left_arm_angle = self._calculate_arm_angle_from_horizontal(
            landmarks["LEFT_SHOULDER"], landmarks["LEFT_ELBOW"], landmarks["LEFT_WRIST"]
        )

        right_arm_angle = self._calculate_arm_angle_from_horizontal(
            landmarks["RIGHT_SHOULDER"],
            landmarks["RIGHT_ELBOW"],
            landmarks["RIGHT_WRIST"],
        )

        # Arm symmetry
        arm_symmetry_diff = abs(left_arm_angle - right_arm_angle)

        # Elbow position (check if elbows are forward of shoulders)
        # In side view, z-coordinate indicates depth
        left_elbow_forward = abs(
            landmarks["LEFT_ELBOW"][2] - landmarks["LEFT_SHOULDER"][2]
        )
        right_elbow_forward = abs(
            landmarks["RIGHT_ELBOW"][2] - landmarks["RIGHT_SHOULDER"][2]
        )

        # Alternative: use y-coordinate difference if viewing from side
        # If elbow is higher than shoulder in image, might indicate forward position
        left_elbow_forward_alt = max(
            0, landmarks["LEFT_SHOULDER"][1] - landmarks["LEFT_ELBOW"][1]
        )
        right_elbow_forward_alt = max(
            0, landmarks["RIGHT_SHOULDER"][1] - landmarks["RIGHT_ELBOW"][1]
        )

        # Use the larger of the two measures
        left_elbow_forward = max(left_elbow_forward, left_elbow_forward_alt * 0.5)
        right_elbow_forward = max(right_elbow_forward, right_elbow_forward_alt * 0.5)

        # Torso alignment (shoulder-hip alignment)
        shoulder_mid = self._midpoint(
            landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"]
        )
        hip_mid = self._midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])

        # Calculate if torso is upright (close to vertical)
        torso_angle = self._angle_from_vertical_2d(shoulder_mid, hip_mid)

        return {
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle,
            "arm_symmetry_diff": arm_symmetry_diff,
            "left_elbow_forward": left_elbow_forward,
            "right_elbow_forward": right_elbow_forward,
            "torso_angle": abs(torso_angle),
            "avg_arm_angle": (left_arm_angle + right_arm_angle) / 2,
        }

    def validate_form(
        self, metrics: Dict[str, float], frame_count: int
    ) -> ValidationResult:
        """
        Validate wall angels form with state machine and rep counting.
        """

        feedback = []
        is_correct = True
        score = 100.0

        left_arm = metrics["left_arm_angle"]
        right_arm = metrics["right_arm_angle"]
        avg_arm = metrics["avg_arm_angle"]
        symmetry = metrics["arm_symmetry_diff"]

        # Add to history
        self.arm_angle_history.append(avg_arm)

        # ========== STATE DETECTION ==========
        previous_state = self.current_state

        # Detect arm position
        if self.MIN_ARM_ANGLE_DOWN <= avg_arm <= self.MAX_ARM_ANGLE_DOWN:
            self.current_state = self.STATE_ARMS_DOWN
        elif self.MIN_ARM_ANGLE_UP <= avg_arm <= self.MAX_ARM_ANGLE_UP:
            self.current_state = self.STATE_ARMS_UP
        else:
            self.current_state = self.STATE_TRANSITION

        # ========== REP COUNTING ==========
        # Rep complete when: arms down -> arms up -> arms down
        if self.current_state == self.STATE_ARMS_DOWN:
            self._down_position_reached = True
            if self._up_position_reached:
                # Completed a full rep (up and back down)
                self.rep_count += 1
                feedback.append(f"✓✓✓ REP {self.rep_count} COMPLETE!")
                self._up_position_reached = False

        elif self.current_state == self.STATE_ARMS_UP:
            if self._down_position_reached:
                self._up_position_reached = True
                self._down_position_reached = False

        # ========== POSITION VALIDATION ==========

        # 1. Check arm angles
        if self.current_state == self.STATE_ARMS_DOWN:
            if avg_arm < self.MIN_ARM_ANGLE_DOWN:
                feedback.append("⚠️ Raise arms higher to shoulder level")
                score -= 10
                is_correct = False
            elif avg_arm > self.MAX_ARM_ANGLE_DOWN:
                feedback.append("⚠️ Lower arms to shoulder level")
                score -= 10
                is_correct = False
            else:
                feedback.append("✓ Good starting position - arms at shoulder level")

        elif self.current_state == self.STATE_ARMS_UP:
            if avg_arm < self.MIN_ARM_ANGLE_UP:
                feedback.append("⚠️ Raise arms higher - reach overhead")
                score -= 15
                is_correct = False
            elif avg_arm > self.MAX_ARM_ANGLE_UP:
                feedback.append("⚠️ Don't over-extend - keep controlled")
                score -= 10
                is_correct = False
            else:
                feedback.append("✓ Excellent! Arms fully raised overhead")

        elif self.current_state == self.STATE_TRANSITION:
            feedback.append("→ Move slowly and controlled")

        else:
            feedback.append("ℹ️ Start with arms at shoulder level")

        # 2. Check arm symmetry (critical for posture correction)
        if symmetry > self.MAX_ARM_ANGLE_DIFF:
            feedback.append(f"⚠️ Keep arms symmetrical - {symmetry:.1f}° difference")
            self.symmetry_violations += 1
            score -= 15
            is_correct = False
        else:
            feedback.append("✓ Good arm symmetry")

        # 3. Check elbow position (should stay back near wall)
        left_elbow_fwd = metrics["left_elbow_forward"]
        right_elbow_fwd = metrics["right_elbow_forward"]

        if left_elbow_fwd > self.MAX_ELBOW_FORWARD:
            feedback.append("⚠️ Keep LEFT elbow back against wall")
            self.elbow_violations += 1
            score -= 10
            is_correct = False

        if right_elbow_fwd > self.MAX_ELBOW_FORWARD:
            feedback.append("⚠️ Keep RIGHT elbow back against wall")
            self.elbow_violations += 1
            score -= 10
            is_correct = False

        if (
            left_elbow_fwd <= self.MAX_ELBOW_FORWARD
            and right_elbow_fwd <= self.MAX_ELBOW_FORWARD
        ):
            feedback.append("✓ Elbows staying close to wall")

        # 4. Check torso alignment
        if metrics["torso_angle"] > 15:
            feedback.append("⚠️ Keep back flat against wall")
            score -= 10

        # ========== PROGRESS TRACKING ==========
        # Rep count removed from feedback - now shown in main UI widget

        score = max(0, score)

        return ValidationResult(
            is_correct=is_correct,
            feedback_messages=feedback,
            angles={
                "left_arm_angle": left_arm,
                "right_arm_angle": right_arm,
                "arm_symmetry": symmetry,
                "torso_angle": metrics["torso_angle"],
            },
            score=score,
            details={
                "state": self.current_state,
                "reps": self.rep_count,
                "symmetry_violations": self.symmetry_violations,
                "elbow_violations": self.elbow_violations,
                "down_reached": self._down_position_reached,
                "up_reached": self._up_position_reached,
            },
        )

    def get_visualization_points(self) -> List[Tuple[str, str]]:
        """Define skeleton connections to draw"""
        return [
            # Head to shoulders
            ("NOSE", "LEFT_SHOULDER"),
            ("NOSE", "RIGHT_SHOULDER"),
            # Shoulders
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            # Torso
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            # Left arm
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            # Right arm
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ]

    def reset(self):
        """Reset all exercise state"""
        self.current_state = self.STATE_NEUTRAL
        self.last_state = self.STATE_NEUTRAL
        self.rep_count = 0
        self.arm_angle_history.clear()
        self.symmetry_violations = 0
        self.elbow_violations = 0
        self._down_position_reached = False
        self._up_position_reached = False

    def get_instructions(self) -> List[str]:
        return [
            "Stand with your back flat against a wall",
            "Feet should be about 6 inches from wall",
            "Press lower back, shoulders, and head against wall",
            "Start with arms at shoulder height, elbows bent 90°",
            "Slowly slide arms up the wall overhead",
            "Keep elbows and wrists touching the wall throughout",
            "Slowly slide arms back down to starting position",
            "Move smoothly and controlled - 3 seconds up, 3 seconds down",
            "Perform 10-15 repetitions",
        ]

    def get_common_mistakes(self) -> List[str]:
        return [
            "Lower back arching away from wall",
            "Elbows coming forward off the wall",
            "Moving arms too quickly - should be slow and controlled",
            "Not raising arms fully overhead",
            "Asymmetric arm movement (one arm higher than other)",
            "Holding breath - remember to breathe steadily",
            "Shoulders shrugging up toward ears",
            "Head tilting forward instead of staying against wall",
            "Wrists bending instead of staying flat",
            "Not maintaining contact with wall throughout movement",
        ]

    def get_video_url(self) -> str | None:
        """Return demonstration video URL"""
        return "https://www.youtube.com/watch?v=ywYi4rBhRBQ"

    # ========== HELPER METHODS ==========

    def _midpoint(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Calculate 2D midpoint between two 3D points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _angle_from_vertical_2d(self, p1: Tuple, p2: Tuple) -> float:
        """
        Calculate angle from vertical using 2D coordinates.
        Returns signed angle.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        return angle

    def _calculate_arm_angle_from_horizontal(
        self, shoulder: Tuple, elbow: Tuple, wrist: Tuple
    ) -> float:
        """
        Calculate arm angle from horizontal plane.
        Uses the shoulder-wrist vector.

        Returns:
        - 0° = arm pointing straight down
        - 90° = arm horizontal (shoulder level)
        - 180° = arm pointing straight up
        """
        # Calculate vector from shoulder to elbow
        dx = abs(wrist[0] - shoulder[0])
        dy = wrist[1] - shoulder[1]

        # Calculate angle from vertical
        # atan2(dy, dx) gives angle from horizontal
        # We want angle where up is 180° and down is 0°
        angle = math.degrees(
            math.atan2(-dy, dx)
        )  # Negative dy because y increases downward

        # Convert to 0-180 scale where:
        # 0° = straight down, 90° = horizontal, 180° = straight up
        angle = angle + 90

        # Clamp to 0-180 range
        angle = max(0, min(180, angle))
        return angle

    def _calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """
        Calculate angle formed by three points (p2 is vertex).
        Uses law of cosines.
        """
        # Calculate distances
        a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        b = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
        c = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)

        # Avoid division by zero
        if a * b == 0:
            return 0

        # Law of cosines
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range

        return math.degrees(math.acos(cos_angle))

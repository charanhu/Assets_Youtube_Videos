# Personal Trainer Application with Automatic Exercise Recognition
# This script detects and counts reps for various exercises using MediaPipe and OpenCV.

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    Parameters:
        a, b, c: Each a list or array of two elements representing x and y coordinates.

    Returns:
        angle: The angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


class Exercise:
    def __init__(self, name, angle_points, up_threshold, down_threshold, direction='up_down'):
        """
        Initializes an Exercise instance.

        Parameters:
            name (str): Name of the exercise.
            angle_points (tuple): Tuple of three PoseLandmark enums for angle calculation.
            up_threshold (float): Angle threshold to detect the "up" position.
            down_threshold (float): Angle threshold to detect the "down" position.
            direction (str): The direction of movement ('up_down' or 'down_up').
        """
        self.name = name
        self.angle_points = angle_points
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.direction = direction
        self.counter = 0
        self.stage = None

    def update(self, landmarks):
        """
        Updates the exercise state based on current landmarks.

        Parameters:
            landmarks (list): List of landmarks detected by MediaPipe.

        Returns:
            angle (float): The calculated angle for the exercise.
        """
        # Extract the required landmarks
        try:
            a = [landmarks[self.angle_points[0]].x, landmarks[self.angle_points[0]].y]
            b = [landmarks[self.angle_points[1]].x, landmarks[self.angle_points[1]].y]
            c = [landmarks[self.angle_points[2]].x, landmarks[self.angle_points[2]].y]

            # Calculate angle
            angle = calculate_angle(a, b, c)

            # Update stage and counter based on direction
            if self.direction == 'up_down':
                if angle > self.up_threshold:
                    self.stage = "down"
                if angle < self.down_threshold and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                    print(f"{self.name} Reps: {self.counter}")
            elif self.direction == 'down_up':
                if angle < self.up_threshold:
                    self.stage = "up"
                if angle > self.down_threshold and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                    print(f"{self.name} Reps: {self.counter}")

            return angle
        except IndexError:
            return None


def detect_exercise(angles, prev_exercise, angle_changes, angle_change_threshold=10):
    """
    Detects the current exercise based on angles and their changes.

    Parameters:
        angles (dict): Dictionary containing current angles for each exercise.
        prev_exercise (str): Previously detected exercise.
        angle_changes (dict): Dictionary containing the changes in angles since the last frame.
        angle_change_threshold (float): Threshold for detecting significant changes.

    Returns:
        detected_exercise (str): The name of the detected exercise.
    """
    detected_exercise = None

    # Define exercise priority or rules
    exercise_priority = ["Bicep Curl", "Squat", "Push-Up"]  # Modify as needed

    for exercise in exercise_priority:
        change_key = exercise.lower().replace(" ", "_")
        if abs(angle_changes.get(change_key, 0)) > angle_change_threshold:
            detected_exercise = exercise
            break

    return detected_exercise


def main():
    # Configuration for exercises
    exercise_config = {
        "Bicep Curl": {
            "angle_points": (
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
            ),
            "up_threshold": 160,
            "down_threshold": 30,
            "direction": "up_down",
        },
        "Squat": {
            "angle_points": (
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ),
            "up_threshold": 160,
            "down_threshold": 70,
            "direction": "up_down",
        },
        "Push-Up": {
            "angle_points": (
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
            ),
            "up_threshold": 160,
            "down_threshold": 90,
            "direction": "down_up",
        },
    }

    # Initialize exercises
    exercises = {name: Exercise(name, **config) for name, config in exercise_config.items()}

    # Variables for exercise detection
    prev_angles = defaultdict(float)
    current_exercise = None

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate angles for each exercise
                angles = {}
                for name, exercise in exercises.items():
                    try:
                        a = [landmarks[exercise.angle_points[0]].x, landmarks[exercise.angle_points[0]].y]
                        b = [landmarks[exercise.angle_points[1]].x, landmarks[exercise.angle_points[1]].y]
                        c = [landmarks[exercise.angle_points[2]].x, landmarks[exercise.angle_points[2]].y]
                        angles_key = name.lower().replace(" ", "_")
                        angles[angles_key] = calculate_angle(a, b, c)
                    except IndexError:
                        angles[angles_key] = 0

                # Calculate angle changes
                angle_changes = {k: angles[k] - prev_angles[k] for k in angles}
                prev_angles = angles.copy()

                # Detect current exercise
                detected_exercise = detect_exercise(angles, current_exercise, angle_changes)

                if detected_exercise:
                    current_exercise = detected_exercise

                # Update the current exercise
                if current_exercise:
                    exercise = exercises[current_exercise]
                    angle = exercise.update(landmarks)

                    if angle is not None:
                        # Get coordinates for angle display
                        a = [landmarks[exercise.angle_points[0]].x, landmarks[exercise.angle_points[0]].y]
                        b = [landmarks[exercise.angle_points[1]].x, landmarks[exercise.angle_points[1]].y]
                        c = [landmarks[exercise.angle_points[2]].x, landmarks[exercise.angle_points[2]].y]

                        image_height, image_width, _ = image.shape
                        a_pixel = tuple(np.multiply(a, [image_width, image_height]).astype(int))
                        b_pixel = tuple(np.multiply(b, [image_width, image_height]).astype(int))
                        c_pixel = tuple(np.multiply(c, [image_width, image_height]).astype(int))

                        # Display angle
                        cv2.putText(
                            image,
                            str(int(angle)),
                            b_pixel,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        # Render rep counter and stage
                        cv2.rectangle(image, (0, 0), (300, 150), (245, 117, 16), -1)

                        # Rep data
                        cv2.putText(
                            image,
                            "REPS",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            str(exercise.counter),
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        # Stage data
                        cv2.putText(
                            image,
                            "STAGE",
                            (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        stage_text = exercise.stage if exercise.stage else ""
                        cv2.putText(
                            image,
                            stage_text,
                            (150, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        # Display current exercise
                        cv2.putText(
                            image,
                            f"Exercise: {current_exercise}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

            # Display the resulting image
            cv2.imshow("Personal Trainer", image)

            # Handle keypresses
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

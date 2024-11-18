# Personal Trainer Application with Automatic Exercise Recognition
# This script detects and counts reps for Bicep Curls, Squats, and Push-Ups using MediaPipe and OpenCV.

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Define a function to calculate angles
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    Parameters:
    a, b, c: Each a list of two elements representing x and y coordinates.

    Returns:
    angle: The angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # Calculate the angle in radians
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is within [0, 180]
    if angle > 180.0:
        angle = 360 - angle

    return angle


# Define an Exercise class to handle different exercises
class Exercise:
    def __init__(self, name, landmarks, angle_points, up_threshold, down_threshold):
        """
        Initializes an Exercise instance.

        Parameters:
        name: Name of the exercise.
        landmarks: Tuple containing the relevant landmarks for the exercise.
        angle_points: Tuple indicating which landmarks to use for angle calculation.
        up_threshold: Angle threshold to detect the "up" position.
        down_threshold: Angle threshold to detect the "down" position.
        """
        self.name = name
        self.landmarks = landmarks
        self.angle_points = angle_points
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.counter = 0
        self.stage = None

    def update(self, landmarks):
        """
        Updates the exercise state based on current landmarks.

        Parameters:
        landmarks: List of landmarks detected by MediaPipe.

        Returns:
        angle: The calculated angle for the exercise.
        """
        # Extract the required landmarks
        a = [landmarks[self.angle_points[0]].x, landmarks[self.angle_points[0]].y]
        b = [landmarks[self.angle_points[1]].x, landmarks[self.angle_points[1]].y]
        c = [landmarks[self.angle_points[2]].x, landmarks[self.angle_points[2]].y]

        # Calculate angle
        angle = calculate_angle(a, b, c)

        # Update stage and counter based on exercise type
        if self.name == "Bicep Curl":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                print(f"{self.name} Reps: {self.counter}")

        elif self.name == "Squat":
            if angle > self.up_threshold:
                self.stage = "up"
            if angle < self.down_threshold and self.stage == "up":
                self.stage = "down"
                self.counter += 1
                print(f"{self.name} Reps: {self.counter}")

        elif self.name == "Push-Up":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                print(f"{self.name} Reps: {self.counter}")

        return angle


# Function to detect current exercise based on angles
def detect_exercise(angles, prev_exercise, angle_changes):
    """
    Detects the current exercise based on angles and their changes.

    Parameters:
    angles: Dictionary containing current angles for each exercise.
    prev_exercise: Previously detected exercise.
    angle_changes: Dictionary containing the changes in angles since the last frame.

    Returns:
    detected_exercise: The name of the detected exercise.
    """
    detected_exercise = None

    # Define simple rules for exercise detection
    # These rules can be enhanced with more sophisticated logic or machine learning models

    # Example rules:
    # - Bicep Curl: Significant change in elbow angle with relatively stable hip and knee angles
    # - Squat: Significant change in knee and hip angles
    # - Push-Up: Significant change in elbow angle and torso position

    # Thresholds for detecting significant changes
    angle_change_threshold = 10  # degrees

    # Check for Bicep Curl
    if (
        abs(angle_changes["bicep_curl"]) > angle_change_threshold
        and abs(angle_changes["squat_knee"]) < angle_change_threshold
        and abs(angle_changes["pushup_elbow"]) < angle_change_threshold
    ):
        detected_exercise = "Bicep Curl"

    # Check for Squat
    elif (
        abs(angle_changes["squat_knee"]) > angle_change_threshold
        and abs(angle_changes["bicep_curl"]) < angle_change_threshold
        and abs(angle_changes["pushup_elbow"]) < angle_change_threshold
    ):
        detected_exercise = "Squat"

    # Check for Push-Up
    elif (
        abs(angle_changes["pushup_elbow"]) > angle_change_threshold
        and abs(angle_changes["bicep_curl"]) < angle_change_threshold
        and abs(angle_changes["squat_knee"]) < angle_change_threshold
    ):
        detected_exercise = "Push-Up"

    return detected_exercise


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        # Initialize exercises
        exercises = {
            "Bicep Curl": Exercise(
                name="Bicep Curl",
                landmarks=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                ),
                up_threshold=160,
                down_threshold=30,
            ),
            "Squat": Exercise(
                name="Squat",
                landmarks=("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value,
                ),
                up_threshold=160,
                down_threshold=70,
            ),
            "Push-Up": Exercise(
                name="Push-Up",
                landmarks=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                ),
                up_threshold=160,
                down_threshold=90,
            ),
        }

        # Variables for exercise detection
        prev_angles = {"bicep_curl": 0, "squat_knee": 0, "pushup_elbow": 0}
        current_exercise = None

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
            try:
                landmarks = results.pose_landmarks.landmark

                # Calculate angles for each exercise
                angles = {
                    "bicep_curl": calculate_angle(
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[0]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[1]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[2]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[2]].y,
                        ],
                    ),
                    "squat_knee": calculate_angle(
                        [
                            landmarks[exercises["Squat"].angle_points[0]].x,
                            landmarks[exercises["Squat"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Squat"].angle_points[1]].x,
                            landmarks[exercises["Squat"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Squat"].angle_points[2]].x,
                            landmarks[exercises["Squat"].angle_points[2]].y,
                        ],
                    ),
                    "pushup_elbow": calculate_angle(
                        [
                            landmarks[exercises["Push-Up"].angle_points[0]].x,
                            landmarks[exercises["Push-Up"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Push-Up"].angle_points[1]].x,
                            landmarks[exercises["Push-Up"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Push-Up"].angle_points[2]].x,
                            landmarks[exercises["Push-Up"].angle_points[2]].y,
                        ],
                    ),
                }

                # Calculate angle changes
                angle_changes = {
                    "bicep_curl": angles["bicep_curl"] - prev_angles["bicep_curl"],
                    "squat_knee": angles["squat_knee"] - prev_angles["squat_knee"],
                    "pushup_elbow": angles["pushup_elbow"]
                    - prev_angles["pushup_elbow"],
                }

                # Update previous angles
                prev_angles = angles.copy()

                # Detect current exercise
                detected_exercise = detect_exercise(
                    angles, current_exercise, angle_changes
                )

                if detected_exercise:
                    current_exercise = detected_exercise

                # Update the current exercise
                if current_exercise:
                    angle = exercises[current_exercise].update(landmarks)

                    # Get coordinates for angle display
                    if current_exercise == "Bicep Curl":
                        a = [
                            landmarks[exercises[current_exercise].angle_points[0]].x,
                            landmarks[exercises[current_exercise].angle_points[0]].y,
                        ]
                        b = [
                            landmarks[exercises[current_exercise].angle_points[1]].x,
                            landmarks[exercises[current_exercise].angle_points[1]].y,
                        ]
                        c = [
                            landmarks[exercises[current_exercise].angle_points[2]].x,
                            landmarks[exercises[current_exercise].angle_points[2]].y,
                        ]
                    elif current_exercise == "Squat":
                        a = [
                            landmarks[exercises[current_exercise].angle_points[0]].x,
                            landmarks[exercises[current_exercise].angle_points[0]].y,
                        ]
                        b = [
                            landmarks[exercises[current_exercise].angle_points[1]].x,
                            landmarks[exercises[current_exercise].angle_points[1]].y,
                        ]
                        c = [
                            landmarks[exercises[current_exercise].angle_points[2]].x,
                            landmarks[exercises[current_exercise].angle_points[2]].y,
                        ]
                    elif current_exercise == "Push-Up":
                        a = [
                            landmarks[exercises[current_exercise].angle_points[0]].x,
                            landmarks[exercises[current_exercise].angle_points[0]].y,
                        ]
                        b = [
                            landmarks[exercises[current_exercise].angle_points[1]].x,
                            landmarks[exercises[current_exercise].angle_points[1]].y,
                        ]
                        c = [
                            landmarks[exercises[current_exercise].angle_points[2]].x,
                            landmarks[exercises[current_exercise].angle_points[2]].y,
                        ]

                    # Calculate coordinates in pixels
                    image_height, image_width, _ = image.shape
                    a_pixel = tuple(
                        np.multiply(a, [image_width, image_height]).astype(int)
                    )
                    b_pixel = tuple(
                        np.multiply(b, [image_width, image_height]).astype(int)
                    )
                    c_pixel = tuple(
                        np.multiply(c, [image_width, image_height]).astype(int)
                    )

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
                    # Setup status box
                    cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)

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
                        str(exercises[current_exercise].counter),
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
                    cv2.putText(
                        image,
                        (
                            exercises[current_exercise].stage
                            if exercises[current_exercise].stage
                            else ""
                        ),
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

            except AttributeError:
                pass

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )

            # Display the resulting image
            cv2.imshow("Personal Trainer", image)

            # Handle keypresses
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                # Quit the application
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

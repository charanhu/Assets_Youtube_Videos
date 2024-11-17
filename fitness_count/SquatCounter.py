# Simple Squat Repetition Counter using MediaPipe and OpenCV

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


class SquatCounter:
    def __init__(self, up_threshold=160, down_threshold=70):
        """
        Initializes the SquatCounter.

        Parameters:
            up_threshold (float): Angle threshold to detect the "up" position.
            down_threshold (float): Angle threshold to detect the "down" position.
        """
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.counter = 0
        self.stage = None

    def update(self, landmarks):
        """
        Updates the squat counter based on current landmarks.

        Parameters:
            landmarks (list): List of landmarks detected by MediaPipe.

        Returns:
            angle (float): The calculated knee angle.
        """
        # Get coordinates for left knee
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]

        # Calculate angle
        angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Squat logic
        if angle > self.up_threshold:
            self.stage = "up"
        if angle < self.down_threshold and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            print(f"Squats: {self.counter}")

        return angle


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        # Initialize SquatCounter
        squat_counter = SquatCounter()

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

            try:
                landmarks = results.pose_landmarks.landmark

                # Update squat counter
                angle = squat_counter.update(landmarks)

                # Get coordinates for angle display
                left_hip = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                ]
                left_knee = [
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                ]
                left_ankle = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                ]

                image_height, image_width, _ = image.shape
                hip_pixel = tuple(
                    np.multiply(left_hip, [image_width, image_height]).astype(int)
                )
                knee_pixel = tuple(
                    np.multiply(left_knee, [image_width, image_height]).astype(int)
                )
                ankle_pixel = tuple(
                    np.multiply(left_ankle, [image_width, image_height]).astype(int)
                )

                # Display angle
                cv2.putText(
                    image,
                    str(int(angle)),
                    knee_pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Render rep counter and stage
                # Setup status box
                cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)

                # Rep data
                cv2.putText(
                    image,
                    "REPS",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(squat_counter.counter),
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
                    (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    squat_counter.stage if squat_counter.stage else "",
                    (120, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
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
            cv2.imshow("Squat Counter", image)

            # Handle keypresses
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

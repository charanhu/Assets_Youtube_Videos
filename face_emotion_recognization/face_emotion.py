# Face Emotion Detection Application
# This script captures video from the webcam, detects faces, analyzes emotions,
# and displays the results in real-time using OpenCV and FER libraries.

# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
from fer import FER  # FER library for facial emotion recognition
import matplotlib.pyplot as plt  # Matplotlib for plotting emotion probabilities (optional)


def main():
    """
    The main function that initializes the webcam, processes video frames to detect emotions,
    and displays the annotated video feed.
    """

    # Initialize video capture from the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return  # Exit the function if webcam is not accessible

    # Initialize the FER detector with MTCNN face detector for better accuracy
    detector = FER(mtcnn=True)

    print("Face Emotion Detection started. Press 'q' to exit.")

    # Start an infinite loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        # If frame is read correctly, ret is True. Otherwise, ret is False.
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break  # Exit the loop if no frame is captured

        # Convert the captured frame from BGR color space (used by OpenCV) to RGB color space
        # FER library expects images in RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use FER to detect emotions in the frame
        # detect_emotions returns a list of dictionaries, each containing:
        # - 'box': Bounding box coordinates of the detected face
        # - 'emotions': Dictionary of emotions with their corresponding probabilities
        emotions = detector.detect_emotions(rgb_frame)

        # Iterate over each detected face and its emotions
        for face in emotions:
            # Extract the bounding box coordinates (x, y, width, height) of the face
            (x, y, w, h) = face["box"]

            # Extract the emotions dictionary which contains emotions and their scores
            face_emotions = face["emotions"]

            # Determine the emotion with the highest probability
            # max() is used with a key function that retrieves the second item of each emotion tuple
            dominant_emotion, emotion_score = max(
                face_emotions.items(), key=lambda item: item[1]
            )

            # Draw a rectangle around the detected face in the original frame (BGR color space)
            # Parameters:
            # - frame: The image on which to draw
            # - (x, y): Top-left corner of the rectangle
            # - (x + w, y + h): Bottom-right corner of the rectangle
            # - (0, 255, 0): Color of the rectangle in BGR (green)
            # - 2: Thickness of the rectangle border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare the label text with the dominant emotion and its confidence score
            # The score is converted to a percentage with one decimal place
            label = f"{dominant_emotion} ({emotion_score*100:.1f}%)"

            # Determine the position to place the label
            # If there's enough space above the face, place the label above; otherwise, below
            label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 20)

            # Put the label text on the frame
            # Parameters:
            # - frame: The image on which to write
            # - label: The text to write
            # - label_position: Bottom-left corner of the text string in the image
            # - cv2.FONT_HERSHEY_SIMPLEX: Font type
            # - 0.9: Font scale (size)
            # - (255, 0, 0): Text color in BGR (blue)
            # - 2: Thickness of the text
            # - cv2.LINE_AA: Line type for better quality
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Display the resulting frame with annotations in a window named 'Face Emotion Detection'
        cv2.imshow("Face Emotion Detection", frame)

        # Optional: Display emotion probabilities using Matplotlib
        # To enable, uncomment the following block of code
        """
        for face in emotions:
            face_emotions = face["emotions"]  # Get emotions for the face
            # Sort emotions by their scores in descending order
            emotions_sorted = sorted(face_emotions.items(), key=lambda item: item[1], reverse=True)
            emotions_names = [emotion for emotion, score in emotions_sorted]  # List of emotion names
            emotions_scores = [score for emotion, score in emotions_sorted]  # Corresponding scores
            
            # Create a new figure for the bar chart
            plt.figure(figsize=(6,4))
            # Create a bar chart with emotion names and their scores
            bars = plt.bar(emotions_names, emotions_scores, color='skyblue')
            plt.ylim(0,1)  # Set y-axis limits from 0 to 1
            plt.title('Emotion Probabilities')  # Title of the chart
            plt.xlabel('Emotions')  # X-axis label
            plt.ylabel('Probability')  # Y-axis label
            
            # Annotate each bar with its score
            for bar, score in zip(bars, emotions_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.2f}", ha='center')
            
            # Display the bar chart
            plt.show()
        """

        # Wait for 1 millisecond to check if 'q' key is pressed to quit
        # & 0xFF is used to get the last 8 bits of the pressed key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting Face Emotion Detection.")
            break  # Exit the loop if 'q' is pressed

    # Release the webcam resource
    cap.release()
    # Destroy all OpenCV windows to free up resources
    cv2.destroyAllWindows()


# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function to start the application

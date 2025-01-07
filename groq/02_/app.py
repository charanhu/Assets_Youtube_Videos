import threading
import pyaudio
import wave
import time
import cv2
import base64
import pyttsx3
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

###############################################################################
# Global Variables
###############################################################################
last_frame = None               # Will store the latest camera frame
frame_lock = threading.Lock()   # Lock for accessing last_frame safely

stop_event = threading.Event()  # To stop the camera thread

###############################################################################
# Voice Activity Detection Settings
###############################################################################
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_SILENCE_BLOCKS = 2 * (RATE // CHUNK)  # ~2 seconds of silence
THRESHOLD = 800  # Adjust for your environment

###############################################################################
# 1) Camera Thread: continuously capture frames
###############################################################################
def camera_thread():
    """
    Continuously captures frames from the default camera and saves
    the latest valid frame to a global variable 'last_frame'.
    """
    global last_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CameraThread] Could not open camera.")
        return

    print("[CameraThread] Camera opened. Starting capture loop...")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[CameraThread] Failed to read frame. Stopping.")
            break

        # Store the latest frame in a thread-safe manner
        with frame_lock:
            last_frame = frame

        # Sleep a bit so we don't max out CPU
        time.sleep(0.01)

    cap.release()
    print("[CameraThread] Camera released. Exiting thread...")

###############################################################################
# 2) Text-to-Speech
###############################################################################
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

###############################################################################
# 3) Voice-Activated Recording (capture image after 3s)
###############################################################################
def listen_for_speech(output_filename="temp.wav"):
    """
    Continuously listens on the microphone for speech. Once amplitude
    crosses THRESHOLD, we start recording. After 3 seconds from
    the first detection of speech, we grab the latest camera frame.

    We keep recording until the user is silent for ~2 seconds.

    Returns: (audio_path, frame_at_3s)
      - audio_path: The path to the saved WAV, or None if no speech.
      - frame_at_3s: The camera frame captured ~3s after speech start, or None.
    """
    import time

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[Audio] Listening for speech... (Ctrl+C to exit)")

    frames = []
    is_recording = False
    silent_blocks = 0

    record_start_time = 0.0
    image_captured = False
    frame_at_3s = None

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        amplitude = max(
            abs(int.from_bytes(data[i:i+2], 'little', signed=True))
            for i in range(0, len(data), 2)
        )

        if amplitude > THRESHOLD:
            # If we just detected speech
            if not is_recording:
                print("[Audio] Detected speech, starting recording...")
                is_recording = True
                record_start_time = time.time()

            frames.append(data)
            silent_blocks = 0

            # Check if 3 seconds have passed since first detection
            if not image_captured and (time.time() - record_start_time >= 5.0):
                # Capture the last_frame from the camera thread
                with frame_lock:
                    if last_frame is not None:
                        frame_at_3s = last_frame.copy()
                        print("[Audio] Captured frame at ~3s of speech.")
                image_captured = True

        else:
            # If we're already recording, track silence
            if is_recording:
                frames.append(data)
                silent_blocks += 1
                # If silent for ~2 seconds, stop
                if silent_blocks >= MAX_SILENCE_BLOCKS:
                    print("[Audio] Silence detected. Stopping recording.")
                    break
            # If not recording, do nothing and keep listening

    # Close audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    if not frames:
        print("[Audio] No speech captured.")
        return None, None

    # Save to WAV
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return output_filename, frame_at_3s

###############################################################################
# 4) Audio Transcription
###############################################################################
def transcribe_audio(client, audio_path="temp.wav"):
    """
    Send the recorded WAV file to Groq’s Whisper model for transcription.
    Returns the transcribed text or empty string on error.
    """
    print("[Audio] Transcribing audio with Whisper...")
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        text = transcription.text.strip()
        print(f"[Audio] Transcription Result: {text}")
        return text
    except Exception as e:
        print("[Audio] Error sending audio to Groq:", e)
        return ""

###############################################################################
# 5) Image Encoding
###############################################################################
def encode_frame_to_base64(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("[Vision] Failed to encode frame to JPEG.")
        return None
    return base64.b64encode(buffer).decode("utf-8")

###############################################################################
# 6) Main Logic with Conversation History
###############################################################################
def main():
    client = Groq()

    # Start the camera thread
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    # We'll keep up to 2 previous (Q, A) pairs
    conversation_history = []

    print("[Main] Voice-activated loop starting. Press Ctrl+C to exit. Press 'q' to exit camera window.")
    try:
        while True:
            # ------------------ (A) Wait for user speech & capture image after 3s ------------------
            audio_path, frame_at_3s = listen_for_speech(output_filename="temp.wav")
            if audio_path is None:
                # Means no speech captured, keep listening
                continue

            # ------------------ (B) Transcribe audio ------------------
            transcription_text = transcribe_audio(client, audio_path=audio_path)
            if not transcription_text:
                transcription_text = "[No transcription captured]"

            # ------------------ (C) Build conversation history string ------------------
            while len(conversation_history) > 2:
                conversation_history.pop(0)

            if conversation_history:
                history_text = ""
                for idx, (prev_q, prev_a) in enumerate(conversation_history, start=1):
                    history_text += f"\n[Q{idx}]: {prev_q}\n[A{idx}]: {prev_a}\n"
            else:
                history_text = "[No prior conversation]"

            prompt_template = """
You are a personal helpful assistant for Charan. You must address him as "Sir."

Sir may ask you doubts. It may be related to vision (e.g., providing an image) and asking questions.

Keep your responses polite, helpful, informative, and short.

Here is the earlier conversation(s):
{conversation_history}

Here is Sir's new query: {transcription_text}
"""
            prompt = prompt_template.format(
                conversation_history=history_text,
                transcription_text=transcription_text
            )

            # ------------------ (D) Determine which frame to use ------------------
            # If the user spoke for less than 3s, 'frame_at_3s' will be None.
            # Optionally, you could fallback to the "last_frame" if you want:
            if frame_at_3s is None:
                print("[Vision] The user spoke less than 3s, so no 3s capture available.")
                speak_text("You spoke quickly. I don't have an image at three seconds.")
                # If you want to STILL capture, you can do:
                with frame_lock:
                    if last_frame is not None:
                        frame_at_3s = last_frame.copy()

            if frame_at_3s is None:
                # We have no valid image to send; skip or handle differently
                continue

            # Show the frame in a window (optional)
            cv2.imshow("Captured Frame (after ~3s of speech)", frame_at_3s)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] 'q' pressed in camera window.")
                break

            # ------------------ (E) Encode & Send to Vision Model ------------------
            base64_image = encode_frame_to_base64(frame_at_3s)
            if not base64_image:
                speak_text("Image encoding failed. I'll try again later.")
                continue

            try:
                print("[Vision] Sending text + image to Groq Vision...")
                vision_response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="llama-3.2-11b-vision-preview",
                )
                response_text = vision_response.choices[0].message.content.strip()
                print("[Vision] Model response:\n", response_text)

                # ------------------ (F) Speak the system's reply ------------------
                speak_text(response_text)

                # ------------------ (G) Update conversation history ------------------
                conversation_history.append((transcription_text, response_text))

            except Exception as e:
                print("[Vision] Error sending request to Groq Vision:", e)
                speak_text("I encountered an error reading the image or text. Please try again.")

    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C. Exiting...")

    # Stop camera thread
    stop_event.set()
    cam_thread.join()

    # Clean up OpenCV window
    cv2.destroyAllWindows()
    print("[Main] Exited cleanly.")

if __name__ == "__main__":
    main()

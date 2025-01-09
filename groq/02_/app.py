import os
import threading
import time
import cv2
import base64
import pyttsx3
import speech_recognition as sr

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv, find_dotenv

##############################################################################
# 1) Watsonx code from your reference
##############################################################################
def load_env():
    _ = load_dotenv(find_dotenv())

def llama32(messages, model_size=11):
    """
    Generate a chat completion using IBM Watsonx Llama 3.2 Vision Instruct.
    """
    load_env()  # Load environment variables from .env
    model_id = f"meta-llama/llama-3-2-{model_size}b-vision-instruct"

    api_key = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not all([api_key, project_id, model_id]):
        raise Exception(
            "Missing required environment variables or model_id mismatch."
        )

    credentials = Credentials(api_key=api_key, url=url)
    model_inference = ModelInference(
        model_id=model_id, credentials=credentials, project_id=project_id
    )

    response = model_inference.chat(messages=messages)
    return response["choices"][0]["message"]["content"]

##############################################################################
# 2) Camera Globals
##############################################################################
last_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

##############################################################################
# 3) Camera Thread
##############################################################################
def camera_thread():
    """
    Continuously capture frames from the default camera and store
    the latest valid frame in 'last_frame'.
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
            print("[CameraThread] Failed to read frame. Exiting loop.")
            break
        with frame_lock:
            last_frame = frame
        time.sleep(0.01)

    cap.release()
    print("[CameraThread] Camera released. Exiting thread...")

##############################################################################
# 4) Text-to-Speech
##############################################################################
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

##############################################################################
# 5) AudioRecorderThread: captures audio in background
##############################################################################
class AudioRecorderThread(threading.Thread):
    def __init__(self, device_index=None, language="en-US"):
        super().__init__()
        self.device_index = device_index
        self.language = language
        self.r = sr.Recognizer()
        self.audio_data = None
        self.exception = None

    def run(self):
        """
        This method runs in a separate thread.
        We'll open the mic, calibrate for 1s, then listen until the user stops.
        """
        try:
            with sr.Microphone(device_index=self.device_index) as source:
                print("[Audio] Calibrating for ambient noise (1s)...")
                self.r.adjust_for_ambient_noise(source, duration=1)
                self.r.pause_threshold = 2.0  # stops after 2s of silence
                print("[Audio] Now listening...")
                self.audio_data = self.r.listen(source)
        except Exception as e:
            self.exception = e
            print(f"[Audio] Error accessing microphone: {e}")

    def transcribe(self):
        """
        Use Google's free STT to transcribe the recorded audio_data.
        Returns recognized text or None if not recognized.
        """
        if self.audio_data is None:
            return None
        try:
            text = self.r.recognize_google(self.audio_data, language=self.language)
            return text.strip()
        except sr.UnknownValueError:
            print("[Audio] Could not understand the audio.")
        except sr.RequestError as e:
            print(f"[Audio] RequestError: {e}")
        return None

##############################################################################
# 6) Encode Frame to Base64
##############################################################################
def encode_frame_to_base64(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")

##############################################################################
# 7) Send to Watsonx (Vision + LLM)
##############################################################################
def send_to_vision_model(prompt_text, base64_image, model_size=11):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]
    return llama32(messages, model_size=model_size)

##############################################################################
# 8) Main Logic
##############################################################################
def main():
    # Start the camera thread
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    conversation_history = []
    print("[Main] Press Ctrl+C to exit. Press 'q' in camera window to exit camera loop.")

    try:
        while True:
            print("\n=== Speak now (the system will only capture image if you actually speak) ===")
            # ----------------------------------------------------------------
            # (A) Start the audio recording in a background thread
            # ----------------------------------------------------------------
            audio_thread = AudioRecorderThread(device_index=None, language="en-US")
            audio_thread.start()
            audio_thread.join()

            if audio_thread.exception:
                print("[Main] Error in audio thread:", audio_thread.exception)
                continue

            # ----------------------------------------------------------------
            # (B) Transcribe the audio
            # ----------------------------------------------------------------
            recognized_text = audio_thread.transcribe()
            if not recognized_text:
                # Means no valid speech recognized => skip capturing
                print("[Main] No words recognized (could be noise/silence). Skipping image capture.")
                continue

            print(f"[Main] Transcribed text: '{recognized_text}'")

            # ----------------------------------------------------------------
            # (C) Sleep 2s, then capture the image
            #     (We do this only because the user actually spoke words)
            # ----------------------------------------------------------------
            time.sleep(2)
            with frame_lock:
                local_frame = None if (last_frame is None) else last_frame.copy()

            if local_frame is None:
                print("[Main] No valid camera frame at 2s. Skipping this round...")
                continue

            # Show the frame if desired
            cv2.imshow("Frame Captured at ~2s", local_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] 'q' pressed. Exiting camera loop.")
                stop_event.set()
                break

            base64_image = encode_frame_to_base64(local_frame)
            if not base64_image:
                print("[Main] Failed to encode image. Skipping request.")
                continue

            # ----------------------------------------------------------------
            # (D) Build conversation prompt (up to 2 prior Q/A pairs)
            # ----------------------------------------------------------------
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
                transcription_text=recognized_text
            )

            # ----------------------------------------------------------------
            # (E) Send to Watsonx
            # ----------------------------------------------------------------
            try:
                print("[Main] Sending text + image to Watsonx Llama 3.2 Vision...")
                response_text = send_to_vision_model(prompt, base64_image, model_size=11)
                response_text = response_text.strip()
                print("[Main] Watsonx Response:\n", response_text)

                # Speak the response
                speak_text(response_text)

                # Update conversation history
                conversation_history.append((recognized_text, response_text))

            except Exception as e:
                print("[Main] Error calling Watsonx vision:", e)
                speak_text("I encountered an error reading the image or text. Please try again.")

    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C. Exiting...")

    # Stop the camera
    stop_event.set()
    cam_thread.join()
    cv2.destroyAllWindows()
    print("[Main] Exited cleanly.")

if __name__ == "__main__":
    main()

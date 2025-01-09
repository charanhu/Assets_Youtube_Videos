import speech_recognition as sr
from enum import Enum

class Language(Enum):
    ENGLISH = "en-US"
    CHINESE = "zh-CN"
    FRENCH = "fr-FR"
    SPANISH_SPAIN = "es-ES"
    SPANISH_LATAM = "es-US"
    KOREAN = "ko-KR"
    JAPANESE = "ja-JP"

class SpeechToText:
    @staticmethod
    def print_mic_device_index():
        """
        Print the microphone device index for reference.
        """
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{name} (device index: {index})")

    @staticmethod
    def speech_to_text(device_index=None, language=Language.ENGLISH):
        """
        Record audio from the microphone and return recognized text.
        """
        r = sr.Recognizer()
        try:
            with sr.Microphone(device_index=device_index) as source:
                print("[Audio] Calibrating for ambient noise. Please remain silent...")
                r.adjust_for_ambient_noise(source)
                # Extend pause threshold to wait for 2 seconds of silence before stopping
                r.pause_threshold = 2.0
                print("[Audio] Listening...")
                audio = r.listen(source)
        except Exception as e:
            print(f"[Audio] Error accessing microphone: {e}")
            return None

        try:
            print("[Audio] Recognizing...")
            text = r.recognize_google(audio, language=language.value)
            print(f"[Audio] Recognized text: '{text}'")
            return text
        except sr.UnknownValueError:
            print("[Audio] Could not understand audio")
        except sr.RequestError as e:
            print(f"[Audio] Could not request results; {e}")
        return None

def check_mic_device_index():
    SpeechToText.print_mic_device_index()

def run_speech_to_text_english(device_index=None):
    SpeechToText.speech_to_text(device_index, Language.ENGLISH)

if __name__ == "__main__":
    # List available microphone devices
    check_mic_device_index()
    
    # Use the correct microphone device index (0 for MacBook Pro Microphone)
    run_speech_to_text_english(device_index=0)

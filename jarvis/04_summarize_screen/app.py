import os
import re
import time
import base64
import webbrowser
import subprocess
import pyautogui
import cv2
import pytesseract
import pyttsx3
import speech_recognition as sr
import numpy as np
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from duckduckgo_search import DDGS

##############################################################################
# 1) Environment & LLM Setup
##############################################################################
def load_env():
    _ = load_dotenv(find_dotenv())

def get_model_inference(model_id: str):
    load_env()
    api_key = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not all([api_key, project_id, model_id]):
        raise Exception("Missing environment variables or invalid model_id.")
    creds = Credentials(api_key=api_key, url=url)
    return ModelInference(model_id=model_id, credentials=creds, project_id=project_id)

##############################################################################
# 2) LLM Calls
##############################################################################
def classify_intent(user_text: str) -> str:
    """
    Uses meta-llama/llama-3-3-70b-instruct to classify user_text
    into one of four labels: SCREENSHOT, CAMERA, OTHER, APP_CONTROL.
    """
    classification_model_id = "meta-llama/llama-3-3-70b-instruct"
    model_inference = get_model_inference(classification_model_id)

    system_prompt = """
You are a text classifier. Classify the user's request as one of these:
 - SCREENSHOT (capture or summarize screen)
 - CAMERA (capture from webcam)
 - OTHER (general question => search + Q&A)
 - APP_CONTROL (open/close apps, open URL, search in site, scroll, or bounding box interactions)
Return exactly one label: SCREENSHOT, CAMERA, OTHER, or APP_CONTROL.
"""
    user_prompt = f"User said: {user_text}\nWhich label fits best?"

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    resp = model_inference.chat(messages=messages)
    raw_output = resp["choices"][0]["message"]["content"].strip().upper()

    if "SCREENSHOT" in raw_output:
        return "SCREENSHOT"
    elif "CAMERA" in raw_output:
        return "CAMERA"
    elif "APP_CONTROL" in raw_output:
        return "APP_CONTROL"
    else:
        return "OTHER"

def llama_vision_inference(messages, model_size=11):
    vision_model_id = f"meta-llama/llama-3-2-{model_size}b-vision-instruct"
    model_inference = get_model_inference(vision_model_id)
    resp = model_inference.chat(messages=messages)
    return resp["choices"][0]["message"]["content"].strip()

def llama_text_inference(system_prompt: str, user_prompt: str) -> str:
    text_model_id = "meta-llama/llama-3-3-70b-instruct"
    model_inference = get_model_inference(text_model_id)
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    resp = model_inference.chat(messages=messages)
    return resp["choices"][0]["message"]["content"].strip()

##############################################################################
# 3) TTS
##############################################################################
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

##############################################################################
# 4) Speech Recognition
##############################################################################
def recognize_speech_from_mic(device_index=None, language="en-US"):
    r = sr.Recognizer()
    try:
        with sr.Microphone(device_index=device_index) as source:
            print("[Audio] Calibrating for 1s of ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            r.pause_threshold = 2.0
            print("[Audio] Listening...")
            audio = r.listen(source)
    except Exception as e:
        print("[Audio] Mic error:", e)
        return None

    try:
        print("[Audio] Recognizing speech (Google)...")
        text = r.recognize_google(audio, language=language)
        print("[Audio] You said:", text)
        return text.strip()
    except sr.UnknownValueError:
        print("[Audio] Could not understand audio.")
    except sr.RequestError as e:
        print("[Audio] Request error:", e)
    return None

##############################################################################
# 5) Screenshot => Base64
##############################################################################
def capture_screenshot_as_base64():
    screenshot = pyautogui.screenshot()
    if screenshot.mode in ("RGBA", "LA"):
        screenshot = screenshot.convert("RGB")
    import io
    buffer = io.BytesIO()
    screenshot.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

##############################################################################
# 6) Camera => Base64
##############################################################################
def capture_camera_frame_as_base64():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] Could not open camera.")
        return None

    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("[Camera] No frame captured.")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_image = Image.fromarray(frame_rgb)
    if pil_image.mode in ("RGBA", "LA"):
        pil_image = pil_image.convert("RGB")

    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

##############################################################################
# 7) Vision Prompt
##############################################################################
def build_vision_prompt(conversation_history, user_question):
    while len(conversation_history) > 2:
        conversation_history.pop(0)

    if conversation_history:
        history_text = ""
        for idx, (q, a) in enumerate(conversation_history, start=1):
            history_text += f"\n[Q{idx}]: {q}\n[A{idx}]: {a}\n"
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
    return prompt_template.format(
        conversation_history=history_text,
        transcription_text=user_question
    )

##############################################################################
# 8) Searching + Q&A
##############################################################################
def duckduckgo_search_query(query, max_results=5):
    from duckduckgo_search import DDGS
    results_list = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results_list.append(r)
    return results_list

def build_text_prompt_with_search(conversation_history, user_question, search_results):
    while len(conversation_history) > 2:
        conversation_history.pop(0)

    if conversation_history:
        history_text = ""
        for idx, (q, a) in enumerate(conversation_history, start=1):
            history_text += f"\n[Q{idx}]: {q}\n[A{idx}]: {a}\n"
    else:
        history_text = "[No prior conversation]"

    context_text = "\nRelevant search results:\n"
    for i, result in enumerate(search_results, start=1):
        context_text += f"\n[{i}] Title: {result['title']}\n    URL: {result['href']}\n    Snippet: {result['body']}\n"

    system_prompt = f"""
You are a personal helpful assistant for Charan. You must address him as "Sir."

You have access to these search results for up-to-date info:
{context_text}

Here is the earlier conversation(s):
{history_text}

Use the search info to answer accurately. Keep it polite, helpful, short.
"""
    user_prompt = f"Sir's new query: {user_question}"
    return system_prompt, user_prompt

##############################################################################
# 9) BOUNDING BOX DETECTION (OCR) + CLICK
##############################################################################
bounding_boxes = []  # global or shared structure: list of (id, text, x, y, w, h)

def detect_bounding_boxes_on_screen():
    """
    1. Take a screenshot.
    2. Run OCR via pytesseract to get bounding boxes for each recognized text chunk.
    3. Store them in bounding_boxes as (id, text, x, y, w, h).
    4. Show them in a temporary OpenCV window so the user can see them visually.
    """
    global bounding_boxes
    bounding_boxes.clear()

    screenshot = pyautogui.screenshot()
    open_cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Use tesseract with bounding box data
    data = pytesseract.image_to_data(open_cv_image, output_type=pytesseract.Output.DICT)
    # data keys: level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text

    # build boxes
    n_boxes = len(data['text'])
    box_id = 1

    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text or text == "":
            continue
        # filter out very low-confidence detections
        conf = int(data['conf'][i])
        if conf < 40:
            continue

        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bounding_boxes.append((box_id, text, x, y, w, h))
        box_id += 1

    # Now draw them in an OpenCV window
    # We'll create a copy
    overlay = open_cv_image.copy()
    for (bid, t, x, y, w, h) in bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, str(bid), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show it. We'll open a window named "BoundingBoxes"
    cv2.imshow("BoundingBoxes", overlay)
    cv2.waitKey(1000)  # show for 1 second or so
    # you might prefer an interactive approach: waitKey(0) until user presses key
    cv2.destroyWindow("BoundingBoxes")

def click_bounding_box(box_id):
    """
    1. Find bounding box with that ID.
    2. pyautogui.click() at the center of the box.
    """
    global bounding_boxes
    for (bid, t, x, y, w, h) in bounding_boxes:
        if bid == box_id:
            center_x = x + w // 2
            center_y = y + h // 2
            speak_text(f"Clicking box {bid}, Sir.")
            pyautogui.click(center_x, center_y)
            return
    speak_text("I couldn't find that box, Sir.")

##############################################################################
# 10) APP CONTROL (open/close, scroll, bounding boxes, etc.)
##############################################################################
def handle_app_control(user_text: str):
    """
    Includes logic for:
      - "open youtube.com and search for Elon Musk"
      - "search elon musk in youtube"
      - "open <app>"
      - "close <app>"
      - "scroll up/down"
      - "show bounding boxes" => detect_bounding_boxes_on_screen()
      - "click item #3" => click_bounding_box(3)
    """
    user_lower = user_text.lower().strip()

    # 0) SCROLLING
    if "scroll down" in user_lower:
        speak_text("Scrolling down, Sir.")
        # For example press down arrow 5 times
        for _ in range(5):
            pyautogui.press('down')
        return
    if "scroll up" in user_lower:
        speak_text("Scrolling up, Sir.")
        for _ in range(5):
            pyautogui.press('up')
        return

    # 1) Show bounding boxes
    if "show bounding boxes" in user_lower or "show boxes" in user_lower:
        speak_text("Detecting bounding boxes on screen, Sir.")
        detect_bounding_boxes_on_screen()
        speak_text(f"I found {len(bounding_boxes)} boxes. Say 'click item #n' to click.")
        return

    # 2) "click item #n"
    #    e.g. "click item #3", "click box 2"
    box_click_pattern = r"click.*(?:box|item)\s*#?(\d+)"
    match = re.search(box_click_pattern, user_lower)
    if match:
        box_id_str = match.group(1)
        if box_id_str.isdigit():
            box_id = int(box_id_str)
            click_bounding_box(box_id)
        else:
            speak_text("I couldn't parse that box ID, Sir.")
        return

    # 3) "open <domain> and search for <term>"
    pattern = r"^open\s+(\S+)\s+and\s+search\s+for\s+(.+)$"
    match = re.match(pattern, user_lower)
    if match:
        domain = match.group(1)
        search_term = match.group(2)
        encoded_term = search_term.replace(" ", "+")
        if "youtube" in domain:
            final_url = f"https://www.youtube.com/results?search_query={encoded_term}"
            speak_text(f"Searching for {search_term} on YouTube, Sir.")
            webbrowser.open(final_url)
            return
        elif "google" in domain:
            final_url = f"https://www.google.com/search?q={encoded_term}"
            speak_text(f"Searching for {search_term} on Google, Sir.")
            webbrowser.open(final_url)
            return
        else:
            # Fallback
            if not domain.startswith(("http://", "https://")):
                domain = "https://" + domain
            speak_text(f"Opening {domain}, but I'm not sure how to pass your search query, Sir.")
            webbrowser.open(domain)
            return

    # 4) "search <term> in <domain>"
    if "search" in user_lower and " in " in user_lower:
        try:
            after_search = user_lower.split("search", 1)[1].strip()
            parts = after_search.split(" in ")
            search_term = parts[0].strip()
            domain = parts[1].strip()
            encoded_term = search_term.replace(" ", "+")
            if "youtube" in domain:
                final_url = f"https://www.youtube.com/results?search_query={encoded_term}"
                speak_text(f"Searching {search_term} on YouTube, Sir.")
                webbrowser.open(final_url)
                return
            elif "google" in domain:
                final_url = f"https://www.google.com/search?q={encoded_term}"
                speak_text(f"Searching {search_term} on Google, Sir.")
                webbrowser.open(final_url)
                return
            else:
                speak_text(f"I don't know how to search on {domain}, Sir. Opening site directly.")
                if not domain.startswith(("http://", "https://")):
                    domain = "https://" + domain
                webbrowser.open(domain)
        except Exception as e:
            speak_text("I couldn't parse your search request, Sir.")
            print("Search parse error:", e)
        return

    # 5) "open <something>"
    if user_lower.startswith("open "):
        to_open = user_lower.replace("open ", "").strip()
        if "." in to_open or "/" in to_open:
            if not to_open.startswith(("http://", "https://")):
                to_open = "https://" + to_open
            speak_text(f"Opening {to_open}, Sir.")
            webbrowser.open(to_open)
            return
        else:
            speak_text(f"Opening {to_open}, Sir.")
            open_app(to_open)
            return

    # 6) "close <something>"
    if user_lower.startswith("close "):
        to_close = user_lower.replace("close ", "").strip()
        speak_text(f"Closing {to_close}, Sir.")
        close_app(to_close)
        return

    # 7) If none matched
    speak_text("I am not sure how to handle that request, Sir.")

def open_app(app_name: str):
    # naive example
    print("[Open App]", app_name)
    if os.name == "posix":
        subprocess.Popen(["open", "-a", app_name])
    else:
        subprocess.Popen(["start", app_name], shell=True)

def close_app(app_name: str):
    print("[Close App]", app_name)
    if os.name == "posix":
        subprocess.run(["pkill", "-f", app_name])
    else:
        exe_name = app_name if app_name.endswith(".exe") else app_name + ".exe"
        subprocess.run(["taskkill", "/IM", exe_name, "/F"], shell=True)

##############################################################################
# 11) Main Loop
##############################################################################
def main():
    print("[Main] Starting. Press Ctrl+C to exit.")
    conversation_history = []

    try:
        while True:
            # Listen for user
            recognized_text = recognize_speech_from_mic()
            if not recognized_text:
                continue

            user_question = recognized_text
            intent_label = classify_intent(user_question)
            print(f"[Intent] => {intent_label}")

            if intent_label == "SCREENSHOT":
                base64_image = capture_screenshot_as_base64()
                if not base64_image:
                    speak_text("I couldn't capture a screenshot, Sir.")
                    continue
                final_prompt = build_vision_prompt(conversation_history, user_question)
                final_prompt += "\n\nSir, please summarize what's on this screenshot."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ]
                try:
                    response_text = llama_vision_inference(messages, model_size=11)
                    speak_text(response_text)
                    conversation_history.append((user_question, response_text))
                except Exception as e:
                    print("Screenshot error:", e)
                    speak_text("Error summarizing screenshot, Sir.")

            elif intent_label == "CAMERA":
                base64_image = capture_camera_frame_as_base64()
                if not base64_image:
                    speak_text("I couldn't capture a camera frame, Sir.")
                    continue
                final_prompt = build_vision_prompt(conversation_history, user_question)
                final_prompt += "\n\nSir, here's a camera image. Summarize what you see."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ]
                try:
                    response_text = llama_vision_inference(messages, model_size=11)
                    speak_text(response_text)
                    conversation_history.append((user_question, response_text))
                except Exception as e:
                    print("Camera error:", e)
                    speak_text("Error describing camera image, Sir.")

            elif intent_label == "APP_CONTROL":
                handle_app_control(user_question)

            else:  # OTHER => search + Q&A
                try:
                    results = duckduckgo_search_query(user_question, max_results=5)
                    system_prompt, user_prompt = build_text_prompt_with_search(
                        conversation_history, user_question, results
                    )
                    response_text = llama_text_inference(system_prompt, user_prompt)
                    speak_text(response_text)
                    conversation_history.append((user_question, response_text))
                except Exception as e:
                    speak_text("I encountered an error searching the internet, Sir.")
                    print("Search error:", e)

    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C. Exiting.")

    print("[Main] Exited cleanly.")

if __name__ == "__main__":
    main()

import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

##############################
# NEW PROMPT: Jarvis Persona
##############################
# This prompt instructs your model to respond "in character" as Jarvis.
template = """
You are Jarvis, Charan's personal AI assistant from the Iron Man universe.
Your style is polite, witty, and succinct. 
You address the user respectfully as "Sir," or by name if provided. 
You add subtle humor where appropriate, and you always stay in character as a resourceful AI.
Keep the responses short and to the point, and avoid overly verbose or complex replies.

Context / Conversation so far:
{history}

User just said: {question}

Now, Jarvis, please reply:
"""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.1")

# Create a chain that will combine our prompt + LLM
chain = prompt | model


# ========== Text-to-Speech Function ==========
def SpeakText(text):
    engine = pyttsx3.init()

    # Get the voices and pick the one that suits you
    voices = engine.getProperty("voices")

    # Example: pick a specific voice index or search by name
    # Option A: By index
    # engine.setProperty('voice', voices[2].id)

    # Option B: By partial name match (e.g., find a 'Daniel' voice)
    # for v in voices:
    #     if "Daniel" in v.name:
    #         engine.setProperty("voice", v.id)
    #         break

    engine.setProperty("rate", 180)  # Adjust speed
    engine.setProperty("volume", 1.0)  # 0.0 to 1.0

    engine.say(text)
    engine.runAndWait()


# ========== Speech Recognition Setup ==========
r = sr.Recognizer()
r.pause_threshold = 2.0  # seconds of non-speaking audio before a phrase is considered complete



def record_text():
    """
    Continuously listen from the microphone and return recognized text.
    """
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("Listening...")
                audio2 = r.listen(source2)
                print("Recognizing...")
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("You said:", MyText)
                return MyText
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.UnknownValueError:
            print("Unknown error occurred, please try again.")


##############################
# KEEP A "MESSAGE HISTORY"
##############################
# We'll store all conversation turns here so Jarvis has context.
messages = []


def send_to_jarvis(message):
    """
    Given a user message, incorporate conversation history and
    invoke the chain to get Jarvis's response.
    """
    # Build 'history' text from stored messages
    history_text = ""
    for msg in messages:
        role = msg["role"].title()  # "user" -> "User", "assistant" -> "Assistant"
        content = msg["content"]
        history_text += f"{role}: {content}\n"

    # Now call the chain with {history} and {question}
    response = chain.invoke({"history": history_text, "question": message})

    return response


# Start the conversation with a system-level "role" or context, if desired
messages.append({"role": "system", "content": "You are Jarvis, Charans's AI."})

while True:
    user_message = record_text()  # get user speech
    messages.append({"role": "user", "content": user_message})

    jarvis_response = send_to_jarvis(user_message)

    # Add the assistant's message to the conversation history
    messages.append({"role": "assistant", "content": jarvis_response})

    # Output the response to console & speak it aloud
    print("Jarvis:", jarvis_response)
    SpeakText(jarvis_response)

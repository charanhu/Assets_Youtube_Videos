import os
import base64
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import nest_asyncio
nest_asyncio.apply()

# ---- Environment Loader ----
from dotenv import load_dotenv, find_dotenv
def load_env():
    _ = load_dotenv(find_dotenv())

# ---- Watsonx Model Inference Classes ----
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# ---- Chat Completion Function (Llama 3.2 Vision) ----
def llama32(messages, model_size=11):
    """
    Generate a chat completion using IBM Watsonx Llama 3.2 Vision Instruct.

    Parameters:
    - messages (list of dict): The messages for the chat session.
    - model_size (int, optional): Specifies the model size. Default is 11.

    Returns:
    - str: The generated response content.
    """
    load_env()  # Load environment variables from .env
    model_id = f"meta-llama/llama-3-2-{model_size}b-vision-instruct"

    # Fetch environment variables
    api_key = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not all([api_key, project_id, model_id]):
        raise Exception(
            "Missing one or more required environment variables:\n"
            "  - WATSONX_API_KEY\n"
            "  - WATSONX_URL\n"
            "  - WATSONX_PROJECT_ID\n"
            f"  - model_id={model_id} from code"
        )

    # Define generation parameters
    params = {
        "max_new_tokens": 4096,  # or adjust to your preference
        "temperature": 0.0,      # adjust if you want more creative answers
        "stop_sequences": ["<|eot_id|>", "<|eom_id|>"],
    }

    credentials = Credentials(api_key=api_key, url=url)
    model_inference = ModelInference(
        model_id=model_id, params=params, credentials=credentials, project_id=project_id
    )

    # Call the chat method
    response = model_inference.chat(messages=messages)

    # Extract and return the generated content
    return response["choices"][0]["message"]["content"]


# ---- Helper to Display an Image in Notebook/Script ----
def disp_image(address):
    """
    Display an image from either a URL or a local file path.
    """
    if address.startswith("http://") or address.startswith("https://"):
        response = requests.get(address)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(address)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

# ---- Helper to Encode a Local Image in Base64 ----
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ----------------- MAIN SCRIPT EXAMPLE -----------------

if __name__ == "__main__":
    # 1) Load any environment variables (API keys, project IDs, etc.)
    load_env()

    # 2) Specify your local image path
    local_image_path = "image.jpeg"  # Change if needed

    # 3) Encode local image to Base64
    base64_image = encode_image(local_image_path)

    # 4) Display the local image (optional, for notebooks or interactive sessions)
    disp_image(local_image_path)

    # 5) Prepare a chat-style prompt with text + image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this local image in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]

    # 6) Call llama32 to get an answer about the image
    response = llama32(messages, model_size=11)

    # 7) Print the result
    print("Model Response:\n", response)

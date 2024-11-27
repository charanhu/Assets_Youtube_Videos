import gradio as gr
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain import prompts, chat_models, hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List
# from pydantic import BaseModel, Field
from typing import List
import requests
import base64
from PIL import Image
import os
import os
from dotenv import load_dotenv

load_dotenv()

# Replace 'your_nvapi_key_here' with your actual NVIDIA API key
nvapi_key = os.getenv("NVIDIA_API_KEY")  # Obtain this from NVIDIA API catalog

# # Set your NVIDIA API key
# nvapi_key = "YOUR_NVIDIA_API_KEY"

# Define the Content Creator Agent

# 1. Construct the system prompt
prompt_template = """
### [INST]

You are an expert social media content creator.
Your task is to create a different promotion message with the following 
Product Description :
------
{product_desc}
------
The output promotion message MUST use the following format :
'''
Title: a powerful, short message that depict what this product is about 
Message: be creative for the promotion message, but make it short and ready for social media feeds.
Tags: the hashtag humans will normally use in social media
'''
Begin!
[/INST]
"""

prompt = PromptTemplate(
    input_variables=["product_desc"],
    template=prompt_template,
)


# 3. Structural output using LMFE
class StructureOutput(BaseModel):
    Title: str = Field(description="Title of the promotion message")
    Message: str = Field(description="The actual promotion message")
    Tags: List[str] = Field(
        description="Hashtags for social media, usually starts with #"
    )


# 4. A powerful LLM
llm_with_output_structure = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct"
).with_structured_output(StructureOutput)

# Construct the content_creator agent
content_creator = prompt | llm_with_output_structure

# Now define the Digital Artist Agent


def llm_rewrite_to_image_prompts(user_query):
    prompt_template = """
Summarize the following user query into a very short, one-sentence theme for image generation, MUST follow this format : A iconic, futuristic image of , no text, no amputation, no face, bright, vibrant

{input}
"""
    prompt = PromptTemplate(
        input_variables=["input"],
        template=prompt_template,
    )
    model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
    chain = prompt | model | StrOutputParser()
    out = chain.invoke({"input": user_query})
    return out


def generate_image(prompt_str: str) -> str:
    """
    Generate image from text.
    Args:
        prompt_str: input text
    """
    # Re-writing the input promotion title into appropriate image_gen prompt
    gen_prompt = llm_rewrite_to_image_prompts(prompt_str)
    print("Start generating image with LLM rewritten prompt:", gen_prompt)
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {nvapi_key}",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{"text": gen_prompt}],
        "seed": 0,
        "sampler": "K_EULER_ANCESTRAL",
        "steps": 50,
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    imgdata = base64.b64decode(response_body["artifacts"][0]["base64"])
    filename = "output.jpg"
    with open(filename, "wb") as f:
        f.write(imgdata)
    # Return the path to the image file
    return filename


# Bind the image generation into the selected LLM and wrap it to create the Digital Artist Agent:

llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
llm_with_img_gen_tool = llm.bind_tools([generate_image], tool_choice="generate_image")


# For the purpose of this code, we define a simple output_to_invoke_tools function
def output_to_invoke_tools(output):
    # Since the LLM is supposed to invoke the generate_image tool based on output, we'll just call it directly
    image_path = generate_image(output)
    return image_path


digital_artist = llm_with_img_gen_tool | output_to_invoke_tools

# Now, we need to adapt the code to work in Gradio.

# Since in Gradio, we can provide a UI for the user to input the product description and select the agent.

# Let's define a function that takes user inputs and returns outputs.


def agent_workflow(product_desc, agent_choice):
    if agent_choice == "ContentCreator":
        structured_respond = content_creator.invoke({"product_desc": product_desc})
        respond = "\n".join(
            [
                f"Title: {structured_respond.Title}",
                f"Message: {structured_respond.Message}",
                f"Tags: {' '.join(structured_respond.Tags)}",
            ]
        )
        return respond, None
    elif agent_choice == "DigitalArtist":
        image_path = generate_image(product_desc)
        # Load the image and return it
        image = Image.open(image_path)
        return None, image
    else:
        return (
            "Invalid agent choice. Please select ContentCreator or DigitalArtist.",
            None,
        )


# Now, we can build the Gradio interface


def main():
    # Define Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# AI Agent for Personalized Social Media Content")
        with gr.Row():
            product_desc_input = gr.Textbox(
                label="Product Description",
                lines=3,
                placeholder="Enter the product description",
            )
        with gr.Row():
            agent_choice = gr.Radio(
                choices=["ContentCreator", "DigitalArtist"], label="Select Agent"
            )
        with gr.Row():
            submit_button = gr.Button("Submit")
        with gr.Row():
            output_text = gr.Textbox(label="Output Text")
            output_image = gr.Image(label="Output Image")

        def run_agent(product_desc, agent_choice):
            text_output, image_output = agent_workflow(product_desc, agent_choice)
            return text_output, image_output

        submit_button.click(
            fn=run_agent,
            inputs=[product_desc_input, agent_choice],
            outputs=[output_text, output_image],
        )
    demo.launch()


if __name__ == "__main__":
    main()

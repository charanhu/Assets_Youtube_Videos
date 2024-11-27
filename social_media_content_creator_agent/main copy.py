# Install necessary packages
# !pip install langchain langchain_nvidia_ai gradio

# Import necessary libraries
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain import prompts
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import requests
import base64
from PIL import Image
import gradio as gr
from langchain_community.tools import HumanInputRun
from langgraph.graph import END, StateGraph
from colorama import Fore, Style
import os
from dotenv import load_dotenv

load_dotenv()

# Replace 'your_nvapi_key_here' with your actual NVIDIA API key
nvapi_key = os.getenv("NVIDIA_API_KEY")  # Obtain this from NVIDIA API catalog

# Define the Content Creator Agent
# 1. Define the prompt template
prompt_template = """
### [INST]

You are an expert social media content creator.
Your task is to create a promotional message with the following 
Product Description:
------
{product_desc}
------
The output promotion message MUST use the following format:
'''
Title: [Title]
Message: [Message]
Tags: [Tags]
'''
Begin!
[/INST]
"""

prompt = PromptTemplate(
    input_variables=["product_desc"],
    template=prompt_template,
)


# 2. Define structured output using Pydantic
class StructureOutput(BaseModel):
    Title: str = Field(description="Title of the promotion message")
    Message: str = Field(description="The actual promotion message")
    Tags: List[str] = Field(
        description="Hashtags for social media, usually starting with #"
    )


# 3. Initialize the LLM with structured output
llm_with_output_structure = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct"
).with_structured_output(StructureOutput)

# 4. Construct the Content Creator Agent
content_creator = prompt | llm_with_output_structure


# Define the Digital Artist Agent
# Function to rewrite user queries into image generation prompts
def llm_rewrite_to_image_prompts(user_query):
    prompt = prompts.PromptTemplate(
        input_variables=["input"],
        template="""
Summarize the following user query into a very short, one-sentence theme for image generation. 
Format: An iconic, futuristic image of [theme], no text, no amputation, no face, bright, vibrant.
{input}
""",
    )

    model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
    chain = prompt | model
    out = chain.invoke({"input": user_query})
    return out


# Function to generate images using NVIDIA's sdXL-turbo model
def generate_image(prompt: str) -> str:
    # Rewriting the prompt for image generation
    gen_prompt = llm_rewrite_to_image_prompts(prompt)
    print("Generating image with prompt:", gen_prompt)

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
    return filename


# Bind the image generation function as a tool
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
llm_with_img_gen_tool = llm.bind_tools([generate_image], tool_choice="generate_image")

# Construct the Digital Artist Agent
digital_artist = llm_with_img_gen_tool


# Define Human-in-the-Loop Interaction
# Function to get human input and choose the agent
def get_human_input() -> str:
    print("Select an agent to help with the task:")
    print("1. ContentCreator")
    print("2. DigitalArtist")
    choice = input("Enter 1 or 2: ")
    if choice == "1":
        return "ContentCreator"
    elif choice == "2":
        return "DigitalArtist"
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return get_human_input()


# Define functions for the workflow nodes
def human_assign_to_agent(state):
    agent_choice = get_human_input()
    return {"agent_choice": agent_choice}


def agent_execute_task(state):
    input_to_agent = state["input_to_agent"]
    choosen_agent = state["agent_choice"]
    if choosen_agent == "ContentCreator":
        structured_response = content_creator.invoke({"product_desc": input_to_agent})
        response = f"Title: {structured_response.Title}\nMessage: {structured_response.Message}\nTags: {' '.join(structured_response.Tags)}"
    elif choosen_agent == "DigitalArtist":
        image_path = generate_image(input_to_agent)
        response = f"Image generated and saved at {image_path}"
    else:
        response = "Please reselect the agent."
    print(response)
    return {"agent_response": response}


# Build the workflow graph
workflow = StateGraph()
workflow.add_node("start", human_assign_to_agent)
workflow.add_node("end", agent_execute_task)
workflow.set_entry_point("start")
workflow.add_edge("start", "end")
workflow.add_edge("end", END)
app = workflow.compile()


# Create the Gradio Interface
def interact(input_text, input_to_agent):
    response = app.invoke({"input": input_text, "input_to_agent": input_to_agent})
    return response["agent_response"]


iface = gr.Interface(
    fn=interact,
    inputs=["text", "text"],
    outputs="text",
    title="AI Agent Interaction",
    description="Interact with Content Creator and Digital Artist Agents.\n\nEnter your task description and the input for the agent.",
)

# Launch the Gradio app
iface.launch()

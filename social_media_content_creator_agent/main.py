import gradio as gr
import requests
import base64
import io
from PIL import Image
from typing import List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain import prompts
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pymupdf  # Use PyMuPDF
import os
from dotenv import load_dotenv

load_dotenv()

# Replace 'your_nvapi_key_here' with your actual NVIDIA API key
nvapi_key = os.getenv("NVIDIA_API_KEY")  # Obtain this from NVIDIA API catalog


# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_input):
    if pdf_input is None:
        return ""
    # pdf_input is the path to the uploaded PDF file
    doc = pymupdf.open(pdf_input)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Function to generate LinkedIn post content
def generate_promo_content(paper_text):
    # Define the prompt template
    prompt_template = """
### [INST]

You are a social media content creator.
Your task is to write a clear and engaging LinkedIn post to share and promote a research paper from arXiv. The post should be informative, concise, and appealing to a professional audience. 

Details of the research paper:
------
{paper_text}
------

The post should:
- Start with a simple, attention-grabbing title or opening line to spark interest.
- Provide a brief summary of the paper:
  * What the paper is about (the main topic or problem it addresses).
  * Why it is important or interesting (its significance or potential impact).
  * Key insights, findings, or methods (without getting too technical).
- End with a short call to action or question to engage readers, such as:
  * “Check out the paper here: [link].”
  * “What are your thoughts on this?”
- Include relevant hashtags (e.g., #ResearchPaper, #AI, #arXiv) at the end for visibility.

Keep the tone professional but easy to understand, ensuring the post is suitable for a LinkedIn audience.

Begin!

[/INST]
"""
    prompt = PromptTemplate(
        input_variables=["paper_text"],
        template=prompt_template,
    )

    # Define the structured output
    class StructureOutput(BaseModel):
        Title: str = Field(description="Title of the promotion message")
        Message: str = Field(description="The actual promotion message")
        Tags: List[str] = Field(
            description="Hashtags for social media, usually starts with #"
        )

    # Use the LLM with structured output
    llm_with_output_structure = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct"
    ).with_structured_output(StructureOutput)

    # Construct the content_creator agent
    content_creator = prompt | llm_with_output_structure

    # Invoke the agent
    output = content_creator.invoke({"paper_text": paper_text})

    # Format the output
    result = f"Title: {output.Title}\n\nMessage: {output.Message}\n\nTags: {' '.join(output.Tags)}"
    return result


# Function to rewrite the input text into an image prompt
def llm_rewrite_to_image_prompts(user_query):
    prompt = prompts.ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following user query into a very short, one-sentence theme for image generation, MUST follow this format: An iconic, futuristic image of ..., no text, no amputation, no face, bright, vibrant, high quality 4k HDR image",
            ),
            ("user", "{input}"),
        ]
    )
    model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
    chain = prompt | model | StrOutputParser()
    out = chain.invoke({"input": user_query})
    return out


# Function to generate an image using the rewritten prompt
def generate_image(paper_text):
    # Rewriting the input prompt
    gen_prompt = llm_rewrite_to_image_prompts(paper_text)

    # Generate image using NVIDIA sdXL-turbo text-to-image model
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"
    headers = {
        "Authorization": f"Bearer {nvapi_key}",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{"text": gen_prompt}],
        "seed": 0,
        "sampler": "K_EULER_ANCESTRAL",
        "steps": 20,
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    imgdata = base64.b64decode(response_body["artifacts"][0]["base64"])
    image = Image.open(io.BytesIO(imgdata))
    return image


# Main function to generate both LinkedIn post and image
def generate_post_and_image(text_input, pdf_input):
    if text_input:
        paper_text = text_input
    elif pdf_input:
        paper_text = extract_text_from_pdf(pdf_input)
    else:
        return "Please provide either text or PDF of the arXiv paper.", None

    # Now generate the LinkedIn post
    promo_content = generate_promo_content(paper_text)

    # Generate the image
    image = generate_image(paper_text)

    return promo_content, image


# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Agent for LinkedIn Post and Image Creation")
    gr.Markdown("## Input arXiv Paper")
    with gr.Row():
        text_input = gr.Textbox(
            label="Paste arXiv paper text here",
            placeholder="Paste the text of the paper here",
            lines=10,
        )
        pdf_input = gr.File(
            label="Or upload arXiv paper PDF",
            file_types=[".pdf"],
            file_count="single",
        )
    generate_button = gr.Button("Generate LinkedIn Post and Image")
    promo_output = gr.Textbox(label="LinkedIn Post")
    image_output = gr.Image(label="Generated Image")
    generate_button.click(
        fn=generate_post_and_image,
        inputs=[text_input, pdf_input],
        outputs=[promo_output, image_output],
    )

demo.launch()

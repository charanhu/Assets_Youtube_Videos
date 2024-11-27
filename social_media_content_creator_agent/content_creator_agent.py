from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain import prompts, chat_models, hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List
 
 
## 1. construct the system prompt ---------
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
Title: a powerful, short message that dipict what this product is about 
Message: be creative for the promotion message, but make it short and ready for social media feeds.
Tags: the hash tag human will nomally use in social media
'''
Begin!
[/INST]
 """
prompt = PromptTemplate(
input_variables=['produce_desc'],
template=prompt_template,
)
 
 
## 2. provide seeded product_desc text
product_desc="Explore the latest community-built AI models with an API optimized and accelerated by NVIDIA, then deploy anywhere with NVIDIA NIM™ inference microservices."
 
 
## 3. structural output using LMFE 
class StructureOutput(BaseModel):     
    Title: str = Field(description="Title of the promotion message")
    Message : str = Field(description="The actual promotion message")
    Tags: List[str] = Field(description="Hashtags for social media, usually starts with #")
 
 
## 4. A powerful LLM 
llm_with_output_structure=ChatNVIDIA(model="meta/llama-3.1-405b-instruct").with_structured_output(StructureOutput)     
 
 
## construct the content_creator agent
content_creator = ( prompt | llm_with_output_structure )
out=content_creator.invoke({"product_desc":product_desc})
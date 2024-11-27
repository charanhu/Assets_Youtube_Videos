from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain import prompts, chat_models, hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
 
 
def llm_rewrite_to_image_prompts(user_query):
    prompt = prompts.ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following user query into a very short, one-sentence theme for image generation, MUST follow this format : A iconic, futuristic image of , no text, no amputation, no face, bright, vibrant",
            ),
            ("user", "{input}"),
        ]
    )
    model = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
    chain = ( prompt    | model   | StrOutputParser() )
    out= chain.invoke({"input":user_query})
    #print(type(out))
    return out

## bind image generation as tool into llama3.1-405b llm
llm=ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
llm_with_img_gen_tool=llm.bind_tools([generate_image],tool_choice="generate_image")
## use LCEL to construct Digital Artist Agent
digital_artist = (
    llm_with_img_gen_tool
    | output_to_invoke_tools
)
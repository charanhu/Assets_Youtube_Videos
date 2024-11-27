import requests
import base64, io
from PIL import Image
import requests, json
def generate_image(prompt :str) -> str :
    """
    generate image from text
    Args:
        prompt: input text
    """
    ## re-writing the input promotion title in to appropriate image_gen prompt 
    gen_prompt=llm_rewrite_to_image_prompts(prompt)
    print("start generating image with llm re-write prompt:", gen_prompt)
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"
     
    headers = {
        "Authorization": f"Bearer {nvapi_key}",
        "Accept": "application/json",
    }
     
    payload = {
        "text_prompts": [{"text": gen_prompt}],
        "seed": 0,
        "sampler": "K_EULER_ANCESTRAL",
        "steps": 2
    }
     
    response = requests.post(invoke_url, headers=headers, json=payload)
     
    response.raise_for_status()
    response_body = response.json()
    ## load back to numpy array 
    print(response_body['artifacts'][0].keys())
    imgdata = base64.b64decode(response_body["artifacts"][0]["base64"])
    filename = 'output.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)   
    im = Image.open(filename)  
    img_location=f"the output of the generated image will be stored in this path : {filename}"
    return img_location

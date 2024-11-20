import requests
import json

from dotenv import load_dotenv
import os
_ = load_dotenv() #loads 'TOGETHER_API_KEY'


def llama31(prompt_or_messages, model_size=8, temperature=0, raw=False, debug=False):
    model = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct-Turbo"
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }
    else:
        messages = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

    if debug:
        print(payload)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()  # Raises HTTPError for bad responses
        res = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    if 'error' in res:
        raise Exception(f"API Error: {res['error']}")

    if raw:
        return res

    if isinstance(prompt_or_messages, str):
        return res['choices'][0].get('text', '')
    else:
        return res['choices'][0].get('message', {}).get('content', '')

# pretty print JSON with syntax highlighting
import json
from pygments import highlight, lexers, formatters
def cprint(response):
    formatted_json = json.dumps(response, indent=4)
    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(colorful_json)












# The code below should be added to a util.py file

import requests
import json

from dotenv import load_dotenv, find_dotenv
import os
from wolframalpha import Client

def load_env():
    _ = load_dotenv(find_dotenv())

  # The right API to pass in a prompt (of type string) is the completions API https://docs.together.ai/reference/completions-1
  # The right API to pass in a messages (of type of list of message) is The chat completions API https://docs.together.ai/reference/chat-completions-1

def get_wolfram_alpha_api_key():
    load_env()
    wolfram_alpha_api_key = os.getenv("WOLFRAM_ALPHA_KEY")
    return wolfram_alpha_api_key

def get_tavily_api_key():
    load_env()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return tavily_api_key

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

import nest_asyncio
nest_asyncio.apply()
    
def wolfram_alpha(query: str) -> str:
    WOLFRAM_ALPHA_KEY = get_wolfram_alpha_api_key()
    client = Client(WOLFRAM_ALPHA_KEY)
    result = client.query(query)

    results = []
    for pod in result.pods:
        if pod["@title"] == "Result" or pod["@title"] == "Results":
          for sub in pod.subpods:
            results.append(sub.plaintext)

    return '\n'.join(results)




def trending_songs(country_name, top_number):
  try:
      top_number = int(top_number)
  except Exception:
      country_name, top_number = top_number, int(country_name)
     
  songs = {
        "US": [
            "Blinding Lights - The Weeknd",
            "Levitating - Dua Lipa",
            "Peaches - Justin Bieber",
            "Save Your Tears - The Weeknd",
            "Good 4 U - Olivia Rodrigo",
            "Montero (Call Me By Your Name) - Lil Nas X",
            "Kiss Me More - Doja Cat",
            "Stay - The Kid LAROI, Justin Bieber",
            "Drivers License - Olivia Rodrigo",
            "Butter - BTS"
        ],
        "France": [
            "Dernière danse - Indila",
            "Je te promets - Johnny Hallyday",
            "La Vie en rose - Édith Piaf",
            "Tout oublier - Angèle",
            "Rien de tout ça - Amel Bent",
            "J'ai demandé à la lune - Indochine",
            "Bella - Maître Gims",
            "À nos souvenirs - Tino Rossi",
            "Le Sud - Nino Ferrer",
            "La Nuit je mens - Alain Bashung"
        ],
        "Spain": [
            "Despacito - Luis Fonsi",
            "Bailando - Enrique Iglesias",
            "Con altura - Rosalía, J.Balvin",
            "Súbeme la Radio - Enrique Iglesias",
            "Hawái - Maluma",
            "RITMO (Bad Boys for Life) - Black Eyed Peas, J Balvin",
            "Dákiti - Bad Bunny, Jhay Cortez",
            "Vivir mi vida - Marc Anthony",
            "Una vaina loca - Farruko, Sharlene",
            "Te boté - Nio García, Casper Mágico, Ozuna"
        ]
    }

  # Find the list of songs for the given country
  if country_name in songs:
    return songs[country_name][:top_number]

  # If the country is not found, return an empty list
  return []



def get_boiling_point(liquid_name, celsius):
  # function body
  return []

# Import necessary module
import math

def calculate_loan(loan_amount, annual_interest_rate, loan_term, down_payment):
    monthly_interest_rate = annual_interest_rate / 12
    loan_term_months = loan_term * 12
    
    # Calculate the loan amount after down payment
    loan_amount_after_down_payment = loan_amount - down_payment
    
    # Calculate the monthly payment
    monthly_payment = loan_amount_after_down_payment * (monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months) / ((1 + monthly_interest_rate) ** loan_term_months - 1)
    
    # Calculate the total payment
    total_payment = monthly_payment * loan_term_months
    
    # Calculate the total interest paid
    total_interest_paid = total_payment - loan_amount_after_down_payment
    return math.floor(monthly_payment), math.floor(total_payment), math.floor(total_interest_paid)

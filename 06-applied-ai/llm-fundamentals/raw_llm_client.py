import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.1-8b-instant"
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY")

class GroqClient:
    def __init__(self, api_key, base_url=BASE_URL, default_model=MODEL):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        
    def chat(self, messages, model=None, temperature=0):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model or self.default_model,
            'messages': messages,
            'temperature': temperature
        }
        response = requests.post(self.base_url, headers=headers, json=payload)
        print(response.status_code)
        print(response.text)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    

def send(prompt):
    client = GroqClient(api_key=API_KEY)
    messages = [
        {'role': 'system', 'content': "You are Jarvis, from Iron-Man; a very helpfull assistent."},
        {'role': 'user', 'content': prompt}
    ]
    resp = client.chat(messages)
    return resp


if __name__ == "__main__":
    print(send("hello"))
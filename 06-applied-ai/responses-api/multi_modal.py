import os, json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys_prompt = "You are a helpful Teacher who provides concise, personalized explanations. Be very brief."

IMAGE_PATH = Path(__file__).resolve().parents[2] / "files/images" / "llm-development-cycle.png"


def response(messages):
    return client.responses.create(
        model="gpt-5.4-mini",       
        reasoning={
            "effort": "none",   # options = none | low | medium | high | xhigh
            "summary": "auto"
        }, 
        max_output_tokens=300,
        instructions=sys_prompt,
        input=messages
    )


def load_local_image(image_path):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64


def main():
    local_image = load_local_image(IMAGE_PATH)
    local_img_url = f"data:image/png;base64,{local_image}"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "input_text",  "text": "Briefly interpret this stock plot"},
            {"type": "input_image", "image_url": local_img_url}]
        },
    ]
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            
            messages.append({"role": "user", "content": user_input})
            resp = response(messages)
            messages += resp.output

            print(f"\nAssistant: {resp.output_text}\n")
            messages.append({"role": "assistant", "content": resp.output_text})
    except Exception as e:
        print(f"Error: {str(e)}")



if __name__ == "__main__":
    main()
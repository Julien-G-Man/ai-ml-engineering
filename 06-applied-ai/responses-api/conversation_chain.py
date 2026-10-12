import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tool_calling import tools, convert_currency

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys_prompt = "You are a helpful Teacher who provides concise, personalized explanations. Be very brief."

def response(input: str, prev_resp_id: str | None = None):
    prev_id = prev_resp_id if prev_resp_id is not None else None
    return client.responses.create(
        model="gpt-5.4-mini",
        tools=tools,            #[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        reasoning={
            "effort": "none",   # options = none | low | medium | high | xhigh
            "summary": "auto"
        }, 
        max_output_tokens=300,
        instructions=sys_prompt,
        input=input,
        previous_response_id=prev_id
    )


def main():
    latest_response_id = None
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower == "exit":
                break
            resp = response(user_input, latest_response_id)
         
            print(f"\nAssistant: {resp.output_text}\n")
            latest_response_id = resp.id
    except Exception as e:
        print(f"Error: {str(e)}")


def resp_results_by_type(resp):
    for item in resp.output:
        if item.type == "message":
            if item.status == "incomplete":
                print("\n Message cut-off")
        
        if item.type == "reasoning":
            if item.summary:
                print(f"\nReasoning Summary: {item.summary[0]}")
            else:
                print("\nNo reasoning summary available")
        
        if item.type == "message":
            print(f"\nAssistant: {item.content[0].text}")
            
        if item.type == "web_search_call":
            print(f"\nSearch sources: {item.action.sources}")
                
            # other item types: web_search_call | function_call | function_call_output
             
     
if __name__ == "__main__":
    main()
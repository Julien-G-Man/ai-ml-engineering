import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tool_calling import tools

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def streaming_response(prompt: str):
    with client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        stream=True
    ) as stream:
        current_text = ""
        for event in stream:
            if event.type == "response.output_text.delta":
                current_text += event.delta
                print(current_text)


def stream_multiple_events(prompt: str):
    with client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        stream=True
    ) as stream:
        current_text = ""
        for event in stream:
            if event.type == "response.created":
                print("Response started...\n")
            
            if event.type == "response.output_text.done":
                print("\n\n---- Text block finished ----\n")
                
            if event.type == "response.completed":
                current_text += event.delta
                print(f"\nFUll Response: \n{current_text}")
    
    
def stream_with_tool_call(prompt: str):
    with client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        tools=tools,
        stream=True
    ) as stream:
        current_args = ""
        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                current_args += event.delta
                print("Streaming args: ", current_args)
            elif event.type == "response.function_call_arguments.done":
                print("\nFinal arguments: ", event.arguments)
            elif event.type == "response.completed":
                print("\n----- Completed -----")
            
    
prompt = """Explain how a neural network learns concisely for a child"""

def main():
    streaming_response(prompt)
    

if __name__ == "__main__":
    main()
    
    
    
"""
Structured updates that describe what's happening during streaming

Event Type                                       Description
- response.created                            -> The model has started generating
- response.output_text.delta                  -> Partial text update
- response.output_text.done                   -> Text block complete
- response.function_calling.arguments.delta   -> Streaming tool arguments
- response.completed                          -> The entire response is finished

"""

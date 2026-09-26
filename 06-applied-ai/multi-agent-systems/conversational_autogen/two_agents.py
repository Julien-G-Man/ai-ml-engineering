import os 
from dotenv import load_dotenv 
from autogen import AssistantAgent, UserProxyAgent 
  
load_dotenv() 
  
llm_config = { 
    "config_list": [{"model": "gpt-4o-mini", 
                     "api_key": os.getenv("OPENAI_API_KEY")}], 
    "temperature": 0,
    "timeout": 120,
}

assistant = AssistantAgent( 
    name="Coder", 
    system_message=( 
        "You write clean Python. When the task is solved and verified, " 
        "reply with the single word TERMINATE." 
    ), 
   llm_config=llm_config
)


def is_done(msg: dict):
    content = (msg.get("content") or "").strip()
    return content.endswith("TERMINATE")

# The proxy RUNS code blocks the assistant writes. No human typing needed. 
executor = UserProxyAgent( 
    name="Executor", 
    human_input_mode="NEVER",          # fully automated 
    max_consecutive_auto_reply=8,      # safety cap 
    is_termination_msg=is_done,
    code_execution_config={"work_dir": "coding", "use_docker": False}, 
)

prompt = "Write a function is_palindrome(s) and test it on 3 cases."

executor.initiate_chat( 
    assistant, 
    message=prompt
) 


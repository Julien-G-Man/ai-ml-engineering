import os
from dotenv import load_dotenv 
from autogen import AssistantAgent, UserProxyAgent 
  
load_dotenv() 
  
LLM_CONFIG = { 
    "config_list": [{
        "model": "gpt-4o-mini", 
        "api_key": os.getenv("OPENAI_API_KEY")
    }], 
    "temperature": 0, 
    "timeout": 120, 
} 
  
def is_done(msg: dict) -> bool: 
    return (msg.get("content") or "").strip().endswith("TERMINATE") 
  
def make_agents(): 
    planner = AssistantAgent("Planner", llm_config=LLM_CONFIG, system_message=( 
        "You are a software planner. Break the user's request into 2-4 concrete " 
          "steps. Do NOT write code. Hand off to the Coder.")) 
  
    coder = AssistantAgent("Coder", llm_config=LLM_CONFIG, system_message=( 
        "You are a Python developer. Implement the plan in a single fenced python " 
        "code block, including a few asserts that test it. Keep it self-contained.")) 
  
    reviewer = AssistantAgent("Reviewer", llm_config=LLM_CONFIG, system_message=( 
        "You review code against the request. If the Executor reports the code ran " 
        "and all asserts passed, reply with a one-line summary ending in TERMINATE. " 
        "Otherwise, state the specific problem for the Coder to fix.")) 
  
    executor = UserProxyAgent("Executor", 
        human_input_mode="NEVER", 
        max_consecutive_auto_reply=10, 
        is_termination_msg=is_done, 
        code_execution_config={"work_dir": "coding", "use_docker": False}) 
  
    return planner, coder, reviewer, executor

 
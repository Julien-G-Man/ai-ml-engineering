from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager 
from two_agents import llm_config
  
planner = AssistantAgent("Planner", llm_config=llm_config, 
    system_message="Break the task into clear steps. Do not write code yourself.") 
  
coder = AssistantAgent("Coder", llm_config=llm_config, 
    system_message="Implement the Planner's steps in Python. Write a code block.") 
  
reviewer = AssistantAgent("Reviewer", llm_config=llm_config, 
    system_message=("Critique the Coder's code against the task. If it is correct " 
                    "AND the Executor reports passing tests, reply TERMINATE.")) 
  
executor = UserProxyAgent("Executor", 
    human_input_mode="NEVER", max_consecutive_auto_reply=10, 
    is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""), 
    code_execution_config={"work_dir": "coding", "use_docker": False}) 
  
group = GroupChat( 
    agents=[planner, coder, reviewer, executor], 
    messages=[], max_round=12,          # hard cap on total turns 
    speaker_selection_method="auto",    # manager chooses next speaker 
) 
manager = GroupChatManager(groupchat=group, llm_config=llm_config) 
  
prompt = "Build and test a function that returns the first N prime numbers."

executor.initiate_chat(manager, 
    message=prompt
) 
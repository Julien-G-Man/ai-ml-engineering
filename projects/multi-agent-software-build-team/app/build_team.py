from autogen import GroupChat, GroupChatManager
from app.build_agents import make_agents, LLM_CONFIG 

def run_build(task: str) -> dict:
    planner, coder, reviewer, executor = make_agents() 
  
    group = GroupChat( 
        agents=[planner, coder, reviewer, executor], 
        messages=[], 
        max_round=12,
        speaker_selection_method="round_robin",
    ) 
    manager = GroupChatManager(groupchat=group, llm_config=LLM_CONFIG) 
  
    executor.initiate_chat(manager, message=task) 
  
    # Collect the transcript and the last code-execution result 
    transcript = [{"name": m.get("name"), "content": m.get("content")} 
                  for m in group.messages] 
    final = group.messages[-1]["content"] if group.messages else "" 
    return {
        "final": final, 
        "transcript": transcript, 
        "rounds": len(group.messages)
    } 
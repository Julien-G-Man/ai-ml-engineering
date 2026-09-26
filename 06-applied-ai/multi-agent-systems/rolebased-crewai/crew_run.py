from crewai import Crew, Process 
from tasks import intake_task, research_task, differential_task, writeup_task 
from agents import intake, researcher, differential, safety_writer 
  
triage_crew = Crew( 
    agents=[intake, researcher, differential, safety_writer], 
    tasks=[intake_task, research_task, differential_task, writeup_task], 
    process=Process.sequential,   # assembly-line order 
    verbose=True, 
) 
  
# Hierarchical: a manager LLM plans and delegates
# The manager decides who works when -> more flexible, more expensive. 
hier_crew = Crew( 
    agents=[researcher, differential, safety_writer], 
    tasks=[research_task, differential_task, writeup_task], 
    process=Process.hierarchical,  
    manager_llm="gpt-40-mini",   # required for hierarchical    
    verbose=True, 
) 
  

complaint = "I've had a dull headache for two days, worse in the evenings."
result = triage_crew.kickoff(inputs={"complaint": complaint}) 

print("\n===== FINAL BRIEFING =====\n", result) 
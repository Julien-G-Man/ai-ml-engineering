from crewai import Crew, Process 
from app.agents import intake, researcher, differential, safety_writer 
from app.tasks import build_tasks 
  
def run_triage(complaint: str) -> str: 
    crew = Crew( 
        agents=[intake, researcher, differential, safety_writer], 
        tasks=build_tasks(), 
        process=Process.sequential, 
        verbose=True, 
    ) 
    result = crew.kickoff(inputs={"complaint": complaint}) 
    return str(result) 
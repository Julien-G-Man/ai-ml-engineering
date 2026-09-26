from crewai import Task 
from app.agents import intake, researcher, differential, safety_writer 
  
def build_tasks(): 
    intake_task = Task( 
        description=("Patient complaint: '{complaint}'.\n" 
            "Extract and normalise: main symptom(s), duration, severity (1-10 if " 
            "stated), risk factors. Output clean structured text only."), 
        expected_output="Structured normalised complaint.", 
        agent=intake) 
  
    research_task = Task( 
        description=("Using the normalised complaint, call the Symptom Reference " 
            "Lookup for each main symptom. Summarise context and red-flags. " 
            "If a symptom is not in the reference, say so."),
        expected_output="Reference brief: context + red-flags per symptom.", 
        agent=researcher, context=[intake_task]) 
  
    differential_task = Task( 
        description=("From the intake summary and reference brief, list 3-5 possible " 
            "explanations, each with ONE supporting reason. Add the disclaimer that " 
            "this is informational and not a diagnosis."), 
        expected_output="Short ranked list of possibilities with reasons + disclaimer.", 
        agent=differential, context=[intake_task, research_task]) 
  
    writeup_task = Task( 
        description=("Write the final patient-facing briefing. Structure: (1) RED-FLAGS " 
            "- seek care now if any apply, (2) what the symptoms are commonly " 
            "associated with, (3) suggested next step, (4) a clear note to consult a " 
            "qualified clinician. Calm, plain English."), 
        expected_output="A clear, safe, structured triage briefing.", 
        agent=safety_writer, context=[intake_task, research_task, differential_task]) 
  
    return [intake_task, research_task, differential_task, writeup_task] 
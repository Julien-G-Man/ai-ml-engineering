from crewai import Task
from .agents import researcher, differential, safety_writer, intake


intake_task = Task( 
    description=( 
        "A patient described this complaint: '{complaint}'.\n" 
        "Extract and normalise: main symptom(s), duration, severity (1-10 if stated), " 
        "and any mentioned risk factors. Output clean structured text." 
    ), 
    expected_output="A short structured summary of the normalised complaint.", 
    agent=intake, 
) 

 
research_task = Task( 
    description=( 
        "Using the normalised complaint, gather general reference information " 
        "about the reported symptoms: commonly associated conditions, typical " 
        "context, and recognised emergency warning signs. Cite uncertainty." 
    ), 
    expected_output="A reference brief: associated conditions + known red-flags.", 
    agent=researcher, 
    context=[intake_task],   # <-- receives intake_task's output as context 
) 

  
differential_task = Task( 
    description=(
        "From the intake summary and reference brief, list 3-5 possible " 
        "explanations, each with ONE supporting reason. Add the disclaimer that " 
        "this is informational and not a diagnosis."
    ), 
    expected_output="Short ranked list of possibilities with reasons + disclaimer.", 
    agent=differential, 
    context=[intake_task, research_task]
) 

writeup_task = Task( 
    description=(
        "Write the final patient-facing briefing. Structure: (1) RED-FLAGS " 
        "- seek care now if any apply, (2) what the symptoms are commonly " 
        "associated with, (3) suggested next step, (4) a clear note to consult a " 
        "qualified clinician. Calm, plain English."), 
    expected_output="A clear, safe, structured triage briefing.", 
    agent=safety_writer, 
    context=[intake_task, research_task, differential_task]
) 

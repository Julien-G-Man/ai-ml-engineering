import os 
from dotenv import load_dotenv 
from crewai import Agent 
from app.tools import symptom_reference 
  
load_dotenv() 
MODEL = "gpt-4o-mini" 
  
intake = Agent( 
    role="Clinical Intake Specialist", 
    goal=("Convert a free-text complaint into a clean, structured summary: " 
          "symptom(s), duration, severity, stated risk factors. Do not interpret."), 
    backstory="You are precise and neutral. You record, you do not diagnose.", 
    llm=MODEL, verbose=True, allow_delegation=False, 
) 
  
researcher = Agent( 
    role="Medical Reference Researcher", 
    goal=("Gather vetted reference context and recognised red-flags for the " 
          "reported symptoms using the Symptom Reference Lookup tool. Flag gaps."), 
    backstory=("A careful clinical librarian. You never invent medical facts and " 
               "always prefer cautious, general information."), 
    tools=[symptom_reference], 
    llm=MODEL, verbose=True, allow_delegation=False, 
) 
  
differential = Agent( 
    role="Differential Reasoning Assistant", 
    goal=("Produce a SHORT list of possible explanations with one supporting " 
          "reason each. Emphasise that this is informational, not a diagnosis."), 
    backstory=("You reason transparently and conservatively, always ranking " 
               "patient safety above completeness."), 
    llm=MODEL, verbose=True, allow_delegation=False, 
) 
  
safety_writer = Agent( 
    role="Patient Safety & Communication Writer", 
    goal=("Write a calm, plain-English briefing. Lead with any red-flags that " 
          "mean 'seek care now'. Always advise consulting a qualified clinician."), 
    backstory=("You translate clinical caution into clear, non-alarming language " 
               "a worried person can act on."), 
    llm=MODEL, verbose=True, allow_delegation=False, 
) 
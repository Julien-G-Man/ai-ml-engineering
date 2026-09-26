import os
from dotenv import load_dotenv
from crewai import Agent

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
    role="Senior Medical Reference Researcher", 
    goal=( 
        "Find accurate, well-sourced reference information about the " 
        "symptoms provided. Never invent facts. Prefer cautious, " 
        "general medical knowledge over speculation." 
    ), 
    backstory=( 
        "You are a meticulous clinical librarian. You do NOT diagnose. " 
        "You assemble the reference material a clinician would want, and " 
        "you always flag uncertainty rather than guessing." 
    ), 
    llm=MODEL, 
    verbose=True, 
    allow_delegation=False,   # this worker does not hand work to others 
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

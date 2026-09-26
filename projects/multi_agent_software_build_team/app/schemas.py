from pydantic import BaseModel, Field
from typing import List, Dict

class BuildRequest(BaseModel):
    task: str = Field(..., 
                min_length=10, max_length=1000, 
                description="Plain-English coding task for the agent team.") 
    
class BuildResponse(BaseModel):
    final: str
    rounds: int
    transcript: List[Dict [ str, str]]
    
    
from pydantic import BaseModel, Field 
  
class TriageRequest(BaseModel): 
    complaint: str = Field(..., min_length=5, max_length=2000, 
        description="Free-text description of the patient's symptoms.") 
  
class TriageResponse(BaseModel): 
    briefing: str = Field(..., description="The crew's final patient-facing briefing.") 
    disclaimer: str = "Informational support only. Not a medical diagnosis. Consult a qualified clinician." 
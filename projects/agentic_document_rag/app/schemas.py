from pydantic import BaseModel

class AgentResponse(BaseModel):
    content: str
from pydantic import BaseModel

class ChatQuery(BaseModel):
    text: str
    
class ChatResponse(BaseModel):
    response: str
    
class EmbeddingQuery(BaseModel):
    text: str
    
class EmbeddingResponse(BaseModel):
    embedding: list[float]
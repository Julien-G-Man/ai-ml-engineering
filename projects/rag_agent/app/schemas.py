from pydantic import BaseModel

class ChatQuery(BaseModel):
    text: str
    document_id: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    
class ChatResponse(BaseModel):
    response: str
    
class EmbeddingQuery(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

class PastMessagesResponse(BaseModel):
    messages: list[dict[str, str]]

class ClearPastConversationsResponse(BaseModel):
    status: str
    message: str
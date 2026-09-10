from pydantic import BaseModel


class SearchQuery(BaseModel):
    text: str


class SearchMatch(BaseModel):
    score: float
    text: str
    title: str | None = None


class SearchResponse(BaseModel):
    matches: list[SearchMatch]


class EmbeddingQuery(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]

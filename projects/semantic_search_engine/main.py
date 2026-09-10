from engine import v_engine
from fastapi import FastAPI
from .schemas import SearchQuery, SearchMatch, SearchResponse, EmbeddingQuery, EmbeddingResponse

app = FastAPI()


@app.get("/")
def root():
    return {"msg": "Semantic Search API"}


@app.post("/search")
def search(query: SearchQuery):
    query_response = v_engine.create_embeddings(query.text)
    query_emb = query_response.data[0].embedding
    
    retrieved_docs = v_engine.query_vector(
        vector=query_emb,
        top_k=3
    )
    matches = [
        SearchMatch(
            score=round(match["score"], 2),
            text=match["metadata"]["text"],
            title=match["metadata"]["title"]
        )
        for match in retrieved_docs['matches']
    ]
    return SearchResponse(matches=matches)


@app.post("/embed", response_model=EmbeddingResponse)
def embed(query: EmbeddingQuery):
    response = v_engine.create_embeddings(query.text)
    embedding = response.data[0].embedding
    return EmbeddingResponse(embedding=embedding)

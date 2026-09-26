import logging
from app.repo import store
from app.engine import RAGEngine, Agent
from fastapi import FastAPI, HTTPException, Depends
from .schemas import ChatQuery, ChatResponse, EmbeddingQuery, EmbeddingResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent")

agent = None
rag_engine = None


def get_agent() -> Agent:
    global agent
    if agent is None:
        agent = Agent()
    return agent


def get_rag_engine() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        logger.info("Creating RAG Engine...")
        rag_engine = RAGEngine()
        rag_engine.load_index()
    return rag_engine


@app.get("/")
def root():
    return {"message": "RAG Agent API"}


@app.post("/generate")
def generate_answer(
    query: ChatQuery,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    agent: Agent = Depends(get_agent),
):
    try:
        documents, sources = rag_engine.retrieve(query.text, top_k=3)
        context_prompt = agent.build_prompt_with_context(query.text, documents)
        response = agent.generate_response(context_prompt, sources)
        store.save_conversation(query.text, response)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
def embed(query: EmbeddingQuery,
          rag_engine: RAGEngine = Depends(get_rag_engine)
          ):
    try:
        response = rag_engine.create_embeddings(query.text)
        embedding = response.data[0].embedding
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/past_messages")
def get_past_messages():
    past_msg = store.get_past_conversations()
    return {"past_msg": past_msg}



@app.delete("/past_messages/delete")
def clear_all_past_conversations():
    store.clear_past_conversations()
    return {"status": "ok", "message": "All past conversations cleared successfully!"}


@app.delete("/past_messages/delete/<id>")
def clear_past_conversation(id: int):
    try:
        store.clear_single_conversation(id)
        return {"status": "ok", "message": f"conversation deleted successfully!"}
    except Exception as e:
        raise HTTPException(status=500, detail=str(e))
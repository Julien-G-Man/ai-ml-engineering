import logging
from fastapi import File, UploadFile
from app.documents import chunk_pages, document_id, extract_pages
from app.document_store import DocumentRecord, document_store
from app.repo import store
from app.engine import RAGEngine, Agent
from fastapi import FastAPI, HTTPException, Depends
from .schemas import (
    ChatQuery,
    ChatResponse,
    ClearPastConversationsResponse,
    DocumentUploadResponse,
    EmbeddingQuery,
    EmbeddingResponse,
    PastMessagesResponse,
)

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
        namespace = query.document_id or "youtube-rag-dataset"
        if query.document_id and document_store.get(query.document_id) is None:
            raise HTTPException(status_code=404, detail="Document was not found")
        documents, sources = rag_engine.retrieve(query.text, top_k=3, namespace=namespace)
        context_prompt = agent.build_prompt_with_context(query.text, documents)
        response = agent.generate_response(context_prompt, sources)
        store.save_conversation(query.text, response)
        return ChatResponse(response=response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    content = await file.read()
    try:
        pages = extract_pages(file.filename, content)
        doc_id = document_id(file.filename or "document", content)
        chunks = chunk_pages(doc_id, pages)
        rag_engine.upsert_document(chunks)
        record = document_store.add(DocumentRecord(doc_id, file.filename or doc_id, len(chunks)))
        return DocumentUploadResponse(
            document_id=record.document_id,
            filename=record.filename,
            chunks=record.chunks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=502, detail="Document indexing failed") from exc


@app.get("/documents", response_model=list[DocumentUploadResponse])
def list_documents():
    return [
        DocumentUploadResponse(
            document_id=record.document_id,
            filename=record.filename,
            chunks=record.chunks,
        )
        for record in document_store.list()
    ]


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
    return PastMessagesResponse(messages=past_msg)



@app.delete("/past_messages/delete")
def clear_all_past_conversations():
    store.clear_past_conversations()
    return ClearPastConversationsResponse(status="ok",
            message="All past conversations cleared successfully!")


@app.delete("/past_messages/delete/<id>")
def clear_past_conversation(id: int):
    try:
        store.clear_single_conversation(id)
        return ClearPastConversationsResponse(status="ok", message="conversation deleted successfully!")
    except Exception as e:
        raise HTTPException(status=500, detail=str(e))
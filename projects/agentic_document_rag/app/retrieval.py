from __future__ import annotations

import re
import os
import hashlib
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from app.prompts import RAG_SYSTEM_PROMPT_TEMPLATE, RAG_HUMAN_TEMPLATE

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not configured")

BASE_DIR = Path(__file__).resolve().parents[2]
FILES_DIR = (BASE_DIR / "storage").resolve()
CHROMA_PATH = FILES_DIR / "chroma"

SUPPORTED_FILE_TYPES = {
    ".pdf": "pdf",
    ".csv": "csv",
}


class RAGEngine:
    """Load, split, index, retrieve, and answer questions over local files."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size"
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._stores: dict[str, Chroma] = {}


    def available_files(self) -> list[Path]:
        return [
            path
            for path in sorted(FILES_DIR.iterdir())
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_FILE_TYPES
        ]
        
        
    @staticmethod
    def _validate_path(file_path: str | Path) -> Path:
        raw_path = Path(file_path).expanduser()
    
        if raw_path.is_absolute():
            path = raw_path.resolve()
        else:
            path = (FILES_DIR / raw_path.name).resolve()
    
        if not path.is_relative_to(FILES_DIR):
            raise ValueError(
                f"File must be inside {FILES_DIR}. "
                f"Received: {file_path!r}; resolved to: {path}"
            )
    
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
    
        if path.suffix.casefold() not in SUPPORTED_FILE_TYPES:
            supported = ", ".join(SUPPORTED_FILE_TYPES.values())
            raise ValueError(f"Unsupported file type. Supported types: {supported}")
    
        return path
    
    
    @staticmethod
    def _load_documents(path: Path) -> list[Document]:
        file_type = SUPPORTED_FILE_TYPES[path.suffix.casefold()]

        if file_type == "pdf":
            documents = PyPDFLoader(str(path)).load()
        elif file_type == "csv":
            documents = CSVLoader(str(path)).load()
        else:
            raise ValueError(f"No loader configured for file type: {file_type}")

        if not documents:
            raise ValueError(f"No readable content found in {path.name}")

        for document in documents:
            document.metadata.setdefault("source", str(path))
            document.metadata.setdefault("file_name", path.name)
            document.metadata.setdefault("file_type", file_type)

        return documents
    
    
    def split(self, documents: list[Document]) -> list[Document]:
        """Split Documents while preserving page and source metadata."""
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            raise ValueError("Document splitting produced no chunks")
        return chunks
    
    
    @staticmethod
    def _collection_name(path: Path) -> str:
        stat = path.stat()
        value = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"rag_{digest}"
    

    def index(self, file_path: str | Path) -> Chroma:
        """Load, split, and index one file exactly once per process version."""
        path = self._validate_path(file_path)
        collection_name = self._collection_name(path)

        if collection_name in self._stores:
            return self._stores[collection_name]

        documents = self._load_documents(path)
        chunks = self.split(documents)
        
        ids = [f"{path.name}-{index}" for index, _ in enumerate(chunks)]

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            ids=ids,
            collection_name=collection_name,
            persist_directory=str(CHROMA_PATH),
        )

        self._stores[collection_name] = vector_store
        return vector_store


    def retrieve(
        self,
        query: str,
        file_path: str | Path,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents from one indexed file.

        This method does not generate an answer and does not create a vector
        store directly. Indexing and retrieval are intentionally separate.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        vector_store = self.index(file_path)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k or self.top_k},
        )
        documents = retriever.invoke(query)

        if not documents:
            raise LookupError("No relevant documents were retrieved")
        return documents


    @staticmethod
    def format_documents(documents: list[Document]) -> str:
        sections: list[str] = []

        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("file_name", "unknown file")
            page = document.metadata.get("page")

            location = source
            if page is not None:
                location += f", page {page + 1}"

            sections.append(
                f"[Source {index}: {location}]\n{clean_text(document.page_content)}"
            )
            
        return "\n\n---\n\n".join(sections)   
    
    
    def answer(self, query: str, file_path: str | Path) -> str:
        """Retrieve context and generate a grounded answer.
        This method will not be used in out agentic RAG setup"""
        documents = self.retrieve(query, file_path)
        context = self.format_documents(documents)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT_TEMPLATE.strip()),
            ("human", RAG_HUMAN_TEMPLATE.strip())
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "context": context, "question": query,
        })

        return str(response.content)
    

@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    return RAGEngine()

def reset_rag_engine() -> None:
    get_rag_engine.cache_clear()
    
def clean_text(text: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


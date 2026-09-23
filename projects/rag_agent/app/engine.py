import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from app.repo import store
from app.tools import convert_currency, tools
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger(__name__)

PINECONE_INDEX_NAME = 'semantic-search'
NAMESPACE = "youtube-rag-dataset"


class RAGEngine:    
    def __init__(self):
        self._index = None
        self._vector_count = 0
        self._is_loaded = False
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def load(self) -> bool: 
        """Load the Pinecone index and set up LlamaIndex. Called once at startup.""" 
        try:    
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')) 

            existing = [idx["name"] for idx in pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,
                    metric='dotproduct',  # can also be cosine or euclidean
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            self._index = pc.Index('PINECONE_INDEX_NAME') 
  
            stats = self._index.describe_index_stats() 
            self._vector_count = stats.total_vector_count 
  
            if self._vector_count == 0: 
                logger.warning('Pinecone index is empty') 
                return False 
  
            self._is_loaded = True 
            logger.info(f'RAG engine loaded: {self._vector_count} vectors') 
            return True 
  
        except Exception as e: 
            logger.error(f'Failed to load RAG engine: {e}') 
            return False 
  
    @property 
    def is_loaded(self) -> bool: 
        return self._is_loaded 
    
    @property
    def vector_count(self) -> int:
        return self._vector_count
    

    def create_embeddings(self, query: str):
        try:
            return self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
        except Exception as e:
            logger.error(f"Failed to embed the provided text: {e}")

    def retrieve(self, query: str, top_k: int) -> tuple[list[str], list[tuple[str, str]]]:
        query_emb = self.create_embeddings(query).data[0].embedding
        retrieved_docs = []
        sources = []
        docs = self._index.query(
            vector=query_emb, 
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        for doc in docs['matches']:
            retrieved_docs.append(doc["metadata"]["text"])
            sources.append((doc["metadata"]["title"], doc["metadata"]["url"]))
        return retrieved_docs, sources

    def upsert(self, vectors):
        self._index.upsert(vectors=vectors, namespace=NAMESPACE)


class Agent():
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = """You are a helpful assistant. Keep it very brief."""

    def build_prompt_with_context(self, query: str, docs: list):
        delim = '\n\n----\n\n'
        prompt_starter = """Answer the question based on the following context:\n\nContext:\n"""
        prompt_end = f"\n\nQuestion: {query}\nAnswer: "
        prompt = prompt_starter + delim.join(docs) + prompt_end
        return prompt

    def client_response(self, messages: list[dict]):
        try:
            return self.openai_client.responses.create(
                model="gpt-5.4-mini",
                reasoning={"effort": "none"},
                input=messages,
                tools=tools,
                include=["web_search_call.action.sources"],
                tool_choice="auto",
                temperature=0
            )
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")

    def generate_response(self, prompt: str, sources: list[tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in store.get_past_messages():
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["ai"]})

        messages.append({"role": "user", "content": prompt})

        resp = self.client_response(messages)

        messages += resp.output
        has_function_call = False

        for item in resp.output:
            if item.type == "function_call":
                has_function_call = True
                if item.name == "convert_currency":
                    result = convert_currency(**json.loads(item.arguments))
                    messages.append({
                        "type":    "function_call_output",
                        "call_id": item.call_id,
                        "output":  json.dumps({"convert_currency": result}),
                    })

        if has_function_call:
            final_resp = self.client_response(messages)
            messages += final_resp.output
            logger.info(final_resp.output_text)
        else:
            logger.info(resp.output_text)

        answer = resp.output_text
        answer += "\n\nSources:"
        for source in sources:
            answer += "\n" + source[0] + ": " + source[1]
        return answer

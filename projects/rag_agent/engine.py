import os
import json
import logging
from openai import OpenAI
from repo import store
from dotenv import load_dotenv
from tools import convert_currency, tools
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger()

INDEX_NAME = 'semantic-search'
NAMESPACE = "youtube-rag-dataset"


class RAGEngine:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        existing = [idx["name"] for idx in self.pc.list_indexes()]
        if INDEX_NAME not in existing:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric='dotproduct',  # can also be cosine or euclidean
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(INDEX_NAME)

    def create_embeddings(self, query: str):
        return self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )

    def retrieve(self, query: str, top_k: int) -> tuple[list[str], list[tuple[str, str]]]:
        query_emb = self.create_embeddings(query).data[0].embedding
        retrieved_docs = []
        sources = []
        docs = self.index.query(
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
        self.index.upsert(vectors=vectors, namespace=NAMESPACE)


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
        return self.openai_client.responses.create(
            model="gpt-5.4-mini",
            reasoning={"effort": "none"},
            input=messages,
            tools=tools,
            include=["web_search_call.action.sources"],
            tool_choice="auto",
            temperature=0
        )

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

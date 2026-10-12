import os
import csv
import logging
import chromadb
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from chromadb import Collection, PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()
logger = logging.getLogger("name")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NETFLIX_TITLES_PATH = Path(__file__).resolve().parents[2] / "files" / "netflix_titles.csv"

client = PersistentClient(path="./chroma_data")


def get_or_create_collection(collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
    )
    
def insert_embeddings(collection: Collection, show_ids: list[str], descriptions: list[str], metadatas: list[dict] | None = None):
    """Send show descriptions to Chroma. Chroma takes care of embedding them."""
    add_args = {
        "ids": show_ids,
        "documents": descriptions,
    }
    if metadatas is not None:
        add_args["metadatas"] = metadatas
    collection.add(**add_args)
    
def query_collection(collection: Collection, texts: list[str]):
    if isinstance(texts, str):
        texts = [texts]
    return collection.query(
        query_texts=texts,
        n_results=2
    )
    
def query_with_metadatas(collection: Collection, show_ids: str | list[str]):
    if isinstance(show_ids, str):
        show_ids = [show_ids]
    reference_texts = collection.get(ids=show_ids)["documents"]
    return collection.query(
        query_texts = reference_texts,
        n_results=3,
        where={
            "type": {"$eq": "Movie"}
        }
    )

"""
exmaple use case of where

where={
    "$and": [
        {"rating": {"$eq": "G"}},
        {"release_year": {"$lt": 2019}}
    ]
}
"""

def update_collection(collection: Collection, show_ids: list[str], descriptions: list[str]):
    """Void function: updates collection with new show data"""
    collection.update(
        ids=show_ids,
        documents=descriptions
    )
  
def upsert_collection(collection: Collection, show_ids: list[str], descriptions: list[str]):
    """Void function:
    If IDs are missing -> add them
    If IDs are present -> update them
    """
    collection.upsert(
        ids=show_ids,
        documents=descriptions
    )  
    
def delete_shows_from_collection(collection: Collection, show_ids: str | list[str]):
    if isinstance(show_ids, str):
        show_ids = [show_ids]
    collection.delete(ids=show_ids)
    
def delete_collection(collection_name: str):
    client.delete_collection(name=collection_name)
    

def count_shows_in_collection(collection: Collection):
    return collection.count()

def peek_first_ten(collection):
    return collection.peek()

def retrieve_by_show_id(collection: Collection, show_ids: str | list[str]):
    if isinstance(show_ids, str):
        show_ids = [show_ids]
    return collection.get(ids=show_ids)


def count_cost(descriptions: list[str]):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens = sum(len(enc.encode(text)) for text in descriptions)
    cost_per_1k_tokens = 0.00002
    
    print(f'Total tokens: {total_tokens}')
    print(f'Cost: {cost_per_1k_tokens * total_tokens/1000}')


def load_netflix_titles(csv_path: Path = NETFLIX_TITLES_PATH, limit: int = 50):
    show_ids = []
    descriptions = []
    metadatas = []

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            description = row["description"].strip()
            if not description:
                continue

            show_ids.append(row["show_id"])
            descriptions.append(description)
            metadatas.append({
                "title": row["title"],
                "type": row["type"],
                "director": row["director"],
                "country": row["country"],
                "date_added": row["date_added"],
                "release_year": int(row["release_year"]),
                "rating": row["rating"],
                "duration": row["duration"],
                "listed_in": row["listed_in"],
            })

            if len(show_ids) >= limit:
                break

    return show_ids, descriptions, metadatas


def main(q):
    try:
        collection = get_or_create_collection("netflix-shows")
        show_ids, descriptions, metadatas = load_netflix_titles(limit=50)
        insert_embeddings(collection, show_ids, descriptions, metadatas)
        print(query_collection(collection, q))
        
        print()
        print(client.list_collections())
        count_cost(descriptions)
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    
if __name__ == "__main__":    
    q = "I want an exciting sci-fi futuristic tech"
    main(q)

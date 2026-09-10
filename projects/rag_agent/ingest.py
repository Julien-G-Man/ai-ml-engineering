import pandas as pd
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from engine import RAGEngine, NAMESPACE

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "youtube_rag_data.csv"


def ingest_indexes():
    youtube_df = pd.read_csv(CSV_PATH)
    rag_engine = RAGEngine()
    index = rag_engine.index
    batch_limit = 100

    total_batches = (len(youtube_df) + batch_limit - 1) // batch_limit

    for batch_num, start in enumerate(range(0, len(youtube_df), batch_limit), start=1):
        batch = youtube_df.iloc[start:start + batch_limit]

        print(f"[{batch_num}/{total_batches}] Preparing {len(batch)} rows", flush=True)

        metadatas = [
            {
                "text_id": row["id"],
                "text": row["text"],
                "title": row["title"],
                "url": row["url"],
                "published": row["published"],
            }
            for _, row in batch.iterrows()
        ]

        texts = batch["text"].tolist()
        ids = [str(uuid4()) for _ in range(len(texts))]

        print(f"[{batch_num}/{total_batches}] Creating embeddings...", flush=True)
        response = rag_engine.create_embeddings(query=texts)

        embeds = [x.embedding for x in response.data]
        vectors = list(zip(ids, embeds, metadatas))

        print(f"[{batch_num}/{total_batches}] Upserting to Pinecone...", flush=True)
        index.upsert(vectors=vectors, namespace=NAMESPACE)

        print(f"[{batch_num}/{total_batches}] Done", flush=True)

    return index.describe_index_stats()


if __name__ == "__main__":
    print(ingest_indexes())

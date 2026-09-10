import pandas as pd
from uuid import uuid4
from pathlib import Path
from engine import v_engine

index = v_engine.index

NAMESPACE = "squad-dataset"

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "squad_dataset.csv"

print(f"Loading CSV from: {CSV_PATH}", flush=True)
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows", flush=True)

batch_limit = 100
total_batches = (len(df) + batch_limit - 1) // batch_limit

def main():
    for batch_num, start in enumerate(range(0, len(df), batch_limit), start=1):
        batch = df.iloc[start:start + batch_limit]

        print(
            f"[{batch_num}/{total_batches}] Preparing {len(batch)} rows "
            f"({start} to {start + len(batch) - 1})",
            flush=True,
        )

        metadatas = [
            {"text_id": row["id"], "text": row["text"], "title": row["title"]}
            for _, row in batch.iterrows()
        ]

        texts = batch["text"].tolist()
        ids = [str(uuid4()) for _ in range(len(texts))]

        print(f"[{batch_num}/{total_batches}] Creating embeddings...", flush=True)
        response = v_engine.create_embeddings(texts)

        embeds = [x.embedding for x in response.data]
        vectors = list(zip(ids, embeds, metadatas))

        print(f"[{batch_num}/{total_batches}] Upserting to Pinecone...", flush=True)
        index.upsert(vectors=vectors, namespace=NAMESPACE)

        print(f"[{batch_num}/{total_batches}] Done", flush=True)

    print("Ingestion complete. Fetching index stats...", flush=True)
    print(index.describe_index_stats())

if __name__ == "__main__":
    main()
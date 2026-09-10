from .engine import v_engine


query = "To whom did virgin Mary allegedly appear in 1858 in Lourdes France?"

query_response = v_engine.create_embeddings(query)
query_emb = query_response.data[0].embedding

retrieved_docs = v_engine.query_vector(
    vector=query_emb,
    top_k=3
)

for result in retrieved_docs['matches']:
    print(f"\n{round(result['score'], 2)}: {result['metadata']['text']}")
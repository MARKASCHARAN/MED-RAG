import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX"))


async def store_embeddings(embedded_data):

    vectors = []

    for i, item in enumerate(embedded_data):

        vectors.append({
            "id": f"doc-{i}",
            "values": item["embedding"],
            "metadata": {
                "url": item["url"],
                "content": item["content"]
            }
        })

    if vectors:
        index.upsert(vectors=vectors)


async def retrieve_from_pinecone(query_embedding):

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    return [
        {
            "url": match["metadata"]["url"],
            "content": match["metadata"]["content"]
        }
        for match in results["matches"]
    ]
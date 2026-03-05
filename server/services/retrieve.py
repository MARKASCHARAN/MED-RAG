import numpy as np
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# async def retrieve_relevant(query, embedded_data):

#     # 🔁 Use Gemini for query embedding
#     query_embed = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=query
#     )

#     query_embedding = query_embed.embeddings[0].values

#     scored = []

#     for item in embedded_data:
#         score = cosine_similarity(query_embedding, item["embedding"])
#         scored.append((score, item))

#     scored.sort(reverse=True, key=lambda x: x[0])

#     top_results = [
#         {
#             "url": item["url"],
#             "content": item["content"]
#         }
#         for score, item in scored[:3]
#     ]

#     return top_results

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
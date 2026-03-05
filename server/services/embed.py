import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def embed_content(clean_data):

    embeddings = []

    for item in clean_data:

        text = item["content"].strip()

        # 🚨 Skip empty content
        if not text:
            continue

        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

        embeddings.append({
            "url": item["url"],
            "content": text,
            "embedding": response.embeddings[0].values
        })

    return embeddings
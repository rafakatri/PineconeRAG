import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("API"))
embed_model = SentenceTransformer('embaas/sentence-transformers-multilingual-e5-base')

index = pc.Index("jurisprudencio")

query = "regime semiaberto"

results = index.query(
    vector= embed_model.encode(query).tolist(),
    top_k=3,
)

print(results)

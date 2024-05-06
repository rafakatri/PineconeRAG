import requests, dotenv, os
from requests.exceptions import HTTPError
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
dotenv.load_dotenv()

def get_json(key, count):
    try:
        response = requests.get(f'http://localhost:3000/api/jurisprudencia?q={key}&count={count}')
        response.raise_for_status()
        data = response.json()
        return data
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    return None


def insert_vector(index, data, embed_model: SentenceTransformer):
    ementa_vec = embed_model.encode(data['ementa'])
    del data['ementa']
    index.upsert(
    vectors=[
    {
      "id": str(int(index.describe_index_stats()['total_vector_count'] + 1)), 
      "values": ementa_vec, 
      "metadata": data
    }
    ])


def insert_vectors(index, json, embed_model):
    for tribunal in json:
        for jurisprudencia in json[tribunal]:
            data = json[tribunal][jurisprudencia]
            data['id'] = jurisprudencia
            insert_vector(index, data, embed_model)

 

pc = Pinecone(api_key=os.getenv("API"))

index = pc.Index("jurisprudencio")

embed_model = SentenceTransformer('embaas/sentence-transformers-multilingual-e5-base')

j = get_json("homicídio", 10)
insert_vectors(index, j, embed_model)


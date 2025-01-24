from app.settings import EMBEDDINGS_MODEL, DB_CONNECTION, COLLECTION_NAME, OLLAMA_URL
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
import logging

def retrieve_chunks(query):
    """
    Retrieves relevant chunks based on the similarity search for the provided query.
    """

    vector_store = PGVector(
    embeddings=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_URL),
    collection_name=COLLECTION_NAME,
    connection=DB_CONNECTION,
    use_jsonb=True,
    )

    results = vector_store.similarity_search(
        query, k=5
    )

    for doc in results:
        logging.info(f"* {doc.page_content} [{doc.metadata}]")

    return results
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from app.pdf_processing import load_and_split_pdfs
from app.settings import EMBEDDINGS_MODEL, DB_CONNECTION, COLLECTION_NAME, OLLAMA_URL
import logging

logging.basicConfig(level=logging.INFO)

def populate_database():

    vector_store = PGVector(
    embeddings=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_URL),
    collection_name=COLLECTION_NAME,
    connection=DB_CONNECTION,
    use_jsonb=True,
    )

    pdf_docs = load_and_split_pdfs()

    # Add the PDF-based documents to the vector store
    try:
        vector_store.add_documents(pdf_docs, ids=[doc.metadata["id"] for doc in pdf_docs])
        logging.info("Documents added successfully")
    except Exception as e:
        logging.info("Error adding documents:", e)

    logging.info(f"Added {len(pdf_docs)} documents to the vector store.")

    return vector_store
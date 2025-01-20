from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os
from app.pdf_processing import load_and_split_pdfs
from app.settings import DB_SETTINGS, EMBEDDING_MODEL, PDF_FOLDER

def initialize_database_with_pgvector():
    """
    Initializes the PGVector database and optionally loads PDF embeddings.
    """
    # Retrieve database settings from settings.py
    connection_string = (
        f"postgresql://{DB_SETTINGS['user']}:{DB_SETTINGS['password']}@"
        f"{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['dbname']}"
    )

    # Initialize embedder and vector store
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector(connection_string=connection_string, embedding_function=embedder)
    logging.info("PGVector initialized successfully.")

    if not os.path.exists(PDF_FOLDER):
        logging.error(f"PDF folder '{PDF_FOLDER}' does not exist.")
        return

    pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        logging.warning("No PDF files found in the specified folder.")
        return

    documents = load_and_split_pdfs(pdf_files)

    # Insert documents and embeddings into the vector store
    vector_store.add_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents] if hasattr(documents[0], "metadata") else None,
    )
    logging.info(f"Inserted {len(documents)} documents into PGVector.")

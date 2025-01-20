from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

def load_and_split_pdfs(file_paths, chunk_size=1000, chunk_overlap=200):
    """
    Loads and splits PDF files into smaller document chunks.
    """
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for file_path in file_paths:
        try:
            logging.info(f"Processing file: {file_path}")
            # Load documents from the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                logging.warning(f"No content found in {file_path}. Skipping.")
                continue

            # Split documents into chunks
            split_docs = text_splitter.split_documents(documents)
            all_documents.extend(split_docs)

            logging.info(f"Successfully processed {len(split_docs)} chunks from {file_path}.")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Total chunks processed: {len(all_documents)}")
    return all_documents

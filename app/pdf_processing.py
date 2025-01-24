from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from app.settings import DATA_PATH


def load_and_split_pdfs(chunk_size=1000, chunk_overlap=200):
    """
    Loads and splits PDF files into smaller document chunks.
    """
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Specify the path to your PDF files
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    for file_path in pdf_files:
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

            # Convert each chunk to a Document object with metadata
            for i, chunk in enumerate(split_docs):
                doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "id": f"{os.path.basename(file_path)}-{i + 1}",
                        "source": file_path,
                        "page": chunk.metadata.get("page", "unknown"),
                    },
                )
                all_documents.append(doc)

            logging.info(f"Successfully processed {len(split_docs)} chunks from {file_path}.")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Total chunks processed: {len(all_documents)}")
    return all_documents

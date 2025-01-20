# CRASHED

This repository contains a system designed to process PDF documents, store embeddings in a PostgreSQL-based vector database (PGVector), and retrieve and respond to user queries using an LLM-based pipeline. The system is containerized with Docker and accessible via a FastAPI interface.

## Folder Structure
```
app/
├── database.py          # Initializes the PGVector database and manages embeddings 
├── embeddings.py        # Handles embedding generation for documents 
├── llm_response.py      # Manages the LLM pipeline for generating query responses 
├── main.py              # FastAPI application to handle queries 
├── pdf_processing.py    # Loads and splits PDF documents into chunks
├── retrieval.py         # Performs similarity searches on stored embeddings 
├── settings.py          # Configurations for database and embeddings 

data/                    # Contains uploaded PDF documents for processing 

docker/                  # Dockerfile and docker-compose.yml for containerized deployment 
├── Dockerfile           
├── docker-compose.yml   

requirements.txt         # Python dependencies 
```


## Query the System

To test the system via the terminal, use the following curl command:
```
curl -X POST "http://localhost:8000/query/" -H "Content-Type: application/json" -d '{"user_query":"<Type_Your_Query_Here>"}'
```

## Configuration

Database and embedding settings can be modified in app/settings.py.



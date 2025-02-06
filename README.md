# CRASHED

This repository contains a RAG system designed to process PDF documents, store embeddings in a PostgreSQL-based vector database (PGVector), and retrieve and respond to user queries using an LLM-based pipeline. The system is containerized with Docker and accessible via a FastAPI interface.

## Starting the System

To start the system, you need to run the command located below from the `docker` folder.

```
docker compose up --build
```

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

## GPU Docker Configuration

To modify the `docker-compose` file for specifying which GPU will be utilized for running the application, you need to adjust the configuration based on your system's GPU. 

Currently, the build is set up to use an AMD graphics card. If you want to switch to using an NVIDIA GPU, follow the instructions provided on the [Ollama Docker Hub page](https://hub.docker.com/r/ollama/ollama). 

Refer to the section on configuring the application for NVIDIA GPUs and update the `docker-compose` file accordingly to ensure proper GPU integration.


## Configuration

Database and embedding settings can be modified in app/settings.py.



DB_SETTINGS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
    "host": "db",
    "port": 5432
}

PDF_FOLDER = "/app/data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Table name for embeddings
EMBEDDINGS_TABLE = "langchain_pg_embedding"

# Embedding model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default similarity search top-k results
DEFAULT_K = 5
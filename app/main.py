import logging
from fastapi import FastAPI
from pydantic import BaseModel
from app.settings import PDF_FOLDER
from app.retrieval import retrieve_relevant_chunks
from app.database import initialize_database_with_pgvector
from app.llm_response import generate_response

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Define the request schema
class QueryRequest(BaseModel):
    user_query: str

initialize_database_with_pgvector()

@app.post("/query/")
def query_system(request: QueryRequest):
    """
    API endpoint to handle user queries.
    """
    user_query = request.user_query
    logging.info(f"Received query: {user_query}")
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(user_query, k=5)
    logging.debug(f"Retrieved chunks: {retrieved_chunks}")

    # Pass the extracted text to generate_response
    # response = generate_response(user_query, [chunk[2] for chunk in retrieved_chunks]) # This line is commented out because for this model maximum token length is 1024
    response = generate_response(user_query, retrieved_chunks[0][2])

    return {"response": response}

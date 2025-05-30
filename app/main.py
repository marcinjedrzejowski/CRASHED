from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from app.llm_response import generate_response
from app.database import populate_database

app = FastAPI()

# Define the request schema
class QueryRequest(BaseModel):
    user_query: str

populate_database()

@app.post("/query/")
async def query_llm(request: QueryRequest):
    """
    API endpoint to handle user queries.
    """
    user_query = request.user_query

    try:
        # Use `await` to correctly call the async function
        response = await generate_response(user_query)
        return {"query": user_query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

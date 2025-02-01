from langchain_ollama import ChatOllama
from app.retrieval import retrieve_chunks
from app.settings import OLLAMA_URL
import asyncio

async def generate_response(query):
    """
    Asynchronously generates a response to the query using a hybrid retrieval approach.
    """
    context = await retrieve_chunks(query)  # Retrieve relevant chunks

    if not context:
        return "I couldn't find relevant information."

    # Format the retrieved context
    context_text = "\n".join([f"- {doc[1]} (id: {doc[0]})" for doc in context])

    # Define the LLM model
    llm = ChatOllama(
        model="llama3",
        temperature=0,
        base_url=OLLAMA_URL
    )

    # Create a prompt with the retrieved context
    prompt = f"""
    Use the following context to answer the user's question:

    Context:
    {context}

    Question:
    {query}

    Provide a detailed response only in Spanish based on the information above. If you don't know the answer, don't make it up - state that You couldn't find the answer.
    Stick to the facts, be precise and concise. Don't add any redundant information. Don't write about hypothetical situations.
    """
    
    # Generate a response
    response = await asyncio.to_thread(llm.invoke, prompt)
    
    return response.content

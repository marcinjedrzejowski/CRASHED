from langchain_ollama import ChatOllama
from app.retrieval import retrieve_chunks
from app.settings import OLLAMA_URL


def generate_response(query):
    """
    Generates a response to the query based on the provided context using a text-generation model.
    """

    context = retrieve_chunks(query)

    llm = ChatOllama(
        model="llama3",
        temperature=0, 
        base_url=OLLAMA_URL
    )

    # Combine the retrieved documents into a single context string
    context = "\n".join(
        [f"- {doc.page_content} (metadata: {doc.metadata})" for doc in context]
    )

    # Create a prompt with the context
    prompt = f"""
    Use the following context to answer the user's question:

    Context:
    {context}

    Question:
    {query}

    Provide a detailed response based on the information above. If you don't know the answer, don't make it up - state that You couldn't find the answer. Stick to the facts, be precise and concise.
    """

    # Generate a response from ChatOllama
    response = llm.invoke(prompt)

    # Print the response
    print("\nResponse from LLM:")
    print(response.content)

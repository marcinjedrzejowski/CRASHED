from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import logging

def clean_context(raw_context):
    """
    Cleans and formats the provided context by removing newlines and stripping whitespace.
    """
    if isinstance(raw_context, tuple) and len(raw_context) > 2:
        return raw_context[2].replace("\n", " ").strip()
    if isinstance(raw_context, str):
        return raw_context.replace("\n", " ").strip()
    return str(raw_context)

def initialize_llm_pipeline(model_name="distilgpt2", max_tokens=200):
    """
    Initializes the HuggingFace pipeline for text generation.
    """
    generator_pipeline = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=max_tokens
    )
    return HuggingFacePipeline(pipeline=generator_pipeline)

def generate_response(query, raw_context, model_name="distilgpt2", max_tokens=200):
    """
    Generates a response to the query based on the provided context using a text-generation model.
    """
    context = clean_context(raw_context)
    llm = initialize_llm_pipeline(model_name=model_name, max_tokens=max_tokens)

    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer the question based on the context above. Provide a concise and specific response.
    """
    try:
        response = llm.invoke(prompt)
        return response.strip() if response.strip() else "No meaningful response generated."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"An error occurred: {e}"

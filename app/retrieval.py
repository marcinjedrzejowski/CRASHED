from app.settings import EMBEDDINGS_MODEL, DB_CONNECTION, COLLECTION_NAME, OLLAMA_URL
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
import itertools
import psycopg
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

def get_vector_store():
    """
    Initializes the PGVector store for vector search.
    """
    return PGVector(
        embeddings=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_URL),
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True,
    )

async def bm25_search(conn, query, k=100):
    """
    Performs BM25 keyword search using PostgreSQL full-text search.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id, document FROM {COLLECTION_NAME}, plainto_tsquery('spanish', %s) query
            WHERE to_tsvector('spanish', document) @@ query
            ORDER BY ts_rank_cd(to_tsvector('spanish', document), query) DESC
            LIMIT %s
            """,
            (query, k),
        )
        return await cur.fetchall()

def semantic_search(query, k=100):
    """
    Performs dense vector search using PGVector's built-in similarity search.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return [(doc.metadata.get("id", "unknown"), doc.page_content) for doc in results]

def rerank(query, results, top_k=5):
    """
    Re-rank results using a cross-encoder model and return top K.
    """
    results = list(set(results))  # Deduplicate
    encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = encoder.predict([(query, item[1]) for item in results])
    ranked_results = [v for _, v in sorted(zip(scores, results), reverse=True)]
    return ranked_results[:top_k]

async def retrieve_chunks(query):
    """
    Hybrid retrieval combining BM25 and PGVector with reranking.
    """
    conn = await psycopg.AsyncConnection.connect(DB_CONNECTION, autocommit=True)
    
    bm25_results, semantic_results = await asyncio.gather(bm25_search(conn, query), asyncio.to_thread(semantic_search, query))
    
    results = list(itertools.chain(bm25_results, semantic_results))  # Merge results
    results = rerank(query, results, top_k=5)  # Rerank and return top 5

    # Print top 5 retrieved chunks
    logging.info("Top 5 Retrieved Chunks:")
    for idx, chunk in enumerate(results, start=1):
        logging.info(f"{idx}. {chunk[1]} (ID: {chunk[0]})")

    return results
import logging
import psycopg2
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from app.settings import DB_SETTINGS, EMBEDDINGS_TABLE, EMBEDDING_MODEL, DEFAULT_K

embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

CONNECTION_STRING = (
    f"postgresql://{DB_SETTINGS['user']}:{DB_SETTINGS['password']}@"
    f"{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['dbname']}"
)

vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embedder
)

def fetch_first_embedding():
    """
    Fetches the first embedding from the vector store.
    """

    try:
        with psycopg2.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT embedding FROM {EMBEDDINGS_TABLE} LIMIT 1;")
                result = cursor.fetchone()
                return result[0] if result else None
    except Exception as e:
        logging.error(f"Error fetching first embedding: {e}")
        return None

def custom_similarity_search(embedding, table_name=EMBEDDINGS_TABLE, k=DEFAULT_K):
    """
    Perform similarity search directly using PostgreSQL.
    """

    try:
        with psycopg2.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                # Convert the embedding into a PostgreSQL array format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                # SQL query for similarity search
                query = f"""
                SELECT custom_id, 1 - (embedding <-> '{embedding_str}'::vector) AS similarity, document
                FROM {table_name}
                ORDER BY similarity DESC
                LIMIT {k};
                """
                logging.info("Executing similarity search query.")
                cursor.execute(query)
                results = cursor.fetchall()

                # Format results
                return [(row[0], row[1], row[2]) for row in results]

    except psycopg2.Error as db_error:
        logging.error(f"Database error during similarity search: {db_error}")
        return []

    except Exception as e:
        logging.error(f"Unexpected error during similarity search: {e}")
        return []

def retrieve_relevant_chunks(query, k=DEFAULT_K):
    """
    Retrieves relevant chunks based on the similarity search for the provided query.
    """

    try:
        # Generate the query embedding
        query_embedding = embedder.embed_query(query)
        logging.debug(f"Generated query embedding: {query_embedding}")

        # Fetch the first embedding for debug/logging purposes
        '''first_embedding = fetch_first_embedding()
        logging.debug(f"First embedding retrieved: {first_embedding}")'''

        results = custom_similarity_search(
            embedding=query_embedding,
            table_name=EMBEDDINGS_TABLE,
            k=k
        )

        # Format results for readability
        formatted_results = []
        for result in results:
            logging.debug(f"Raw result: {result}")
            try:
                custom_id, similarity, document_chunk = result
                formatted_results.append((custom_id, similarity, document_chunk))
            except ValueError as unpacking_error:
                logging.error(f"Error unpacking result tuple {result}: {unpacking_error}")
                continue

        logging.info(f"Retrieved {len(formatted_results)} chunks.")
        return formatted_results

    except Exception as e:
        logging.error(f"Error retrieving relevant chunks: {e}")
        return []

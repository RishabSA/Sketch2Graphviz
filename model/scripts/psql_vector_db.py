import numpy as np
from tqdm.auto import tqdm
import psycopg2
from pgvector.psycopg2 import register_vector


def store_embeddings_in_db(
    embedding_data: list[tuple[str, np.ndarray]],
    dbname: str = "sketch2graphvizdb",
    table_name: str = "graphviz_embeddings",
    embedding_dim: int = 4096,
) -> bool:
    try:
        conn = psycopg2.connect(f"dbname={dbname}")
        register_vector(conn)
        cur = conn.cursor()

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                code TEXT NOT NULL,
                embedding VECTOR({embedding_dim}) NOT NULL
            );
            """
        )

        for graphviz_code, embedding_vector in tqdm(
            embedding_data, desc="Storing embedding data"
        ):
            embedding_vector = np.asarray(embedding_vector, dtype="float32").tolist()

            cur.execute(
                f"""
                INSERT INTO {table_name} (code, embedding)
                VALUES (%s, %s);
                """,
                (graphviz_code, embedding_vector),
            )

        conn.commit()

        cur.close()
        conn.close()

        print(
            f"Successfully inserted {len(embedding_data)} embeddings and codes into the PostgreSQL DB"
        )

        return True
    except Exception as e:
        print(
            f"An unexpected error occurred while attempting to store embeddings in the postgreSQL DB: {e}"
        )

        return False


def get_top_k_similar_vectors_from_db(
    embedding_vector: np.ndarray,
    top_K: int = 5,
    dbname: str = "sketch2graphvizdb",
    table_name: str = "graphviz_embeddings",
) -> list[tuple[int, str, float]]:
    try:
        conn = psycopg2.connect(f"dbname={dbname}")
        register_vector(conn)
        cur = conn.cursor()

        embedding_vector = np.asarray(embedding_vector, dtype="float32")

        cur.execute(
            f"""
            SELECT id, code, embedding <-> %s AS distance
            FROM {table_name}
            ORDER BY embedding <-> %s
            LIMIT %s;
            """,
            (embedding_vector, embedding_vector, top_K),
        )

        results = cur.fetchall()

        cur.close()
        conn.close()

        print(f"Successfully fetched {len(results)} results from the PostgreSQL DB")

        return results
    except Exception as e:
        print(
            f"An unexpected error occurred while attempting to fetch top-K similar embeddings from the postgreSQL DB: {e}"
        )

        return []

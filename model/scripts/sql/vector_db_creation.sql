CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS graphviz_embeddings (
    id SERIAL PRIMARY KEY,
    code TEXT NOT NULL,
    embedding VECTOR(4096) NOT NULL
);

-- Repeat for each item
INSERT INTO graphviz_embeddings (code, embedding) VALUES ('graph {...}', '[]');

-- Get top-K closest embedding vectors
SELECT id, code, embedding <-> [] AS distance FROM graphviz_embeddings ORDER BY embedding <-> [] LIMIT 5;
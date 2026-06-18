from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents , save_chunks
from src.retrieval.vector_store import add_documents_to_vectorstore


def build_vector_index() -> None:
    documents = load_documents("data/raw")
    cleaned = clean_documents(documents)
    chunks = chunk_documents(cleaned)

    save_chunks(chunks)

    add_documents_to_vectorstore(chunks)

    print("Vector index built successfully.")


if __name__ == "__main__":
    build_vector_index()
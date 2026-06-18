from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone , ServerlessSpec
import os
from langchain_core.documents import Document
from src.retrieval.embeddings import get_embedding_model
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME='adaptive-graphrag'

def get_vector_store()->PineconeVectorStore:
    """Connect to the existing Pinecone index and return a vector store object."""
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embedding_model(),
        pinecone_api_key=api_key,
    )

def add_documents_to_vectorstore(chunks) -> None:
    vector_store = get_vector_store()

    ids = [
        str(chunk.metadata.get("chunk_id", index))
        for index, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )
    print("Added documents to Pinecone database")
    return vector_store

from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_documents
from src.retrieval.embeddings import get_embedding_model


def main() -> None:
    documents = load_documents("data/raw")
    cleaned_documents = clean_documents(documents)
    chunks = chunk_documents(cleaned_documents)
    vector_store = add_documents_to_vectorstore(chunks)
    print(vector_store)


if __name__ == "__main__":
    main()
    
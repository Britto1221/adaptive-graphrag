from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone , ServerlessSpec
import os
from langchain_core.documents import Document
from src.retrieval.embeddings import get_embedding_model
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME='adaptive-graphrag'

def create_pinecone_index()->None:
    """Create a pinecone index to store the embeddings"""
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from the environment.")
    
    pc  = Pinecone(api_key=api_key)

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384 ,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )

def add_documents_to_vectorstore(documents:list[Document])->PineconeVectorStore:
    
    embeddings = get_embedding_model()

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    vector_store.add_documents(
        documents=documents
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
    create_pinecone_index()
    vector_store = add_documents_to_vectorstore(chunks)
    print(vector_store)


if __name__ == "__main__":
    main()
    
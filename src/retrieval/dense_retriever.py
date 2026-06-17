from langchain_pinecone import PineconeVectorStore
import os
from src.retrieval.embeddings import get_embedding_model
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document

INDEX_NAME='adaptive-graphrag'

def get_similar_docs(query:str,k:int=5)->list[Document]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embedding_model(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs ={'k':k})
    return retriever.invoke(query)



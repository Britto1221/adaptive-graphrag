from langchain_community.retrievers import BM25Retriever
import os
from dotenv import load_dotenv
load_dotenv()
import re
from langchain_core.documents import Document


def preprocess_text(text):
    """
    Convert text into lowercase word tokens for BM25.

    This keeps letters, numbers, company names, and common abbreviations
    while removing most punctuation.
    """
    return re.findall(r"\b\w+\b", text.lower())

def bm25_retriever(query:str,chunks:list,k:int=5)->list[Document]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")
    
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k

    return retriever.invoke(query)
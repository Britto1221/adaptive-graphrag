from src.retrieval.bm25_retriever import bm25_retriever
from src.retrieval.dense_retriever import get_similar_docs
from langchain_core.documents import Document


def hybrid_retriever(query:str,chunks:list[Document])->list[Document]:
    final_retrieved_docs:list[Document] = []
    final_retrieved_docs.extend(bm25_retriever(query,chunks))
    final_retrieved_docs.extend(get_similar_docs(query))

    seen = set()
    unique_docs = []
    for doc in final_retrieved_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs


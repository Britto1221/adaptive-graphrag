from src.ingestion.chunker import load_saved_chunks
from src.retrieval.reranker import rerank_documents
from src.retrieval.hybrid_search import hybrid_retriever
from src.generation.answer_generator import generate_answer

CHUNKS = load_saved_chunks() 

def vector_pipeline(query:str):
    chunks = hybrid_retriever(query, CHUNKS)
    reranked = rerank_documents(query, chunks)
    answer = generate_answer(query, reranked)
    return answer

if __name__ == "__main__":
    result = vector_pipeline("How is Elon Musk connected to Jeff Bezos?")
    print(result)
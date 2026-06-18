from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.reranker import rerank_documents
from src.retrieval.hybrid_search import hybrid_retriever
from src.generation.answer_generator import generate_answer

print("Loading Documents......")
documents = load_documents("data/raw")
print("Documents Loaded")
cleaned = clean_documents(documents)
CHUNKS = chunk_documents(cleaned) 

def vector_pipeline(query:str):
    chunks = hybrid_retriever(query, CHUNKS)
    reranked = rerank_documents(query, chunks)
    answer = generate_answer(query, reranked)
    return answer

if __name__ == "__main__":
    result = vector_pipeline("How is Elon Musk connected to Jeff Bezos?")
    print(result)
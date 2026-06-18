from src.ingestion.chunker import load_saved_chunks
from src.retrieval.reranker import rerank_documents
from src.retrieval.hybrid_search import hybrid_retriever
from src.generation.answer_generator import generate_hybrid_answer
from src.graph.graph_retriever import get_graph_answer

CHUNKS = load_saved_chunks() 

def hybrid_pipeline(query:str)->str:
    chunks = hybrid_retriever(query, CHUNKS)
    reranked = rerank_documents(query, chunks)
    graph = get_graph_answer(query)
    return generate_hybrid_answer(
        query,
        reranked,
        graph,
    )
    

if __name__ == "__main__":
    result = hybrid_pipeline("How is Elon Musk connected to Jeff Bezos?")
    print("\nFINAL HYBRID ANSWER:")
    print(result)
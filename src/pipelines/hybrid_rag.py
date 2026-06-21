from src.ingestion.chunker import load_saved_chunks
from src.retrieval.reranker import rerank_documents
from src.retrieval.hybrid_search import hybrid_retriever
from src.generation.answer_generator import generate_hybrid_answer
from src.graph.graph_retriever import ask_graph
from src.evaluation.evidence_formatter import (
    format_vector_evidence,
    format_graph_evidence,
)
from langchain_core.tracers.context import tracing_v2_enabled
CHUNKS = load_saved_chunks() 

def hybrid_pipeline(query:str,embeddings,return_details: bool = False):
    embeddings=embeddings
    chunks = hybrid_retriever(query, CHUNKS)
    reranked = rerank_documents(query, chunks)
    graph_response = ask_graph(query)

    vector_evidence = format_vector_evidence(reranked)
    graph_evidence = format_graph_evidence(graph_response)
    answer = generate_hybrid_answer(
        query,
        reranked,
        graph_evidence,
    )

    evidence_parts = []

    if vector_evidence != "none":
        evidence_parts.append(vector_evidence)

    if graph_evidence != "none":
        evidence_parts.append(graph_evidence)

    combined_evidence = "\n\n".join(evidence_parts) if evidence_parts else "none"

    if return_details:
        return {
            "pipeline": "hybrid_rag",
            "answer": answer,
            "evidence": combined_evidence,
            "vector_evidence": vector_evidence,
            "graph_evidence": graph_evidence,
            "raw_graph_response": graph_response,
            "retrieved_chunks_count": len(chunks),
            "reranked_chunks_count": len(reranked),
        }
    return answer
    

if __name__ == "__main__":
    with tracing_v2_enabled():
        result = hybrid_pipeline(
            "How is Elon Musk connected to Jeff Bezos, and what evidence explains their business competition?",
            return_details=True,
        )

    print("\nFINAL HYBRID ANSWER:")
    print(result["answer"])

    print("\nVECTOR EVIDENCE:")
    print(result["vector_evidence"])

    print("\nGRAPH EVIDENCE:")
    print(result["graph_evidence"])
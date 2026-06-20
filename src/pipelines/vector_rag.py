from src.ingestion.chunker import load_saved_chunks
from src.retrieval.reranker import rerank_documents
from src.retrieval.hybrid_search import hybrid_retriever
from src.generation.answer_generator import generate_answer
from src.evaluation.evidence_formatter import format_vector_evidence


CHUNKS = load_saved_chunks()


def vector_pipeline(query: str, return_details: bool = True):
    chunks = hybrid_retriever(query, CHUNKS)

    reranked = rerank_documents(
        query=query,
        documents=chunks,
        top_k=5,
    )

    vector_evidence = format_vector_evidence(reranked)

    answer = generate_answer(query, reranked)

    if return_details:
        return {
            "pipeline": "vector_rag",
            "answer": answer,
            "evidence": vector_evidence,
            "vector_evidence": vector_evidence,
            "graph_evidence": "none",
            "evidence": vector_evidence,
            "generated_cypher": "",
            "graph_context_count": 0,
        }

    return answer


if __name__ == "__main__":
    result = vector_pipeline(
        "How is Elon Musk connected to Jeff Bezos?",
        return_details=True,
    )

    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nVECTOR EVIDENCE:")
    print(result["vector_evidence"])
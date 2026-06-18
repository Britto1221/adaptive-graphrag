from src.graph.graph_retriever import ask_graph
from src.evaluation.evidence_formatter import format_graph_evidence


def graph_pipeline(query: str, return_details: bool = False):
    graph_response = ask_graph(query)

    answer = graph_response.get("result", "")
    graph_evidence = format_graph_evidence(graph_response)

    if return_details:
        return {
            "pipeline": "graph_rag",
            "answer": answer,
            "evidence": graph_evidence,
            "vector_evidence": "none",
            "graph_evidence": graph_evidence,
            "raw_graph_response": graph_response,
        }

    return answer

if __name__ == "__main__":
    response = graph_pipeline("How is Mark Zukerberg related to Mukesh Ambani?")
    print(response)
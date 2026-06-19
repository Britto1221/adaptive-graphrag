from src.graph.graph_retriever import ask_graph
from src.evaluation.evidence_formatter import format_graph_evidence
from langchain_core.tracers.context import tracing_v2_enabled



def graph_pipeline(query: str, return_details: bool = True):
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
    queries = [
        "Who invested in Jio Platforms?"
    ]
    for query in queries:
        response = graph_pipeline(query)
        print(response)
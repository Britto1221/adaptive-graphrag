def format_vector_evidence(documents, max_chars: int = 2500) -> str:
    if not documents:
        return "none"

    source = documents[0].metadata.get("source", "unknown")

    evidence_parts = [
        f"Document: {source}",
        "All chunks below were retrieved from this document."
    ]

    for index, document in enumerate(documents, start=1):
        chunk_id = document.metadata.get("chunk_id", f"chunk_{index}")
        text = document.page_content.strip().replace("\n", " ")

        evidence_parts.append(
            f"[Chunk {index}]\n"
            f"chunk_id: {chunk_id}\n"
            f"text: {text}"
        )

    evidence = "\n\n".join(evidence_parts)

    return evidence[:max_chars]

def format_graph_evidence(
    graph_response: dict,
    max_chars: int = 2500,
) -> str:
    intermediate_steps = graph_response.get("intermediate_steps", [])

    graph_context = None

    for step in intermediate_steps:
        if isinstance(step, dict) and "context" in step:
            graph_context = step["context"]

    if not graph_context:
        return "none"

    evidence = (
        "Graph database: Neo4j\n"
        "All graph evidence below was retrieved from the Neo4j knowledge graph.\n\n"
        f"Graph result:\n{graph_context}"
    )

    return evidence[:max_chars]
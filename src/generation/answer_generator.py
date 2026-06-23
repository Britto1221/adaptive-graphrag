from src.generation.prompt_templates import get_rag_prompt
from src.graph.neo4j_client import neo4j_manager
from src.models.model_registry import get_model
from langchain_neo4j import GraphCypherQAChain
from src.generation.prompt_templates import get_hybrid_rag_prompt

answer_model ="openai"

llm = get_model(answer_model)

llm1 = get_model("openai")
def format_context(chunks):
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        score = chunk.metadata.get("reranker_score", "unknown")

        context_parts.append(
            f"Source {index}\n"
            f"File: {source}\n"
            f"Reranker score: {score}\n"
            f"Content:\n{chunk.page_content}"
        )
    return "\n\n".join(context_parts)

def generate_answer(query, chunks):
    context = format_context(chunks)

    prompt = get_rag_prompt()

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": query,
            "context": context,
        }
    )

    return response.content

def graph_answer(query,cypher_prompt,qa_prompt):
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm1,
        qa_llm=llm,
        graph=neo4j_manager(),
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )
    return chain.invoke({'query':query})

def generate_hybrid_answer(
    query: str,
    vector_data,
    graph_data: str,
) -> str:
    vector_context = format_context(vector_data)
    if not vector_context.strip():
        vector_context = "No vector context retrieved."
    
    if not graph_data or not graph_data.strip():
        graph_context = "No graph context retrieved."
    else:
        graph_context = graph_data

    prompt = get_hybrid_rag_prompt()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": query,
            "vector_context": vector_context,
            "graph_context": graph_context,
        }
    )

    return response.content
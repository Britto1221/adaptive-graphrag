from src.models.groq_provider import groq_provider
from pydantic import BaseModel
from typing import Literal

llm = groq_provider()

class RouteDecision(BaseModel):
    pipeline: Literal["vector_rag", "graph_rag", "hybrid_rag"]

structured_llm = llm.with_structured_output(RouteDecision)

def route_query(query: str) -> str:
    prompt = f"""
        You are a query router for a RAG system with three pipelines:

        1. vector_rag — for factual, definitional, descriptive questions
        Example: "What is Elon Musk's net worth?"

        2. graph_rag — for relationship, connection, investment questions
        Example: "How is Mark Zuckerberg connected to Mukesh Ambani?"

        3. hybrid_rag — for complex questions needing both facts and relationships
        Example: "Tell me about the company Mark Zuckerberg invested in"

        Return ONLY one of these exact strings:
        vector_rag
        graph_rag
        hybrid_rag

        Query: {query}
        """
    result = structured_llm.invoke(prompt)
    return result.pipeline
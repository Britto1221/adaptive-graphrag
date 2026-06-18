from langchain_core.prompts import PromptTemplate

def get_rag_prompt():
    return PromptTemplate.from_template(
        """
You are a RAG assistant.

Answer the question using only the context below.
Do not use outside knowledge.
If the context does not contain the answer, say:
"The retrieved context does not contain enough information."

Question:
{question}

Context:
{context}

Answer:
"""
    )

def get_hybrid_rag_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """
You are a Hybrid RAG assistant.

You are given two types of evidence:

1. Vector Context:
This comes from retrieved text chunks.

2. Graph Context:
This comes from Neo4j graph results.

Rules:
- Use only the provided Vector Context and Graph Context.
- Do not use outside knowledge.
- Prefer Graph Context for exact relationships and entity connections.
- Prefer Vector Context for descriptions and background details.
- If both contexts support the answer, combine them.
- If neither context contains the answer, say:
"The retrieved context does not contain enough information."

At the end of every answer, include:

Evidence used: vector, graph, both, or none
Evidence summary: short summary of the supporting evidence, or none

Question:
{question}

Vector Context:
{vector_context}

Graph Context:
{graph_context}

Answer:
"""
    )
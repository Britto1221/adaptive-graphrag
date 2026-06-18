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
This comes from text chunks retrieved from documents.

2. Graph Context:
This comes from Neo4j graph queries and structured graph relationships.

Rules:
- Use only the provided Vector Context and Graph Context.
- Do not use outside knowledge.
- Prefer Graph Context for exact relationships, entities, and connections.
- Prefer Vector Context for explanations, background, and descriptive details.
- If both contexts agree, combine them.
- If one context is missing, use the available context.
- If neither context contains the answer, say:
"The retrieved context does not contain enough information."
- At the end, include:
Evidence used: Graph evidence, Vector evidence, or Both.

Question:
{question}

Vector Context:
{vector_context}

Graph Context:
{graph_context}

Hybrid Answer:
"""
    )
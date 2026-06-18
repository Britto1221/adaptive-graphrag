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
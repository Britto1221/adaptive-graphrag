import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.generation.prompt_templates import get_rag_prompt
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from the .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


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
import os
from dotenv import load_dotenv
load_dotenv()
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from src.graph.neo4j_client import neo4j_manager
from langchain_neo4j import GraphCypherQAChain

llm = ChatOpenAI(
        model = 'gpt-4o-mini',
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Cypher query generator.

    Schema:
    {schema}

    Rules:
    - Return only Cypher.
    - Do not use markdown code fences.
    - Use only labels, properties, and relationships in the schema.
    - Generate read-only queries only.
    - Never use CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE, DROP, or CALL.
    - Add LIMIT 10 unless aggregation is requested.
    - Richest-person nodes have a non-null rank property.
    - For wealthy-person questions, use WHERE person.rank IS NOT NULL when needed.
    - The property relationshipTypes is only a list property, not actual graph relationships.
    - Person-to-company links use (:Person)-[:ASSOCIATED_WITH]->(:Company).
    - Business fact sentences use (:Person)-[:HAS_RELATIONSHIP_FACT]->(:RelationshipFact).
    - For questions asking how two people are connected, search:
    - For "How is person A connected to person B?" questions, first try to find paths through associated companies:
    (:Person)-[:ASSOCIATED_WITH]->(:Company)-[*1..3]-(:Company)<-[:ASSOCIATED_WITH]-(:Person)
    - Prefer actual graph paths over RelationshipFact text.
    - Only use RelationshipFact text if no graph path exists.
    1. relationship fact descriptions,
    2. associated companies,
    3. business entities mentioned inside RelationshipFact descriptions.
    - Do not require both person names to appear in the same RelationshipFact.
    - When using UNION, every branch must return exactly the same column names.
    - For UNION queries, use these common aliases:
    person, evidence_type, evidence.

    Question:
    {question}
"""


QA_TEMPLATE = """
You are a strict graph question-answering assistant.

Question:
{question}

Database Results:
{context}

Rules:
- Answer using only the database results.
- Do not add outside knowledge.
- Do not explain facts that are not present in the database results.
- If the result contains relationship facts, summarize only those facts.
- If the result contains node properties, mention only those properties.
- If the database results are empty, say:
  "The graph does not contain enough information."

Answer:
"""

cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=QA_TEMPLATE
    )

@lru_cache(maxsize=1)
def get_graph_chain():
    graph = neo4j_manager()
    graph.refresh_schema()

    return GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
    )


def ask_graph(query: str) -> dict:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chain = get_graph_chain()

    return chain.invoke(
        {
            "query": query,
        }
    )

def get_graph_answer(query: str) -> str:
    response = ask_graph(query)

    if isinstance(response, dict):
        return response.get("result", "")

    return str(response)

if __name__ == "__main__":
    response = ask_graph(
        "Who are the five richest people in the database?"
    )

    print("\nFINAL ANSWER:")
    print(response["result"])

    print("\nINTERMEDIATE STEPS:")
    print(response.get("intermediate_steps"))
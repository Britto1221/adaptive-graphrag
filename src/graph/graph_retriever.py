import os
from dotenv import load_dotenv
load_dotenv()
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from src.graph.neo4j_client import neo4j_manager
from langchain_neo4j import GraphCypherQAChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatOpenAI(
        model = 'gpt-4o-mini',
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

llm1 = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0,
    api_key="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"
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator.

Schema:
{schema}

Generate one valid read-only Cypher query for the question.

Rules:
- Return only Cypher.
- Do not use markdown code fences.
- Use only labels, properties, and relationships from the schema.
- Never use CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE, DROP, or CALL.
- Add LIMIT 10 at the end unless aggregation is requested.
- If using UNION, put LIMIT 10 only once at the very end.

Data rules:
- Richest-person nodes have a non-null rank property.
- Smaller rank means richer. Rank 1 is richer than rank 30.
- For richest/top-ranked questions, use ORDER BY person.rank ASC.
- For lowest-ranked or least-rich among ranked people, use ORDER BY person.rank DESC.
- Use WHERE person.rank IS NOT NULL for richest/wealthy-person questions when needed.
- Person-to-company links use (:Person)-[:ASSOCIATED_WITH]->(:Company).
- Business fact text uses (:Person)-[:HAS_RELATIONSHIP_FACT]->(:RelationshipFact).
- relationshipTypes is only metadata. Do not use it as proof of actual relationships.

Return rules:
- Never return full nodes like p AS person.
- Return only specific properties like p.name, c.name, rf.description, type(r), or maps.
- Use these aliases in every query:
  person, evidence_type, evidence
- Do not use type(r) unless r is explicitly declared in MATCH.
- Do not use type(r) for variable-length paths like [*1..3].

Path rules:
- For path queries, always name the path as path:
  MATCH path = (...)
- For path evidence, return a map with both path_nodes and relationship_types:
  {{
    path_nodes: [n IN nodes(path) | coalesce(n.name, n.title)],
    relationship_types: [rel IN relationships(path) | type(rel)]
  }} AS evidence
- Never return only path nodes as evidence.

For "How is person A connected to person B?" questions:
- First try actual graph paths through associated companies:
  (:Person)-[:ASSOCIATED_WITH]->(:Company)-[*1..3]-(:Company)<-[:ASSOCIATED_WITH]-(:Person)
- Also try person-to-person business paths:
  (:Person)-[:BUSINESS_RELATED_TO*1..3]-(:Person)
- Prefer actual graph paths over RelationshipFact text.
- Use RelationshipFact text only as fallback.
- For RelationshipFact fallback, search each person's fact descriptions using WHERE rf.description CONTAINS the other person's name.
- Do not assume the same RelationshipFact node is connected to both people.

UNION rules:
- Every UNION branch must return exactly:
  person, evidence_type, evidence
- Do not return full nodes in any UNION branch.
- Put LIMIT 10 only once at the very end.
- Every UNION branch must return exactly the same column names in the same meaning:
  person, evidence_type, evidence
- Even path branches must return person, evidence_type, evidence.
- Do not return only evidence_type and evidence in one branch.
- For path branches, use:
  'Person A' AS person,
  'graph_path' AS evidence_type,
  {{
    path_nodes: [n IN nodes(path) | coalesce(n.name, n.title)],
    relationship_types: [rel IN relationships(path) | type(rel)]
  }} AS evidence

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
- Answer using only the Database Results.
- Do not use outside knowledge.
- If Database Results is empty, say:
  "The graph does not contain enough information."
- If Database Results is not empty, answer using only those results.
- Do not use relationshipTypes as proof of actual relationships.
- Do not invent relationship types that are not present in Database Results.
- If the result contains path_nodes and relationship_types, explain the connection naturally.
- Do not print raw Cypher maps like {{path_nodes: ..., relationship_types: ...}} unless the user asks for raw evidence.
- Do not include "AS evidence" in the final answer.
- Prefer simple wording such as:
  "A direct BUSINESS_RELATED_TO path connects A and B."
  "Another path connects A to B through C."
If the result contains path_nodes and relationship_types, explain the connection naturally and mention the relationship type when useful.
At the end of every answer, include exactly:

Evidence used: graph or none
Evidence summary: summarize only the graph evidence explicitly present in the Database Results. If no supporting graph evidence is present, write none.

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
        qa_llm=llm1,
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
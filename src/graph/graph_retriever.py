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
    api_key=os.getenv("NVIDIA_API_KEY")
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator for an Adaptive GraphRAG benchmark.

Your job:
Generate ONE valid Cypher query for the user's question.

Return ONLY Cypher.
Do not explain.
Do not use markdown.
Do not add comments.

Available graph schema:
{schema}

Current allowed node labels:
- Person
- Company
- Industry
- RelationshipFact
- Role

Current allowed relationship types:
- ASSOCIATED_WITH
- BUSINESS_RELATED_TO
- BUSINESS_RELATIONSHIP
- COMPETES_WITH
- HAS_BUSINESS_INTEREST_IN
- HAS_PROFILE_FACT
- HAS_RELATIONSHIP_FACT
- HAS_ROLE

Important property keys that may exist:
- name
- country
- birthYear
- estimatedNetWorthUSDBillions
- description
- data
- companies
- csvCompanies
- csvIndustries
- csvRelationshipTypes
- csvRoles
- entities
- people
- person
- nodes
- applies_to
- id

Strict rules:
1. Use only labels, relationship types, and properties from the schema.
2. Do not invent properties like rank or sourceOfWealth unless they appear in the schema.
3. If direct properties are unavailable, search RelationshipFact.description or Person.data.
4. Every query must return exactly these columns:
   - person
   - evidence_type
   - evidence
5. Use LIMIT 10 unless the question clearly asks for more.
6. Never delete, update, merge, create, or write data.
7. Never use APOC.
8. Never use type(path[0]).
9. If using a path, define it first with MATCH path = (...).
10. If evidence is not available, return a safe empty-result query.

Output format must always be:
RETURN ... AS person, ... AS evidence_type, ... AS evidence

Question-type rules:

A. Direct person property questions
Use Person properties when available:
- country
- birthYear
- estimatedNetWorthUSDBillions
- name

Example for birth year:
MATCH (p:Person)
WHERE toLower(p.name) = toLower("Mark Zuckerberg")
RETURN p.name AS person,
       "person_property" AS evidence_type,
       {{birthYear: p.birthYear}} AS evidence
LIMIT 10

Example for net worth:
MATCH (p:Person)
WHERE toLower(p.name) = toLower("Jeff Bezos")
RETURN p.name AS person,
       "person_property" AS evidence_type,
       {{estimatedNetWorthUSDBillions: p.estimatedNetWorthUSDBillions}} AS evidence
LIMIT 10

B. Ranking questions
If the schema has no direct rank property, do NOT use p.rank.
Search textual evidence in RelationshipFact.description or Person.data.

Example:
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PROFILE_FACT|HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(coalesce(rf.description, "")) CONTAINS "ranked number 1"
   OR toLower(coalesce(p.data, "")) CONTAINS "ranked number 1"
RETURN p.name AS person,
       "ranking_evidence" AS evidence_type,
       coalesce(rf.description, p.data) AS evidence
LIMIT 10

C. Company association questions
For questions asking main company, associated company, company connected to a person, or organizations connected to a person, prefer ASSOCIATED_WITH.

Example:
MATCH (p:Person)-[:ASSOCIATED_WITH]->(c:Company)
WHERE toLower(p.name) = toLower("Jensen Huang")
RETURN p.name AS person,
       "associated_company" AS evidence_type,
       collect(DISTINCT c.name) AS evidence
LIMIT 10

If ASSOCIATED_WITH does not return useful evidence, fall back to csvCompanies or RelationshipFact.description.

D. Role questions
For questions asking founder, CEO, cofounder, chairman, investor, or role:
MATCH (p:Person)-[:HAS_ROLE]->(r:Role)
WHERE toLower(p.name) = toLower("Elon Musk")
RETURN p.name AS person,
       "role" AS evidence_type,
       collect(DISTINCT r.name) AS evidence
LIMIT 10

E. Industry or business-interest questions
For questions asking industry, sector, business interest, or field:
MATCH (p:Person)-[:HAS_BUSINESS_INTEREST_IN]->(i:Industry)
WHERE toLower(p.name) = toLower("Mukesh Ambani")
RETURN p.name AS person,
       "business_interest" AS evidence_type,
       collect(DISTINCT i.name) AS evidence
LIMIT 10

F. Relationship between two people
For questions like:
- How are X and Y connected?
- What is the business relationship between X and Y?
- How is X related to Y?

First search direct person-person relationships.
Then search company-company relationships through associated companies.
Then search RelationshipFact descriptions mentioning both people.

Use this pattern:

MATCH (p1:Person)-[r:BUSINESS_RELATED_TO|COMPETES_WITH|BUSINESS_RELATIONSHIP]-(p2:Person)
WHERE toLower(p1.name) = toLower("PERSON_A")
  AND toLower(p2.name) = toLower("PERSON_B")
RETURN p1.name + " and " + p2.name AS person,
       type(r) AS evidence_type,
       properties(r) AS evidence
LIMIT 10

UNION

MATCH (p1:Person)-[:ASSOCIATED_WITH]->(c1:Company)-[r:BUSINESS_RELATIONSHIP|COMPETES_WITH]-(c2:Company)<-[:ASSOCIATED_WITH]-(p2:Person)
WHERE toLower(p1.name) = toLower("PERSON_A")
  AND toLower(p2.name) = toLower("PERSON_B")
RETURN p1.name + " and " + p2.name AS person,
       type(r) AS evidence_type,
       {{from_company: c1.name, to_company: c2.name, relationship: properties(r)}} AS evidence
LIMIT 10

UNION

MATCH (p:Person)-[:HAS_PROFILE_FACT|HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(coalesce(rf.description, "")) CONTAINS toLower("PERSON_A")
  AND toLower(coalesce(rf.description, "")) CONTAINS toLower("PERSON_B")
RETURN p.name AS person,
       "relationship_fact" AS evidence_type,
       rf.description AS evidence
LIMIT 10

G. Semantic questions
For questions asking:
- Which person is connected to AI GPUs?
- Who is connected to discount supermarkets?
- Who is linked to electronic brokerage?
- Who is connected to bottled water?
- Who is related to telecom, retail, or digital services?

Use RelationshipFact.description, Person.data, csvCompanies, csvIndustries, companies, and Industry nodes.

Example:
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PROFILE_FACT|HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
OPTIONAL MATCH (p)-[:HAS_BUSINESS_INTEREST_IN]->(i:Industry)
WHERE toLower(coalesce(rf.description, "")) CONTAINS toLower("AI GPU")
   OR toLower(coalesce(rf.description, "")) CONTAINS toLower("data center")
   OR toLower(coalesce(p.data, "")) CONTAINS toLower("AI GPU")
   OR toLower(coalesce(i.name, "")) CONTAINS toLower("AI")
RETURN p.name AS person,
       "semantic_evidence" AS evidence_type,
       collect(DISTINCT coalesce(rf.description, i.name, p.data)) AS evidence
LIMIT 10

H. Named-person semantic questions
If the question contains a specific person name, always anchor the query to that person first.

Example:
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PROFILE_FACT|HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(p.name) = toLower("Mukesh Ambani")
RETURN p.name AS person,
       "person_profile_evidence" AS evidence_type,
       collect(DISTINCT coalesce(rf.description, p.data, p.csvCompanies)) AS evidence
LIMIT 10

I. Unsupported or unanswerable questions
If the question asks for information not present in the schema or evidence, return this safe empty query:

MATCH (p:Person)
WHERE false
RETURN p.name AS person,
       "insufficient_evidence" AS evidence_type,
       "No supporting graph evidence found" AS evidence

Question:
{question}
"""

QA_TEMPLATE = """
You are a strict graph-grounded question-answering assistant.

Your task:
Answer the user's question using only the provided Database Results.

Question:
{question}

Database Results:
{context}

RULES:
- Use only the Database Results.
- Do not use outside knowledge.
- Do not mention or print these rules in the final answer.
- If Database Results is empty, answer exactly:
  The graph does not contain enough information.

- If Database Results is not empty, answer using only those results.
- Do not invent facts, relationships, companies, people, dates, or explanations.
- Do not invent relationship types that are not present in Database Results.
- Do not use relationshipTypes metadata as proof of real-world facts beyond what the graph path shows.
- Do not convert ASSOCIATED_WITH into ownership, investment, founder, CEO, acquisition, or partnership unless the evidence explicitly says so.
- If evidence_type is "associated_with", use generic wording such as "associated with" or "connected to".
- If the user asks for a specific relationship, such as investment or ownership, but the Database Results only show associated_with, clearly say that the graph shows association but does not explicitly prove that specific relationship.
- If evidence_type is "rank_property" or "richest_person", explain that the person is highest-ranked/richest because the rank evidence is the smallest rank value.
- If the result contains path_nodes and relationship_types, explain the connection naturally.
- Do not print raw maps like {{path_nodes: ..., relationship_types: ...}} unless the user asks for raw evidence.
- Do not include Cypher syntax such as AS evidence in the final answer.
If Database Results is not empty, Evidence used must be graph.
Only write Evidence used: none when Database Results is empty.

STYLE:
- Keep the answer concise and clear.
- Mention direct paths first, then indirect paths.
- Avoid repeating the same path multiple times.
- If the evidence is weak or generic, say so honestly.

FINAL ANSWER FORMAT:
Write the answer exactly like this:

<clear answer paragraph>

Evidence used: <graph or none>
Evidence summary: <one concise sentence summarizing only the graph evidence explicitly present in Database Results>
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
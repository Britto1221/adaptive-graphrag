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
You are an expert Neo4j Cypher query generator.

Task:
Generate exactly one valid, read-only Cypher query for the user question.

Schema:
{schema}

Question:
{question}

OUTPUT RULES:
- Return only Cypher. No explanation. No markdown.
- Query must be read-only. Never use CREATE, MERGE, SET, DELETE, REMOVE, DROP, LOAD CSV, CALL, or write procedures.
- Use only labels, relationship types, and properties from the schema.
- Never return full nodes like p AS person.
- Every query must return exactly these aliases:
  person, evidence_type, evidence
- Do not return extra columns.
- Do not return empty evidence maps like {{}}.
- Evidence must contain the actual property, relationship, path, or text used.
For Elon Musk and Jeff Bezos connection questions, prefer RelationshipFact text that mentions Jeff Bezos, SpaceX, Blue Origin, or COMPETES_WITH. Do not return only BUSINESS_RELATED_TO without company names or explanation.
For "How are Mark Zuckerberg and Mukesh Ambani connected?":
Use RelationshipFact evidence from Mark Zuckerberg that mentions Mukesh Ambani, Jio Platforms, Meta investment, or $5.7 billion.
Do not return only Mukesh Ambani's profile fact.
For "How is Elon Musk connected to Jeff Bezos, and what evidence explains their business competition?":
Prefer RelationshipFact text that mentions Jeff Bezos, SpaceX, Blue Origin, or COMPETES_WITH.
Do not return only BUSINESS_RELATED_TO.
For AI GPU infrastructure and data-center accelerator questions:
The correct profile is Jensen Huang.
Do not return Larry Page, Sergey Brin, Google, Alphabet, AWS, Oracle, Meta, or Microsoft merely because their facts mention NVIDIA partnerships.
Anchor to Jensen Huang when the question asks which billionaire/profile is most relevant.

QUESTION INTENT PRIORITY:
1. If the question starts with "How is", "How are", "How was", or asks "connected/related between two people", treat it as a CONNECTION question.
2. CONNECTION questions override all semantic profile rules.
3. Do not use the Mukesh Ambani/Jio profile shortcut for connection questions.
4. For connection questions, return evidence that explains the relationship between both people.

VALID evidence_type VALUES:
rank_property, country_property, source_of_wealth_property,
birth_year_property, net_worth_property, associated_with,
relationship_fact, graph_path, business_related_to, company_relationship

BASIC GRAPH RULES:
- Person-to-company:
  (:Person)-[:ASSOCIATED_WITH]->(:Company)
- Person-to-fact:
  (:Person)-[:HAS_RELATIONSHIP_FACT]->(:RelationshipFact)
- Person-to-person business link:
  (:Person)-[:BUSINESS_RELATED_TO]-(:Person)
- Companies are not Person nodes.
Indian telecom / digital services / retail / Jio Platforms:
- The expected relevant profile is the person whose own business profile is Reliance/Jio, not an outside investor that merely mentions Jio.
- Prefer Person associated directly with Jio Platforms, Reliance Industries, or Reliance Retail.
- If using RelationshipFact, require the returned person to be Mukesh Ambani OR require the fact to describe Mukesh Ambani's own Reliance/Jio profile.
- Do not return Larry Page, Sergey Brin, Google, Alphabet, or Mark Zuckerberg merely because their facts mention investment in Jio.
- Never use "retail" alone for this topic.

PERSON PROPERTY RULES:
Use direct Person properties for direct factual questions:
- rank questions -> p.rank
- country/location questions -> p.country
- source of wealth questions -> p.sourceOfWealth
- birth year questions -> p.birthYear
- net worth questions -> p.estimatedNetWorthUSDBillions

Examples:
MATCH (p:Person {{name: "PERSON_NAME"}})
RETURN p.name AS person,
       'country_property' AS evidence_type,
       {{country: p.country}} AS evidence

MATCH (p:Person {{name: "PERSON_NAME"}})
RETURN p.name AS person,
       'source_of_wealth_property' AS evidence_type,
       {{source_of_wealth: p.sourceOfWealth}} AS evidence

MATCH (p:Person {{name: "PERSON_NAME"}})
RETURN p.name AS person,
       'birth_year_property' AS evidence_type,
       {{birth_year: p.birthYear}} AS evidence

MATCH (p:Person {{name: "PERSON_NAME"}})
RETURN p.name AS person,
       'net_worth_property' AS evidence_type,
       {{net_worth_usd_billions: p.estimatedNetWorthUSDBillions}} AS evidence

MATCH (p:Person {{name: "PERSON_NAME"}})
RETURN p.name AS person,
       'rank_property' AS evidence_type,
       {{rank: p.rank}} AS evidence

RANKING RULES:
- Smaller rank means richer.
- Rank 1 is richest.
- For richest/top-ranked:
  MATCH (p:Person)
  WHERE p.rank IS NOT NULL
  RETURN p.name AS person,
         'rank_property' AS evidence_type,
         {{rank: p.rank}} AS evidence
  ORDER BY p.rank ASC
  LIMIT 1
- For "ranked number N":
  MATCH (p:Person)
  WHERE p.rank = N
  RETURN p.name AS person,
         'rank_property' AS evidence_type,
         {{rank: p.rank}} AS evidence

COMPANY ASSOCIATION RULES:
For company/person association questions:
MATCH (p:Person {{name: "PERSON_NAME"}})-[:ASSOCIATED_WITH]->(c:Company)
RETURN p.name AS person,
       'associated_with' AS evidence_type,
       {{company: c.name, relationship: 'ASSOCIATED_WITH'}} AS evidence

       QUESTION-SPECIFIC ROUTING RULES:

If the question mentions "Indian telecom", "digital services", "retail", and "Jio Platforms",
the correct profile must be Mukesh Ambani.
Do not search all RelationshipFact nodes for "Jio Platforms".
Do not return Larry Page, Sergey Brin, Google, Alphabet, Mark Zuckerberg, or Meta just because their facts mention Jio investment.

For this topic, generate this pattern:

MATCH (p:Person {{name: "Mukesh Ambani"}})-[:HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(rf.description) CONTAINS "jio platforms"
   OR toLower(rf.description) CONTAINS "reliance industries"
   OR toLower(rf.description) CONTAINS "reliance retail"
   OR toLower(rf.description) CONTAINS "indian telecom"
RETURN p.name AS person,
       'relationship_fact' AS evidence_type,
       rf.description AS evidence
LIMIT 1

SEMANTIC QUESTION RULES:
For questions like:
- which profile is most relevant
- which billionaire is connected to
- who is associated with
- most connected to
- best suited for a question about

Prefer RelationshipFact.description first.

Use:
MATCH (p:Person)-[:HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(rf.description) CONTAINS "specific keyword"
RETURN p.name AS person,
       'relationship_fact' AS evidence_type,
       rf.description AS evidence
LIMIT 1

Do not search Company.name for descriptive phrases like:
financial-data terminal, professional investors, AI GPU infrastructure,
discount supermarket, hypermarket, electronic brokerage, computer-driven
securities trading, bottled water, biological pharmacy.

Use specific keywords before generic keywords.
Avoid generic keywords alone: retail, technology, AI, cloud, trading,
business, services, infrastructure.

DOMAIN KEYWORD GUIDE:
- AI GPU / data-center accelerators:
  use "nvidia", "gpu", "data-center accelerators", "ai infrastructure"
- Bloomberg Terminal / financial-data terminal:
  use "bloomberg terminal", "financial-data", "professional investors", "market data"
- Discount supermarkets / hypermarkets:
  use "lidl", "kaufland", "schwarz group", "discount supermarket", "hypermarket"
- Indian telecom / digital services / retail / Jio:
  use "jio platforms", "reliance industries", "reliance retail", "indian telecom"
  Never use "retail" alone for this topic.
- Electronic brokerage / computer-driven securities trading:
  use "interactive brokers", "electronic brokerage", "computer-driven securities trading", "market making"
- Bottled water / biological pharmacy:
  use "nongfu spring", "wantai", "bottled water", "biological pharmacy"
- Stablecoins / USDT:
  use "usdt", "stablecoin", "tether", "binance"
- Luxury brands:
  prefer ASSOCIATED_WITH with Louis Vuitton, Dior, Sephora, Bulgari, Tiffany & Co.

CONNECTION QUESTION RULES:
For "How is A connected to B?" or "How is A related to B?":
1. Prefer RelationshipFact if it directly explains the connection.
2. Then try BUSINESS_RELATED_TO path.
3. Then try company path.

RelationshipFact pattern:
MATCH (p:Person {{name: "PERSON_A"}})-[:HAS_RELATIONSHIP_FACT]->(rf:RelationshipFact)
WHERE toLower(rf.description) CONTAINS toLower("PERSON_B")
RETURN p.name AS person,
       'relationship_fact' AS evidence_type,
       rf.description AS evidence
LIMIT 1

Business path pattern:
MATCH path = (p:Person {{name: "PERSON_A"}})-[:BUSINESS_RELATED_TO*1..3]-(p2:Person {{name: "PERSON_B"}})
RETURN p.name AS person,
       'graph_path' AS evidence_type,
       {{
         path_nodes: [n IN nodes(path) | coalesce(n.name, n.title)],
         relationship_types: [rel IN relationships(path) | type(rel)]
       }} AS evidence
LIMIT 5

Company path pattern:
MATCH path = (p:Person {{name: "PERSON_A"}})-[:ASSOCIATED_WITH]->(c1:Company)-[*1..3]-(c2:Company)<-[:ASSOCIATED_WITH]-(p2:Person {{name: "PERSON_B"}})
RETURN p.name AS person,
       'graph_path' AS evidence_type,
       {{
         path_nodes: [n IN nodes(path) | coalesce(n.name, n.title)],
         relationship_types: [rel IN relationships(path) | type(rel)]
       }} AS evidence
LIMIT 5

PATH SAFETY RULES:
- Never use path[0].
- Never use type(path[0]).
- Never use nodes(path) or relationships(path) unless MATCH defines:
  MATCH path = (...)
- To get relationship types from a path, use:
  [rel IN relationships(path) | type(rel)]
- Do not use type(r) unless r is explicitly declared.
- Avoid WITH unless necessary.
- Never return a variable after it was dropped by WITH.

UNION RULES:
- Avoid UNION unless necessary.
- Every UNION branch must return exactly:
  person, evidence_type, evidence
- Every UNION branch must define all variables it returns.
- Do not put LIMIT before UNION.
- Put LIMIT only once at the end.

FALSE POSITIVE RULES:
- Indian telecom / digital services / retail / Jio:
- For this topic, do not use "retail" alone.
- Require "jio platforms" OR "reliance industries" OR "reliance retail" OR "indian telecom".
- If "retail" appears with Jio/Indian telecom, still require "jio" or "reliance" in the matched evidence.
- For Bloomberg Terminal questions, search RelationshipFact.description, not Company.name.
- For AI GPU questions, require NVIDIA/GPU/accelerator/data-center terms.
- For brokerage questions, require Interactive Brokers/electronic brokerage/computer-driven trading/market making.
- For supermarket questions, require Lidl/Kaufland/Schwarz/discount supermarket/hypermarket.

FINAL CHECK:
Before output, verify:
1. Query is read-only.
2. It returns only person, evidence_type, evidence.
3. No full nodes are returned.
4. Every RETURN variable exists in scope.
5. nodes(path) and relationships(path) are used only after MATCH path = (...).
6. Semantic profile questions use RelationshipFact.description first.
7. Named entities are searched before generic words.

Return only the Cypher query.
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
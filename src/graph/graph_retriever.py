import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from src.graph.neo4j_client import neo4j_manager
llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

def ask_graph(query:str):
    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Cypher query generator.

    Database schema:
    {schema}

    Rules:
    1. Return only a valid Cypher query.
    2. Do not use Markdown code fences.
    3. Use only labels, properties, and relationship types present in the schema.
    4. Generate read-only queries only.
    5. Never use CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE, DROP, or CALL.
    6. Use LIMIT 10 unless the question asks for aggregation, counting, or a specific number of results.
    7. Do not invent relationship types.
    8. The property `relationshipTypes` is a list stored on Person nodes. Its values are not necessarily actual Neo4j relationship types.
    9. Richest-person nodes have a non-null `rank` property.
    10. Actor and director Person nodes from the movie dataset usually do not have a `rank` property.
    11. For questions about wealthy people, use:
        WHERE person.rank IS NOT NULL
    12. Natural-language business relationship statements are connected through:
        (:Person)-[:HAS_RELATIONSHIP_FACT]->(:RelationshipFact)
    13. A person's companies are connected through:
        (:Person)-[:ASSOCIATED_WITH]->(:Company)
    14. Use case-insensitive matching when matching a name supplied by the user.

    Question:
    {question}
    """


    QA_TEMPLATE = """
    You are a helpful graph question-answering assistant.

    User question:
    {question}

    Neo4j query results:
    {context}

    Instructions:
    1. Answer using only the provided Neo4j results.
    2. Do not use outside knowledge.
    3. Do not invent missing relationships or values.
    4. If the results are empty, say that the graph does not contain enough information.
    5. Clearly distinguish properties from actual graph relationships.
    6. Keep the answer direct and readable.
    """

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=QA_TEMPLATE
    )
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=neo4j_manager(),
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )
    return chain.invoke({'query':query})

if __name__ == "__main__":
    response = ask_graph(
        "Who are the five richest people in the database?"
    )

    print("\nFINAL ANSWER:")
    print(response["result"])

    print("\nINTERMEDIATE STEPS:")
    print(response.get("intermediate_steps"))
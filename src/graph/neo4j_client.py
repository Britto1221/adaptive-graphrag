import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
load_dotenv()

def neo4j_manager():
    graph=Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")
    )
    return graph
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()
client = Client()

project = client.read_project(
    project_name="adaptive-graphrag-graph-openai"
)

client.update_project(
    project_id=project.id,
    name="adaptive-graphrag-vector-openai"
)

print("Project renamed successfully.")
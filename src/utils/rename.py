import os
import requests
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

LANGSMITH_API_KEY = (
    os.getenv("LANGSMITH_API_KEY")
    or os.getenv("LANGCHAIN_API_KEY")
)

LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
).rstrip("/")

if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY or LANGCHAIN_API_KEY is missing in .env")

client = Client(
    api_key=LANGSMITH_API_KEY,
    api_url=LANGSMITH_ENDPOINT,
)

PROJECT_RENAMES = {
    # old LangSmith project name : new LangSmith project name

    "adaptive-graphrag-vector-llama3.2:981b-gguf":
        "adaptive-graphrag-vector-local-llama3-2-1b-ft",

    "adaptive-graphrag-graph-llama3.2:981b-gguf":
        "adaptive-graphrag-graph-local-llama3-2-1b-ft",

    "adaptive-graphrag-hybrid-llama3.2:981b-gguf":
        "adaptive-graphrag-hybrid-local-llama3-2-1b-ft",
}


def rename_project(old_name: str, new_name: str) -> None:
    print("\n" + "=" * 80)
    print("Old name:", old_name)
    print("New name:", new_name)

    project = client.read_project(project_name=old_name)
    project_id = str(project.id)

    print("Project ID:", project_id)

    url = f"{LANGSMITH_ENDPOINT}/api/v1/sessions/{project_id}"

    headers = {
        "x-api-key": LANGSMITH_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "name": new_name,
    }

    response = requests.patch(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    print("Status code:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()

    print("Renamed successfully.")


def main() -> None:
    for old_name, new_name in PROJECT_RENAMES.items():
        try:
            rename_project(old_name, new_name)
        except Exception as e:
            print("\nFailed to rename:", old_name)
            print("Reason:", repr(e))


if __name__ == "__main__":
    main()
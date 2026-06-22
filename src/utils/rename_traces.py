import os
import time
import requests
from dotenv import load_dotenv
from langsmith import Client


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

LANGSMITH_API_KEY = (
    os.getenv("LANGSMITH_API_KEY")
    or os.getenv("LANGCHAIN_API_KEY")
)

LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
).rstrip("/")


# ============================================================
# Safety checks
# ============================================================

if not PROJECT_NAME:
    raise ValueError(
        "LANGCHAIN_PROJECT is missing. Add it to your .env file."
    )

if not LANGSMITH_API_KEY:
    raise ValueError(
        "LANGSMITH_API_KEY or LANGCHAIN_API_KEY is missing. Add it to your .env file."
    )


# ============================================================
# Settings
# ============================================================

# First keep this True.
# After checking the printed trace IDs, change to False.
DRY_RUN = False

# Delete in small batches to avoid timeout.
BATCH_SIZE = 5

# Delay between delete requests.
SLEEP_BETWEEN_BATCHES_SECONDS = 3


# ============================================================
# LangSmith client
# ============================================================

client = Client(
    api_key=LANGSMITH_API_KEY,
    api_url=LANGSMITH_ENDPOINT,
)


def get_project_id(project_name: str) -> str:
    print("Reading LangSmith project...")
    project = client.read_project(project_name=project_name)
    session_id = str(project.id)

    print("Project name:", project_name)
    print("Project/session ID:", session_id)

    return session_id


def get_error_trace_ids(project_name: str) -> list[str]:
    print("\nListing error runs...")
    print("This may take time if the project has many traces.\n")

    error_runs = list(
        client.list_runs(
            project_name=project_name,
            filter='eq(status, "error")',
        )
    )

    trace_ids = sorted({
        str(run.trace_id)
        for run in error_runs
        if getattr(run, "trace_id", None)
    })

    print("Error child runs found:", len(error_runs))
    print("Unique full traces to delete:", len(trace_ids))

    print("\nError runs found:")
    for run in error_runs:
        print({
            "run_id": str(run.id),
            "trace_id": str(run.trace_id),
            "run_type": getattr(run, "run_type", ""),
            "name": getattr(run, "name", ""),
            "status": getattr(run, "status", ""),
        })

    print("\nTrace IDs that will be deleted:")
    for trace_id in trace_ids:
        print(trace_id)

    return trace_ids


def delete_traces(session_id: str, trace_ids: list[str]) -> None:
    if not trace_ids:
        print("\nNo error traces found. Nothing to delete.")
        return

    if DRY_RUN:
        print("\nDRY_RUN=True, so nothing was deleted.")
        print("After checking the trace IDs, change DRY_RUN=False and run again.")
        return

    delete_url = f"{LANGSMITH_ENDPOINT}/api/v1/runs/delete"

    headers = {
        "x-api-key": LANGSMITH_API_KEY,
        "Content-Type": "application/json",
    }

    total_batches = (len(trace_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    print("\nStarting deletion...")
    print("Delete URL:", delete_url)
    print("Total traces:", len(trace_ids))
    print("Batch size:", BATCH_SIZE)
    print("Total batches:", total_batches)

    for batch_index, start in enumerate(range(0, len(trace_ids), BATCH_SIZE), start=1):
        batch = trace_ids[start:start + BATCH_SIZE]

        payload = {
            "session_id": session_id,
            "trace_ids": batch,
        }

        print(f"\nDeleting batch {batch_index}/{total_batches}")
        print("Trace IDs:", batch)

        response = requests.post(
            delete_url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        print("Status code:", response.status_code)
        print("Response:", response.text)

        response.raise_for_status()

        time.sleep(SLEEP_BETWEEN_BATCHES_SECONDS)

    print("\nDelete request submitted successfully.")
    print("LangSmith deletion is not always instant. Refresh after some time.")


def main() -> None:
    session_id = get_project_id(PROJECT_NAME)
    trace_ids = get_error_trace_ids(PROJECT_NAME)
    delete_traces(session_id, trace_ids)


if __name__ == "__main__":
    main()
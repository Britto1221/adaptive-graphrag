import json
import csv
from pathlib import Path
import time
from src.pipelines.graph_rag import graph_pipeline
from src.pipelines.vector_rag import vector_pipeline
from src.pipelines.hybrid_rag import hybrid_pipeline
from src.evaluation.answer_evaluator import evaluate_answer

PIPELINE_NAME = "graph_rag"
CYPHER_MODEL = "openai"
ANSWER_MODEL = "openai"
experiment_name="VectorRAG + OpenAI answer"
langsmith_project_name="graphrag-graph-openai"
batch_id="exp-batch-02"

QUESTIONS_FILE = Path("data/benchmark/questions/evaluation_questions.jsonl")
RESULTS_FILE = Path("reports/benchmark_results.csv")


CSV_HEADERS = [
    "run_id",

    "question_id",
    "category",
    "question",
    "is_answerable",
    "expected_evidence_type",

    "pipeline",
    "cypher_model",
    "answer_model",

    "generated_cypher",
    "graph_context_count",

    "answer",
    "graph_evidence",
    "vector_evidence",

    "correctness_score",
    "faithfulness_score",
    "hallucination_score",
    "evidence_score",
    "refusal_score",
    "overall_score",
    "evidence_used",
    "reason",

    "error_type",
    "error_message",

    "experiment_name",
    "langsmith_project_name",
    "batch_id",

    "local_latency_seconds",
]


def load_questions() -> list[dict]:
    """Load benchmark questions from JSONL file."""
    questions = []

    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def create_csv_if_needed() -> None:
    """Create results CSV with headers if it does not exist."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        with RESULTS_FILE.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def get_cypher_and_context_count(result: dict) -> tuple[str, int]:
    """Extract generated Cypher and graph context count."""
    steps = result.get("raw_graph_response", {}).get("intermediate_steps", [])

    generated_cypher = ""
    context_count = 0

    for step in steps:
        if isinstance(step, dict) and "query" in step:
            generated_cypher = step["query"]

        if isinstance(step, dict) and "context" in step:
            context = step["context"]
            if isinstance(context, list):
                context_count = len(context)

    return generated_cypher, context_count


def normalize_eval(eval_result) -> dict:
    """Convert evaluator result into a normal dictionary."""
    if hasattr(eval_result, "model_dump"):
        eval_result = eval_result.model_dump()

    return {
        "correctness_score": int(eval_result.get("correctness_score", 1)),
        "faithfulness_score": int(eval_result.get("faithfulness_score", 1)),
        "evidence_score": int(eval_result.get("evidence_score", 1)),
        "refusal_score": int(eval_result.get("refusal_score", 1)),
        "overall_score": int(eval_result.get("overall_score", 1)),
        "evidence_used": eval_result.get("evidence_used", "none"),
        "reason": eval_result.get("reason", ""),
    }


def run_one_question(question_row: dict) -> dict:
    """Run one question through GraphRAG and evaluate the answer."""
    query = question_row["question"]
    start_time = time.perf_counter()
    try:
        result = vector_pipeline(query)
        local_latency_seconds = time.perf_counter() - start_time
        answer = result.get("answer", "")
        graph_evidence = result.get("graph_evidence", "")
        vector_evidence = result.get("vector_evidence", "")
        evidence = [graph_evidence,vector_evidence]
        generated_cypher, context_count = get_cypher_and_context_count(result)

        eval_result = evaluate_answer(
            question=query,
            expected_answer=str(question_row.get("expected_answer", "")),
            answer=answer,
            evidence=evidence,
        )

        scores = normalize_eval(eval_result)
        print("\nANSWER:")
        print(result.get("answer", ""))

        print("\nVECTOR EVIDENCE:")
        print(result.get("vector_evidence", "")[:1000])

        print("\nEVALUATOR REASON:")
        print(scores.get("reason", ""))
        return {
            "run_id":"",

            "question_id": question_row.get("id", ""),
            "category": question_row.get("category", ""),
            "question": query,
            "is_answerable": question_row.get("is_answerable", ""),
            "expected_evidence_type": question_row.get("expected_evidence_type", ""),

            "pipeline": PIPELINE_NAME,
            "cypher_model": CYPHER_MODEL,
            "answer_model": ANSWER_MODEL,

            "generated_cypher": generated_cypher,
            "graph_context_count": context_count,

            "answer": answer,
            "graph_evidence": graph_evidence,
            "vector_evidence": vector_evidence,

            "correctness_score": scores["correctness_score"],
            "faithfulness_score": scores["faithfulness_score"],
            "hallucination_score":6-scores["faithfulness_score"],
            "evidence_score": scores["evidence_score"],
            "refusal_score": scores["refusal_score"],
            "overall_score": scores["overall_score"],
            "evidence_used": scores["evidence_used"],
            "reason":scores["reason"],

            "error_type": "",
            "error_message": "",

            "experiment_name":experiment_name,
            "langsmith_project_name":langsmith_project_name,
            "batch_id":batch_id,

            "local_latency_seconds":local_latency_seconds,
        }

    except Exception as e:
        local_latency_seconds = time.perf_counter() - start_time
        return {
            "run_id":"",

            "question_id": question_row.get("id", ""),
            "category": question_row.get("category", ""),
            "question": query,
            "is_answerable": question_row.get("is_answerable", ""),
            "expected_evidence_type": question_row.get("expected_evidence_type", ""),

            "pipeline": PIPELINE_NAME,
            "cypher_model": CYPHER_MODEL,
            "answer_model": ANSWER_MODEL,

            "generated_cypher": "",
            "graph_context_count": 0,

            "answer": "",
            "graph_evidence": "",
            "vector_evidence": "",

            "correctness_score": 0,
            "faithfulness_score": 0,
            "hallucination_score":0,
            "evidence_score": 0,
            "refusal_score": 0,
            "overall_score": 0,
            "evidence_used": "none",
            "reason": "",

            "error_type": type(e).__name__,
            "error_message": str(e),

            "experiment_name":experiment_name,
            "langsmith_project_name":langsmith_project_name,
            "batch_id":batch_id,

            "local_latency_seconds":local_latency_seconds,
        }


def save_row(row: dict) -> None:
    """Append one result row to CSV."""
    with RESULTS_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writerow(row)

def run_benchmark(limit: int | None = 5) -> None:
    questions = load_questions()
    if limit is not None:
        questions = questions[:limit]
    create_csv_if_needed()
    print(f"Loaded {len(questions)} questions")
    print(f"Saving results to {RESULTS_FILE}")
    for index, question_row in enumerate(questions, start=1):
        print("\n" + "=" * 80)
        print(f"Running question {index}/{len(questions)}")
        print(f"Query: {question_row['question']}")
        row = run_one_question(question_row)
        save_row(row)
        print(f"Overall score: {row['overall_score']}")
        if row["error_type"]:
            print(f"Error: {row['error_type']} - {row['error_message']}")
    print("\nBenchmark completed.")

if __name__ == "__main__":
    run_benchmark(limit=1)
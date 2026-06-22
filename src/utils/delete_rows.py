from pathlib import Path
import pandas as pd

RESULTS_FILE = Path("reports/benchmark_results.csv")
BACKUP_FILE = Path("reports/benchmark_results_backup_before_error_delete.csv")

df = pd.read_csv(RESULTS_FILE)

# Backup first
df.to_csv(BACKUP_FILE, index=False)

# Normalize error column
error_col = df["error_type"].fillna("").astype(str).str.strip()

# Delete ONLY NVIDIA VectorRAG error rows
delete_mask = (
    (df["pipeline"] == "hybrid_rag") &
    (df["answer_model"] == "groq") &
    (error_col != "")
)

print("Rows before:", len(df))
print("Error rows to delete:", delete_mask.sum())

print(
    df[delete_mask][
        ["question_id", "pipeline", "answer_model", "error_type", "error_message"]
    ].to_string(index=False)
)

clean = df[~delete_mask]

clean.to_csv(RESULTS_FILE, index=False)

print("\nRows after:", len(clean))
print("Backup saved to:", BACKUP_FILE)
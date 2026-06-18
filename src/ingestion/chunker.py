from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
import json
from pathlib import Path
from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents

def chunk_documents(documents):
    """Split cleaned documents into chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)
    
    return chunks 

CHUNKS_PATH = "data/chunks/chunks.jsonl"

def save_chunks(chunks: list[Document],output_path: str = CHUNKS_PATH,) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record,ensure_ascii=False,)+ "\n")
    print(f"Saved {len(chunks)} chunks to {path}")

def load_saved_chunks(input_path: str = CHUNKS_PATH,) -> list[Document]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Saved chunks file not found: {path}. "
            "Run python -m scripts.build_vector_index first."
        )
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            chunks.append(Document(page_content=record["page_content"],metadata=record["metadata"],))
    print(f"Loaded {len(chunks)} saved chunks from {path}")
    return chunks

if __name__ == "__main__":
    start = time.perf_counter()
    documents = load_documents("data/raw")
    clean_docs = clean_documents(documents)
    chunked_docs = chunk_documents(clean_docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"No of chunks: {len(chunked_docs)}")
    print(f"Processing time: {elapsed_ms:.2f} ms")
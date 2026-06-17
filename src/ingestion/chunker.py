from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
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

if __name__ == "__main__":
    start = time.perf_counter()
    documents = load_documents("data/raw")
    clean_docs = clean_documents(documents)
    chunked_docs = chunk_documents(clean_docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"No of chunks: {len(chunked_docs)}")
    print(f"Processing time: {elapsed_ms:.2f} ms")
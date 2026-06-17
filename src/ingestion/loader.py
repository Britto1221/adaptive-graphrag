# Imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from pathlib import Path
import time

def load_documents(directory:str="data/raw")->list[Document]:
    """Load all .txt, .pdf and .docx files from the given directory."""
    folder = Path(directory)

    # Checks wheather folder exists or not 
    if not folder.exists():
        raise FileNotFoundError(
            f"Documnet directory does not exist: {folder.resolve()}"
        )
    
    # Declare a empty list to load documents
    all_documents = []

    #Load Text Files
    txt_loader=DirectoryLoader(
        path=str(folder),
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
        },
        show_progress=True,
        use_multithreading=True,
        silent_errors=False,                
    )

    #Load pdf files
    pdf_loader = DirectoryLoader(
        path=str(folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=True,
    )

    # Load DOCX files
    docx_loader = DirectoryLoader(
        path=str(folder),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
        silent_errors=True,
    )

    all_documents.extend(txt_loader.load())
    all_documents.extend(pdf_loader.load())
    all_documents.extend(docx_loader.load())
    
    if not all_documents:
        raise ValueError(
            f"NO documents were found inside: {folder.resolve()}"
        )
    print(f"✅ Loaded {len(all_documents)} documents from {folder}")
    return all_documents

if __name__ == "__main__":
    start_time = time.perf_counter()
    docs = load_documents("data/raw")
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total documents loaded: {len(docs)}")
    print(f"First document preview: {docs[0].page_content[:200]}")
    print(f"Loading time: {elapsed_seconds:.4f} seconds")
    print(f"Loading time: {elapsed_seconds * 1000:.2f} ms")


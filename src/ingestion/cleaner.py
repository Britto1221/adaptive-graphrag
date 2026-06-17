from langchain_core.documents import Document
import re
import time

from src.ingestion.loader import load_documents

def clean_text(text:str)->str:
    """
    Clean extracted document text while preserving useful information.
    """
    if not text:
        return ""
    
    text = text.replace("\t"," ")

    text = text.replace("\r\n","\n").replace("\r","\n")

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    text = re.sub(r"[ ]{2,}", " ", text)

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()


def clean_documents(docs:list)->list[Document]:
    """
    Clean a list of LangChain Document objects
    while preserving their metadata.
    """

    cleaned_docs:list[Document]= []

    for doc in docs:
        cleaned_content = clean_text(doc.page_content)

        if not cleaned_content:
            continue

        cleaned_docs.append(
            Document(
                page_content=cleaned_content,
                metadata = doc.metadata.copy(),
            )
        )

    return cleaned_docs




def main() -> None:
    start = time.perf_counter()

    documents = load_documents("data/raw")
    cleaned_documents = clean_documents(documents)

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Processing time: {elapsed_ms:.2f} ms")

    if cleaned_documents:
        first_document = cleaned_documents[0]

        print("\nSource:")
        print(first_document.metadata.get("source"))

        print("\nCleaned preview:")
        print(first_document.page_content[:500])


if __name__ == "__main__":
    main()
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document


MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

ranker = Ranker(
    model_name=MODEL_NAME,
    cache_dir="models/flashrank",
)
def rerank_documents(query:str,documents:list[Document],top_k:int=3)->list[Document]:
    """
    Rerank retrieved LangChain Documents using FlashRank.
    """

    passages = []

    for index, document in enumerate(documents):
        passages.append(
            {
                "id": index,
                "text": document.page_content,
                "meta": document.metadata,
            }
        )

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    ranked_results = ranker.rerank(request)
    final_documents: list[Document] =[]

    for result in ranked_results[:top_k]:
        original_document = documents[int(result["id"])]

        metadata = original_document.metadata.copy()
        metadata["reranker_score"] = float(result["score"])

        final_documents.append(
            Document(
                page_content=original_document.page_content,
                metadata=metadata,
            )
        )

    return final_documents
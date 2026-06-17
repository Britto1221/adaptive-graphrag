from langchain_huggingface import HuggingFaceEmbeddings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Generate embeddings model for a list of document chunks.
    Returns embedding model ready for Pinecone upsert.
    """
    return HuggingFaceEmbeddings(
        model = MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True,
        },
    )
    


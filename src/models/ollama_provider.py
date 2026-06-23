from langchain_ollama import ChatOllama

models = ["llama3.2-1b-gguf","llama3.2:1b"]
def ollama_provider(model_name: str):
    return ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=2048
    )
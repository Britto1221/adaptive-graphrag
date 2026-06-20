from langchain_ollama import ChatOllama

models = ["qwen2.5:1.5b","llama3.2:1b"]
def ollama_provider(model_name: str):
    return ChatOllama(
        model=model_name,
        temperature=0
    )
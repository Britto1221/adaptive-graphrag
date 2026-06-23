from src.models.openai_provider import openai_provider
from src.models.nvidia_provider import nvidia_provider
from src.models.groq_provider import groq_provider
from src.models.ollama_provider import ollama_provider

MODELS = {
    "openai":      openai_provider(),
    "nvidia":      nvidia_provider(),
    "groq":        groq_provider(),
    "gemma-base":  ollama_provider("llama3.2:1b"),
    "gemma-ft":    ollama_provider("llama3.2-1b-gguf"),
}

def get_model(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]
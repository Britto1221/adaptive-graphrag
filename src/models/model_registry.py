from src.models.openai_provider import openai_provider
from src.models.nvidia_provider import nvidia_provider
from src.models.groq_provider import groq_provider
from src.models.ollama_provider import ollama_provider

MODELS = {
    "openai":      openai_provider(),
    "nvidia":      nvidia_provider(),
    "groq":        groq_provider(),
    "qwen-base":   ollama_provider("qwen2.5:1.5b"),
    "qwen-ft":     ollama_provider("qwen-finetuned"),
    "gemma-base":  ollama_provider("gemma2:2b"),
    "gemma-ft":    ollama_provider("gemma-finetuned"),
}

def get_model(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()

def nvidia_provider():
    return ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0
    )
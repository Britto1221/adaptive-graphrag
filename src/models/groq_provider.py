from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
def groq_provider():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
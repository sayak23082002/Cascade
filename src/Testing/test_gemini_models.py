import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

<<<<<<< HEAD
load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
=======
api_key = ""
>>>>>>> 7c2cfef65a7a9ceffd28a29866ea842060ae1d35

# 1. Use the REST API to list models via HTTP GET
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    print("Available Models:")
    models = response.json().get("models", [])
    for m in models:
        # Check if it supports text generation
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(f"- {m['name'].replace('models/', '')}")
else:
    print(f"Failed to fetch models. Status code: {response.status_code}")

# 2. Initialize LangChain with your chosen model string
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-002", # Choose a valid string from the list above
    api_key=api_key
)

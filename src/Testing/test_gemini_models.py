import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = "AIzaSyDBzOpErkZg7bpHYe2r1EPc-kZIE3ta0Qs"

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
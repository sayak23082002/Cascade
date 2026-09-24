import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(provider: str):
    """
    Factory function to initialize the LLM based on the provider name.
    Expects specific environment variables depending on the provider.
    """
    provider = provider.lower()
    
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"), # or gemini-1.5-flash
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
        
    elif provider == "azure":
        return AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0
        )
        
    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
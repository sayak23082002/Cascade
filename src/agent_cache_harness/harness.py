from typing import Dict, Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from .store import BaseStore
from .builder import PrefixBuilder

class UniversalCacheHarness:
    def __init__(self, store: BaseStore, static_system_prompt: str, static_tools: Optional[List[Dict]] = None):
        self.store = store
        self.builder = PrefixBuilder(static_system_prompt, static_tools)

    def execute(
        self, 
        llm: BaseChatModel, 
        dynamic_context: Dict[str, Any], 
        feedback: str = "",
        bypass_l1: bool = False
    ) -> str:
        
        model_name = getattr(llm, "model_name", getattr(llm, "model", "generic_model"))
        cacheable_payload = dynamic_context if not feedback else None

        # Level 1 Exact Match (Disk or Redis)
        if not bypass_l1 and cacheable_payload:
            cached_response = self.store.get(model_name, cacheable_payload)
            if cached_response:
                print(f">>> ⚡ L1 Exact Cache Hit ({type(self.store).__name__}). Zero tokens used.")
                return cached_response

        # Level 2 Prefix-Optimized Execution
        messages = self.builder.build_messages(dynamic_context, feedback)
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Store the successful generation back to L1
        if cacheable_payload:
            self.store.set(model_name, cacheable_payload, response_text)

        return response_text
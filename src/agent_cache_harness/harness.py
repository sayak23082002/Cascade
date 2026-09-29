from typing import Dict, Any, List
from langchain_core.language_models.chat_models import BaseChatModel
from .store import LocalDiskStore
from .builder import PrefixBuilder

class UniversalCacheHarness:
    def __init__(self, static_system_prompt: str, static_tools: List[Dict] = None, cache_dir: str = ".agent_cache"):
        self.store = LocalDiskStore(cache_dir=cache_dir)
        self.builder = PrefixBuilder(static_system_prompt, static_tools)

    def execute(
        self, 
        llm: BaseChatModel, 
        dynamic_context: Dict[str, Any], 
        feedback: str = "",
        bypass_l1: bool = False
    ) -> str:
        
        model_name = getattr(llm, "model_name", getattr(llm, "model", "generic_model"))
        
        # We only cache requests that do NOT have active error feedback.
        # If the LLM is actively correcting an error, we bypass L1 to force a fresh generation.
        cacheable_payload = dynamic_context if not feedback else None

        # Try Level 1 Exact Match
        if not bypass_l1 and cacheable_payload:
            cached_response = self.store.get(model_name, cacheable_payload)
            if cached_response:
                print(">>> ⚡ L1 Exact Cache Hit. Zero tokens used.")
                return cached_response

        # Level 2 Prefix-Optimized Execution
        messages = self.builder.build_messages(dynamic_context, feedback)
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Store the successful generation back to L1
        if cacheable_payload:
            self.store.set(model_name, cacheable_payload, response_text)

        return response_text
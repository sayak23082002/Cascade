import json
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

class PrefixBuilder:
    def __init__(self, static_system_prompt: str, static_tools: List[Dict] = None):
        """
        Layer 1 & 2: System prompt and Tool Definitions (Always Cached)
        These strings MUST be identical across every execution in the factory.
        """
        self.system_content = static_system_prompt
        if static_tools:
            # Formatting tools deterministically
            tools_str = json.dumps(static_tools, sort_keys=True, indent=2)
            self.system_content += f"\n\nAVAILABLE TOOLS:\n{tools_str}"

    def build_messages(self, dynamic_context: dict, runtime_feedback: str = "") -> List:
        """
        Layer 3 & 4: User Context and Error Feedback (Computed uniquely per request)
        """
        user_content = json.dumps(dynamic_context, indent=2)
        
        if runtime_feedback:
            user_content += f"\n\nQA CRITIQUE TO FIX:\n{runtime_feedback}"

        return [
            SystemMessage(content=self.system_content),
            HumanMessage(content=user_content)
        ]
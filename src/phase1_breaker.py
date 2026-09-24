import json
import tempfile as tempfile
import subprocess
from pathlib import Path
from typing import TypedDict, List, Literal, Union
from pydantic import BaseModel, Field, ValidationError, model_validator
from Provider.LLM_Provider import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import os
import re
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv(override = True)
ACTIVE_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
# APPROVED_TECH_STACK = {"PostgreSQL", "Redis", "Azure OpenAI", "FastAPI", "A2A", "Langchain", "Langgraph", "Langsmith"}
DISALLOWED_PACKAGES = {"os.system", "subprocess", "exec", "eval", "pickle", "shutil.rmtree"}
DISALLOWED_STORAGE = {"mongodb", "dynamodb", "couchbase", "cassandra"}
MANDATORY_STORAGE = {"postgresql", "redis"}

# --- 1. Schemas ---
class Blueprint(BaseModel):
    agent_objective: str = Field(
        ..., 
        description="Write a comprehensive paragraph detailing the primary responsibilities, business value, and core objectives of the agent. Explain the 'why' and the 'what' in detail."
    )
    agent_and_tool_dist: List[str] = Field(
        ..., 
        description="Highly detailed list of specific tools, libraries, engines, and domain knowledge bases required. Write full sentences describing *how* each tool is utilized (e.g., 'FastAPI is deployed to expose RESTful endpoints for Inter-Agent (A2A) communication')."
    )
    agent_access: List[str] = Field(
        ..., 
        description="Thorough, descriptive list of system access requirements, databases, APIs, and permissions. Include explicit descriptions of what data is being accessed, the integration method, and why it is necessary."
    )
    guardrails: List[str] = Field(
        ..., 
        description="Extensive list of use-case specific safety protocols, fallback mechanisms, validation rules, and compliance constraints to ensure production-grade reliability and prevent hallucinations."
    )
    agent_skills: List[str] = Field(
        ..., 
        description="Detailed descriptions of the core analytical, technical, and domain-specific skills the agent must possess. Explain exactly how these skills apply to the specific use case rather than just listing keywords."
    )

    @model_validator(mode='after')
    def run_production_guardrails(self):
        tools_str = " ".join(self.agent_and_tool_dist).lower()
        access_str = " ".join(self.agent_access).lower()
        combined_env = f"{tools_str} {access_str}"

        # Guardrail 1: Prohibit Dangerous Execution APIs
        for dangerous_pkg in DISALLOWED_PACKAGES:
            if dangerous_pkg in tools_str:
                raise ValueError(
                    f"SECURITY BREACH: Dangerous un-sandboxed primitive '{dangerous_pkg}' is strictly prohibited."
                )

        # Guardrail 2: Technology Blacklist vs. Supported Infrastructure
        for forbidden_db in DISALLOWED_STORAGE:
            if forbidden_db in combined_env:
                raise ValueError(
                    f"COMPLIANCE ERROR: Storage system '{forbidden_db}' is unsupported. Factory only permits PostgreSQL and Redis."
                )

        # Guardrail 3: Cross-Field Consistency Check (Tools must have declared Access)
        if "postgresql" in tools_str and "postgresql" not in access_str:
            raise ValueError(
                "CROSS-FIELD INCONSISTENCY: PostgreSQL tool is configured, but no PostgreSQL connection permissions are defined in 'agent_access'."
            )

        # Guardrail 4: Anti-Hallucination & Anti-Laziness Check
        # Enforce technical depth by preventing one-word keyword dumps
        for field_name, items in [
            ("agent_and_tool_dist", self.agent_and_tool_dist),
            ("agent_access", self.agent_access),
            ("guardrails", self.guardrails),
            ("agent_skills", self.agent_skills)
        ]:
            if len(items) < 3:
                raise ValueError(f"INSUFFICIENT DETAIL: '{field_name}' must define at least 3 distinct specifications.")
            
            for item in items:
                word_count = len(item.split())
                if word_count < 4:
                    raise ValueError(
                        f"LAZINESS DETECTED in '{field_name}': Item '{item}' is too brief. "
                        "All items must be detailed sentences defining the 'why' and 'how'."
                    )

        # Guardrail 5: Credential Leakage / Sensitive Environment Variable Exposure
        secret_patterns = [r"sk-[a-zA-Z0-9]{20,}", r"ghp_[a-zA-Z0-9]{20,}", r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+"]
        for pattern in secret_patterns:
            if re.search(pattern, combined_env):
                raise ValueError("LEAKAGE RISK: Hardcoded secret keys or auth tokens detected in blueprint schema.")

        return self

class BlueprintPatch(BaseModel):
    field_to_update: Literal["agent_objective", "agent_and_tool_dist", "agent_access", "guardrails", "agent_skills"] = Field(
        ..., description="The specific field name that needs correction."
    )
    corrected_value: Union[str, List[str]] = Field(
        ..., description="The updated and corrected content for only that field."
    )

class AgentState(TypedDict):
    agent_card: dict
    research_context: str
    blueprint: Blueprint
    validation_feedback: str
    revision_count: int
    is_valid: bool
    approval_status: str

class TaskDefinition(BaseModel):
    name: str
    description: str
    is_async: bool
    requires_human_approval: bool

class BackendConfig(BaseModel):
    redis_required: bool
    postgre_required: bool
    vector_store: str

class AgentCard(BaseModel):
    name: str
    description: str
    specialization: dict
    capabilities: dict
    tasks: dict
    backend: BackendConfig
    config: dict

# --- Helper: VS Code Editor Launcher ---
def edit_blueprint_in_vscode(current_blueprint: Blueprint) -> Blueprint:
    """Writes blueprint to temp file as an array, launches VS Code with --wait, and re-validates."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "agent_factory_blueprint_edit.json")
    
    # 1. Wrap the dictionary in a list to maintain the [{}, {}] format
    data_to_edit = [current_blueprint.model_dump()]
    
    with open(temp_path, "w", encoding="utf-8") as f:
        # Use json.dump for clean, standard indentation that VS Code natively respects
        json.dump(data_to_edit, f, indent=4)
        
    print(f"\n>>> 📝 Opening blueprint in VS Code: {temp_path}")
    print(">>> ⚠️  Save the file and close the tab in VS Code to proceed...")

    while True:
        try:
            subprocess.run(["code", "--wait", temp_path], check=True, shell=True)
            
            with open(temp_path, "r", encoding="utf-8") as f:
                edited_json = json.load(f)
            
            # 2. Safely extract the edited dictionary back out of the list
            if isinstance(edited_json, list) and len(edited_json) > 0:
                edited_dict = edited_json[0]
            else:
                # Fallback just in case the list brackets were accidentally deleted in VS Code
                edited_dict = edited_json
                
            validated_blueprint = Blueprint.model_validate(edited_dict)
            print(">>> ✅ Changes validated successfully against schema.")
            return validated_blueprint

        except json.JSONDecodeError as jde:
            print(f"\n❌ JSON Syntax Error: {jde}")
            retry = input("Re-open file in VS Code to fix syntax? (y/n): ").strip().lower()
            if retry != "y":
                print("Reverting to previous unedited version.")
                return current_blueprint

        except ValidationError as ve:
            print(f"\n❌ Schema Validation Error: {ve}")
            retry = input("Re-open file in VS Code to fix schema fields? (y/n): ").strip().lower()
            if retry != "y":
                print("Reverting to previous unedited version.")
                return current_blueprint

        except FileNotFoundError:
            print("❌ VS Code executable ('code') not found in PATH. Falling back to default notepad.")
            subprocess.run(["notepad", temp_path], check=True)


# --- 2. Node Functions ---
def research_node(state: AgentState):
    # # app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

    # # search_result = app.search("latest news on quantum computing", limit=5)

    # print(">>> 🔍 Researching SDLC patterns via Firecrawl...")
    # app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    # query = f"SDLC automation strategy for {state['agent_description']} handling {state['task_description']}"
    
    # # Retrieve top web context
    # search_results = app.search(query=query)


    # context = "\n\n".join([
    #     getattr(item, 'markdown', 
    #         getattr(item, 'content', 
    #             getattr(item, 'description', 'No content found')
    #         )
    #     ) or '' # The 'or' handles cases where the attribute exists but is None
    #     for item in search_results.web
    # ])

    # with open("FireCrawl Artifacts/search_result.txt", "a", encoding="utf-8") as f:
    #     # Convert the dictionary to a JSON string
    #     json_string = json.dumps(context, ensure_ascii=False, indent=4)
    #     # Write the string followed by a newline
    #     f.write(json_string + "\n")

    # # This prints the first search result with all its available attributes
    # print(search_results.web[0].model_dump_json(indent=4))
    
    return {"research_context": "Sample research context for SDLC automation strategy."}

def breaker_node(state: AgentState):
    print(">>> 🧠 Generating blueprint from agent card and research...")
    llm = get_llm(ACTIVE_PROVIDER).with_structured_output(Blueprint)
    card = state["agent_card"]

    # Extract specific domains from the card for the prompt
    task_names = [t["name"] for t in card["tasks"]["task"]]
    tech_stack = card["capabilities"]["frameworks"] + card["capabilities"]["platforms"]

    # 1. Conditionally build the correction directive
    feedback = state.get("validation_feedback", "").strip()
    validation_feedback_block = ""
    if feedback:
        validation_feedback_block = (
            f"\n\n⚠️ PREVIOUS ATTEMPT REJECTED BY QA:\n"
            f"Critique: {feedback}\n"
            f"Action Required: Directly rectify these defects in the generated specification."
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Agent Architect and Technical Lead. Your task is to take a concise Agent Card and expand it into a highly descriptive, comprehensive, and production-ready architectural blueprint. 
        
        CRITICAL INSTRUCTIONS:
        - DO NOT merely parrot or copy keywords from the provided context. 
        - EXPAND every single concept. Use descriptive, full-sentence formulations for all list items.
        - Provide both high-level system design intent and low-level technical implementation specifics.
        - Imagine you are writing an exhaustive technical specification to instruct a human developer on exactly how to build this agent from scratch. Eliminate ambiguity to prevent downstream hallucinations.
        - For every tool, access permission, or skill, you MUST explain WHY it is needed and HOW it will be implemented in the context of the agent's objective."""),
        
        ("user", """
        Expand the following structural data into a fully realized descriptive blueprint:
        
        Agent Name: {name}
        Core Objective: {description}
        Required Tasks: {tasks}
        Tech Stack & Capabilities: {tech_stack}
        Backend Requirements: {backend}
        Research Context: {research_context}{validation_feedback_block}
        """)
    ])

    blueprint = (prompt | llm).invoke({
        "name": card["name"],
        "description": card["description"],
        "tasks": ", ".join(task_names),
        "tech_stack": ", ".join(tech_stack),
        "backend": str(card["backend"]),
        "research_context": state.get("research_context", ""),
        "validation_feedback_block": validation_feedback_block
    })
    
    return {"blueprint": blueprint}

def validator_node(state: AgentState):
    print(">>> ⚖️ Validating blueprint against Agent Card constraints...")
    llm = get_llm(ACTIVE_PROVIDER)

    # Pass both the generated blueprint and the original card to the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict QA Compliance Agent. Compare the generated architectural blueprint against the source Agent Card. Ensure NO unauthorized tools or databases were added, and ALL tasks are accounted for. If it passes, return 'VALID'. If it fails, return a specific list of corrections."),
        ("user", "Agent Card: {card}\n\nGenerated Blueprint: {blueprint}")
    ])

    response = (prompt | llm).invoke({
        "card": state["agent_card"],
        "blueprint": state["blueprint"].model_dump_json()
    })

    feedback = response.content.strip()
    is_valid = "VALID" in feedback.upper()

    return {
        "validation_feedback": feedback if not is_valid else "",
        "is_valid": is_valid,
        "revision_count": state.get("revision_count", 0) + 1
    }

def patch_node(state: AgentState):
    #This cuts output token generation during retries by 70% to 90% because the model only generates a single string or list rather than re-emitting the entire document.
    print(f">>> 🩹 Surgically patching field: {state['failed_field']}...")
    llm = get_llm(ACTIVE_PROVIDER).with_structured_output(BlueprintPatch)
    
    current_value = getattr(state["blueprint"], state["failed_field"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a surgical code/spec editor. Fix ONLY the requested field based on the validator's critique. Keep the rest unchanged."),
        ("user", """
        Current Field: {field}
        Current Content: {content}
        Validation Feedback: {feedback}
        
        Return the corrected value complying with all rules.
        """)
    ])
    
    patch = (prompt | llm).invoke({
        "field": state["failed_field"],
        "content": current_value,
        "feedback": state["validation_feedback"]
    })
    
    # Mutate only the targeted field on the existing blueprint
    updated_blueprint = state["blueprint"].model_copy(
        update={patch.field_to_update: patch.corrected_value}
    )
    
    return {
        "blueprint": updated_blueprint,
        "revision_count": state.get("revision_count", 0) + 1
    }

def route_validation(state: AgentState):
    # Route to human review if valid, OR if we hit the max retry limit to prevent infinite loops
    if state["is_valid"] or state["revision_count"] >= 3:
        return "human_review"

    print(f"\n>>> 🔄 Revision required. Feedback: {state['validation_feedback']}")
    return "breaker"

def human_review_node(state: AgentState):
    current_blueprint = state["blueprint"]

    while True:
        print("\n" + "="*60)
        print("📋 BLUEPRINT OVERVIEW - HITL CHECKPOINT 1")
        print("="*60)
        print(current_blueprint.model_dump_json(indent=2))
        
        action = input("\nOptions: [y] Approve | [edit] Edit in VS Code | [n] Abort: ").strip().lower()
        
        if action == "y":
            # Save the approved blueprint to an artifact output file
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "blueprint_output.json"
            
            if output_file.exists() and output_file.stat().st_size > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    try:
                        data_list = json.load(f)
                        # Ensure the root is a list
                        if not isinstance(data_list, list):
                            data_list = [data_list]
                    except json.JSONDecodeError:
                        # If file is corrupted, start fresh
                        data_list = []
            else:
                data_list = []
                
            data_list.append(current_blueprint.model_dump())
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2)
                
            print(f"\n>>> 💾 Approved blueprint appended to: {output_file.resolve()}")
            print(">>> ✅ Proceeding to Phase 2 (Tool Generation & Testing)...")
            return {"blueprint": current_blueprint, "approval_status": "approved"}
            
        elif action == "edit":
            current_blueprint = edit_blueprint_in_vscode(current_blueprint)
            
        elif action == "n":
            print(">>> ❌ Pipeline aborted by user.")
            return {"approval_status": "rejected"}
            
        else:
            print("Invalid input. Please enter 'y', 'edit', or 'n'.")

# --- 3. Graph Orchestration ---
workflow = StateGraph(AgentState)

workflow.add_node("research", research_node)
workflow.add_node("breaker", breaker_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("validator", validator_node)

workflow.add_edge(START, "research")

workflow.add_edge("research", "breaker")
workflow.add_edge("breaker", "validator")
workflow.add_conditional_edges(
    "validator",
    route_validation,
    {"human_review": "human_review", "breaker": "breaker"} #Ends loop or restarts
)
workflow.add_edge("human_review", END)

factory_app = workflow.compile()

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Load the specific agent card
    with open("Sample Agent Card/Agent_Card.json", "r", encoding="utf-8") as f:
        card_data = json.load(f)
        
    initial_state = {
        "agent_card": card_data
    }

    factory_app.invoke(initial_state)
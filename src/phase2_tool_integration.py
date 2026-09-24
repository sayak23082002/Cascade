import os
import json
import tempfile
import subprocess
import datetime
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv, set_key
# from langchain_openai import AzureChatOpenAI
from src.Provider.LLM_Provider import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables (Enables LangSmith automatically if configured)
load_dotenv(override=True)

os.makedirs("logs", exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs/phase2_run_{run_id}.json"

# Initialize empty log file
with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump([], f)

def log_event(event_type: str, details: dict):
    """Appends an event to the persistent JSON log file."""
    with open(LOG_FILE, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        })
        f.seek(0)
        json.dump(data, f, indent=2)

# --- 1. Schemas & State ---
class ToolIntegrationState(TypedDict):
    tool_name: str
    tool_description: str
    target_domain: str # e.g., 'database', 'aws', 'github' for federated routing
    existing_mcp_server: Optional[str]
    tool_code: Optional[str]
    eval_passed: bool
    eval_logs: str
    approval_status: str
    required_env_vars: list[str]
    env_var_guidelines: list[dict]
    iteration_count: int
    feedback: str
    missing_dependencies: list[str]
    function_name: Optional[str]
    instruction_manual: Optional[str]
    test_parameters: list[dict]
    test_inputs: Dict[str, Any]

class TestParameter(BaseModel):
    name: str = Field(..., description="The parameter name (e.g., repo_url).")
    description: str = Field(..., description="Description so the user knows what to input.")

class EnvVarGuideline(BaseModel):
    var_name: str = Field(..., description="The exact name of the environment variable (e.g., GITHUB_ACCESS_TOKEN).")
    description: str = Field(..., description="What this input is and why the tool needs it.")
    how_to_get: str = Field(..., description="Step-by-step instructions on where the user can find or generate this key.")

class GeneratedToolCode(BaseModel):
    code: str = Field(..., description="The raw Python code for the tool.")
    function_name: str = Field(..., description="The exact name of the Python function defined with @mcp.tool().")
    required_env_vars: list[str] = Field(..., description="List of environment variable keys required by this tool.")
    env_var_guidelines: list[EnvVarGuideline] = Field(default_factory=list, description="Detailed guidelines for each required environment variable.")
    instruction_manual: str = Field(..., description="A clear instruction manual explaining tool usage.")
    test_parameters: list[TestParameter] = Field(..., description="Parameters required to test the function.")

class EvalFeedback(BaseModel):
    feedback: str = Field(..., description="Specific instructions on how to fix the Python code based on the error traceback.")
    missing_dependencies: list[str] = Field(default_factory=list, description="List of missing pip packages identified in the traceback to be installed (e.g., ['mcp', 'pydantic']).")


# llm = AzureChatOpenAI(
#         deployment_name=os.getenv("DEPLOYMENT_NAME"),
#         api_version=os.getenv("API_VERSION"),
#         azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#         api_key=os.getenv("API_KEY")
#     )

ACTIVE_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

llm = get_llm(ACTIVE_PROVIDER)


def query_mcp_router_for_tool(tool_name: str, domain: str) -> Optional[str]:
    """
    Simulates the tools/list JSON-RPC protocol against a Federated MCP Router.
    In a live environment, this would use the official `mcp` Python SDK client 
    (via stdio or SSE) to broadcast a tools/list request to the domain server.
    """
    print(f">>> 🌐 Querying Federated Router (Domain: {domain}) for '{tool_name}' via tools/list...")
    
    # Simulated Registry for demonstration. 
    # Replace with actual MCP Client tools/list JSON-RPC call.
    mock_mcp_registry = {
        "database": ["sql_query_executor", "schema_analyzer"],
        "aws": ["s3_bucket_reader"],
        "core": ["web_scraper", "file_reader"]
    }

    available_tools = mock_mcp_registry.get(domain, [])
    if tool_name in available_tools:
        # Simulate returning the MCP server URL for the tool
        return f"mcp_server_{domain}"
    return None

# --- 2. Node Functions ---
def check_existing_tool_node(state: ToolIntegrationState):
    """Step 1: Agent checks physical local directory for existing MCP server tools."""
    domain = state["target_domain"]
    tool_name = state["tool_name"]
    
    print(f">>> 🌐 Checking local MCP Registry for '{tool_name}' in domain '{domain}'...")
    server_path = Path(f"mcp_server/{domain}/{tool_name}.py")
    
    if server_path.exists():
        print(f">>> ✅ Tool '{tool_name}' found locally at {server_path}. Skipping generation.")
        return {"existing_mcp_server": str(server_path), "eval_passed": True}

    print(f">>> ❌ Tool '{tool_name}' not found. Routing to Builder.")
    return {"existing_mcp_server": None}

def build_tool_node(state: ToolIntegrationState):
    """Step 2: Construct the tool from scratch using Azure OpenAI."""
    current_iteration = state.get("iteration_count", 0) + 1
    print(f">>> 🏗️ Building tool '{state['tool_name']}' (Attempt {current_iteration})...")
    
    
    structured_llm = llm.with_structured_output(GeneratedToolCode)

    feedback_context = f"\n\nPREVIOUS ERROR FEEDBACK:\n{state['feedback']}" if state.get("feedback") else ""
    previous_code = f"\n\nPREVIOUS CODE:\n{state.get('tool_code')}" if state.get("tool_code") else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python Tool Engineer. Write a complete, standalone Python function.
        Requirements:
        1. Use the FastMCP SDK properly (e.g., `from mcp.server.fastmcp import FastMCP` and define `mcp = FastMCP("test")` then `@mcp.tool()`).
        2. MCP SERVER HOSTING: Make sure the file acts as an executable MCP server by including this exactly at the bottom:
           if __name__ == "__main__":
               mcp.run()
        3. Include all necessary imports.
        4. API KEYS & SECRETS: DO NOT make API keys, tokens, or passwords function parameters. You MUST use `os.getenv("YOUR_VAR_NAME")` inside the function logic.
        5. ENV VAR REGISTRATION: Any environment variable you use must be added to the `required_env_vars` output array.
        6. Do not include markdown code block backticks in the string output.{feedback}
        7. If previous code is provided, fix the specific issues mentioned in the feedback. Do not rewrite from scratch unless necessary.{feedback}{previous_code}"""),
        ("user", "Tool Name: {tool_name}\nDescription: {tool_description}\nDomain: {target_domain}")
    ])

    result = (prompt | structured_llm).invoke({
        "tool_name": state["tool_name"],
        "tool_description": state["tool_description"],
        "target_domain": state["target_domain"],
        "feedback": feedback_context,
        "previous_code": previous_code
    })

    # Forcefully append the MCP server execution block if the LLM omitted it
    final_code = result.code
    if "mcp.run()" not in final_code:
        final_code += '\n\nif __name__ == "__main__":\n    mcp.run()\n'

    return {
        "tool_code": final_code, # Use the validated code
        "function_name": result.function_name,
        "required_env_vars": result.required_env_vars,
        "instruction_manual": result.instruction_manual,
        "test_parameters": [p.model_dump() for p in result.test_parameters],
        "eval_passed": False,
        "iteration_count": current_iteration,
        "feedback": feedback_context
    }

def eval_agent_node(state: ToolIntegrationState):
    """Analyzes the traceback and gives exact feedback to the Builder."""
    print(">>> 🔍 Eval Agent analyzing error logs...")
    
    structured_llm = llm.with_structured_output(EvalFeedback)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python debugging agent. Analyze the broken code and the terminal traceback. Provide explicit instructions to the developer agent on how to fix the code."),
        ("user", "BROKEN CODE:\n{tool_code}\n\nTRACEBACK:\n{eval_logs}")
    ])
    
    result = (prompt | structured_llm).invoke({
        "tool_code": state["tool_code"],
        "eval_logs": state["eval_logs"]
    })

    log_event("eval_feedback_generated", {
        "feedback": result.feedback,
        "missing_dependencies": result.missing_dependencies
    })
    
    return {
        "feedback": result.feedback, 
        "missing_dependencies": result.missing_dependencies
    }

def install_dependencies_node(state: ToolIntegrationState):
    """Dynamically installs missing packages using uv in the conda environment."""
    deps = state.get("missing_dependencies", [])
    if deps:
        # Check current conda environment name
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
        if env_name == "agentfactory":
            print(f">>> 📦 Installing missing dependencies via uv: {', '.join(deps)}")
            try:
                subprocess.run(["uv", "pip", "install"] + deps, check=True)
                print(">>> ✅ Dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f">>> ❌ Failed to install dependencies: {e}")
        else:
            print(f">>> ⚠️ Conda env is '{env_name}', expected 'agentfactory'. Skipping auto-install.")
    
    return state

def setup_env_node(state: ToolIntegrationState):
    """Step 2.5: Set env vars and gather functional test inputs."""
    # Print the manual
    if state.get("instruction_manual"):
        print("\n" + "="*50 + "\n📖 INSTRUCTION MANUAL\n" + "="*50)
        print(state["instruction_manual"])

    # if state.get("required_env_vars"):
    #     for var in state["required_env_vars"]:
    #         if not os.getenv(var):
    #             print(f"\n🔐 TOOL REQUIRES CONFIGURATION: {var}")
    #             val = input(f"Please provide value for '{var}': ").strip()
    #             os.environ[var] = val
    #             print(f"[{var}] injected into isolated execution memory (no duplicate file created)")

    if state.get("env_var_guidelines"):
        print("\n" + "-"*50)
        print("🔐 REQUIRED ENVIRONMENT VARIABLES")
        print("-"*50)
        for guide in state["env_var_guidelines"]:
            var = guide['var_name']
            if not os.getenv(var):
                print(f"\n🔑 {var}")
                print(f"   Why: {guide['description']}")
                print(f"   How to get it: {guide['how_to_get']}")
                val = input(f"\n   Please provide value for '{var}': ").strip()
                os.environ[var] = val
                print(f"   [{var}] injected into isolated execution memory.")

    test_inputs = state.get("test_inputs", {})
    
    # NEW: Detect if the LLM changed parameter names during a retry loop
    current_param_names = {p['name'] for p in state.get("test_parameters", [])}
    if test_inputs and set(test_inputs.keys()) != current_param_names:
        print("\n⚠️ LLM altered parameter names during retry. Please re-enter test inputs.")
        test_inputs = {} # Clear inputs to trigger the prompt below

    # Ask for functional test parameters
    if state.get("test_parameters") and not test_inputs:
        print("\n" + "-"*50)
        print("🧪 REQUIRED TEST INPUTS FOR FUNCTIONAL EVALUATION")
        print("-"*50)
        for param in state["test_parameters"]:
            val = input(f"Test value for '{param['name']}' ({param['description']}): ").strip()
            test_inputs[param['name']] = val

    return {"test_inputs": test_inputs}

def eval_tool_node(state: ToolIntegrationState):
    """Step 3: Run high-level tests on the generated tool in the Anaconda sandbox."""
    print(">>> 🧪 Evaluating tool in isolated Anaconda sandbox...")
    
    # Write code to a temporary file
    temp_dir = tempfile.gettempdir()
    # 1. Save the generated MCP server file
    tool_script_path = os.path.join(temp_dir, f"{state['tool_name']}.py")
    with open(tool_script_path, "w", encoding="utf-8") as f:
        f.write(state["tool_code"])

    # 2. Build the function arguments safely
    test_inputs = state.get("test_inputs", {})
    kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in test_inputs.items()])
    func_name = state["function_name"]

    # 3. Create a runner script that imports the tool to avoid triggering mcp.run()
    runner_script = f"""
import sys
from {state['tool_name']} import {func_name}

if __name__ == '__main__':
    try:
        result = {func_name}({kwargs_str})
        print(result)
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
"""
    runner_path = os.path.join(temp_dir, f"run_test_{state['tool_name']}.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner_script)

    try:
        # We must pass PYTHONPATH so the runner can find the generated tool module
        env = os.environ.copy()
        env["PYTHONPATH"] = temp_dir
        
        result = subprocess.run(["python", runner_path], capture_output=True, text=True, check=True, env=env)
        print(">>> ✅ Functional evaluation passed.\n", result.stdout)
        log_event("tool_execution_succes", {"command": f"python {runner_path}", "output": result.stdout})
        return {"eval_passed": True, "eval_logs": result.stdout}

    except subprocess.CalledProcessError as e:
        full_error = e.stderr + "\n" + e.stdout
        print(f">>> ❌ Evaluation failed. Routing to Eval Agent...")
        return {"eval_passed": False, "eval_logs": full_error}

    finally:
        # Safely delete the temporary test runner script after execution
        if os.path.exists(runner_path):
            os.remove(runner_path)

def hitl_config_node(state: ToolIntegrationState):
    """Step 4: Prompt for API keys and provide an isolated CLI test sandbox."""
    # 4a. Check Environment Variables
    if state.get("required_env_vars"):
        print("\n" + "="*50)
        print("🔐 REQUIRED ENVIRONMENT VARIABLES DETECTED")
        print("="*50)
        for var in state["required_env_vars"]:
            if not os.getenv(var):
                val = input(f"Please provide value for '{var}': ").strip()
                # Save to local .env securely
                set_key(".env", var, val)
                os.environ[var] = val
                print(f"[{var}] injected into .env")
    
    # 4b. Isolated Sandbox Test Loop
    print("\n" + "="*50)
    print(f"🛠️  ISOLATED SANDBOX: {state['tool_name']}")
    print("="*50)
    print("You can now review the code or approve it for MCP registration.")
    
    while True:
        action = input("\nOptions: [view] View Code | [approve] Register to MCP | [reject] Abort: ").strip().lower()
        if action == "view":
            print("\n--- GENERATED CODE ---")
            print(state["tool_code"])
            print("----------------------")
        elif action == "approve":
            return {"approval_status": "approved"}
        elif action == "reject":
            return {"approval_status": "rejected"}
        else:
            print("Invalid input.")

def register_tool_node(state: ToolIntegrationState):
    """Final Step: Saves the tool into the Federated MCP Domain directory for hot-reloading."""
    domain = state["target_domain"]

    registry_path = Path(f"mcp_server/{domain}")
    registry_path.mkdir(parents=True, exist_ok=True)

    file_path = registry_path / f"{state['tool_name']}.py"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(state["tool_code"])

    print(f"\n>>> 💾 Tool successfully registered to MCP Server at: {file_path}")
    return state

# --- 3. Edge Routing Logic ---
def route_after_discovery(state: ToolIntegrationState):
    if state["existing_mcp_server"]:
        return "hitl_config"
    return "build_tool"

def route_after_eval(state: ToolIntegrationState):
    MAX_RETRIES = 3
    if state["eval_passed"]:
        return "hitl_config"
    elif state.get("iteration_count", 0) >= MAX_RETRIES:
        print(f">>> 🛑 Rate Limiter Triggered: Max iterations ({MAX_RETRIES}) reached. Routing to HITL.")
        return "hitl_config" # Force manual review if it keeps failing
    return "eval_agent" # Route to the new debugger node

def route_after_hitl(state: ToolIntegrationState):
    if state["approval_status"] == "approved" and state.get("tool_code"):
        return "register_tool"
    return END


# --- 4. Graph Orchestration ---
workflow = StateGraph(ToolIntegrationState)

# Add the new nodes
workflow.add_node("check_existing_tool", check_existing_tool_node)
workflow.add_node("build_tool", build_tool_node)
workflow.add_node("setup_env", setup_env_node)
workflow.add_node("eval_tool", eval_tool_node)
workflow.add_node("eval_agent", eval_agent_node)
workflow.add_node("install_dependencies", install_dependencies_node)
workflow.add_node("hitl_config", hitl_config_node)
workflow.add_node("register_tool", register_tool_node)

workflow.add_edge(START, "check_existing_tool")

workflow.add_conditional_edges("check_existing_tool", route_after_discovery)

# Insert setup_env between builder and evaluator
workflow.add_edge("build_tool", "setup_env")
workflow.add_edge("setup_env", "eval_tool")

workflow.add_conditional_edges("eval_tool", route_after_eval)

# Insert the dependency installer after the eval agent, then loop back to builder
workflow.add_edge("eval_agent", "install_dependencies")
workflow.add_edge("install_dependencies", "build_tool")

workflow.add_conditional_edges("hitl_config", route_after_hitl)
workflow.add_edge("register_tool", END)

tool_factory_app = workflow.compile()

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Example state driven by the Phase 1 Blueprint's "agent_and_tool_dist"
    initial_state = {
        "tool_name": "website_scrapper",
        "tool_description": "Scrapes a specified website and extracts relevant information. Store the info in a file in the same folder. The tool should use the FastMCP SDK and be executable as an MCP server.",
        "target_domain": "web_scraping",
        "existing_mcp_server": None,
        "tool_code": None,
        "function_name": None,
        "instruction_manual": None,
        "test_parameters": [],
        "test_inputs": {},
        "eval_passed": False,
        "eval_logs": "",
        "approval_status": "pending",
        "required_env_vars": [],
        "iteration_count": 0,
        "feedback": "",
        "missing_dependencies": []
    }

    tool_factory_app.invoke(initial_state)
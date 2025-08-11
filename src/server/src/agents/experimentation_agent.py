# agents/experimentation_agent.py
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from src.modal.modal_service import modal_runner

# Load environment variables and set Modal credentials
load_dotenv()
if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
    os.environ["MODAL_TOKEN_ID"] = os.environ.get("MODAL_TOKEN_ID")
    os.environ["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN_SECRET")

def run(executable_code: Dict) -> Dict[str, Any]:
    """
    Executes a script on Modal cloud infrastructure, captures output, and reports results or structured errors.
    """
    print("INFO: [Experimentation Agent] Running experiment script on Modal...")
    
    script_content = executable_code.get("script", "")
    if not script_content:
        return {"status": "failed", "error_type": "CODE_GENERATION_FAILED", "error_log": "No script provided."}

    try:
        # Extract any additional configuration from the executable_code
        config = executable_code.get("config", {})
        
        # Run the experiment on Modal
        result = modal_runner.run_experiment(script_content, config)
        
        # Return the result directly (Modal service handles all the execution details)
        return result
            
    except Exception as e:
        return {
            "status": "failed", 
            "error_type": "EXECUTION_ENVIRONMENT_ERROR", 
            "error_log": f"Failed to execute experiment on Modal: {str(e)}"
        }
# agents/implementation_agent.py
import json
from typing import Dict, Any

def run(client, experiment_plan: Dict, model_to_implement: Dict, model_name: str) -> Dict[str, Any]:
    """
    Generates the full, executable Python script for a single model experiment.
    Assumes the experiment_plan is already validated by the orchestrator.
    """
    print(f"INFO: [Implementation Agent] Generating code for model: {model_to_implement['model_id']}...")
    
    system_prompt = """
    You are an expert PyTorch engineer. Your task is to write a complete, standalone Python script to run a machine learning experiment based on a provided plan.

    CRITICAL INSTRUCTIONS:
    1. The script must be fully self-contained and runnable.
    2. Import all necessary libraries (torch, torchvision, scikit-learn, etc.).
    3. The script MUST define the model class, load the specified dataset, and contain the full training and evaluation loop.
    4. Add clear print logging to console so the experimenter can track what's going on.
    5. AT THE VERY END of the script, it MUST print a single JSON object to standard output containing the final evaluation metrics. This is how the system will capture the results. The JSON object should look like: {"status": "completed", "metrics": {"accuracy": 0.91, ...}}. Do not print anything else after this JSON object.
    6. Handle both training and validation/testing loops correctly.
    """

    # This context can now be built directly, without .get(), because the plan is guaranteed to be valid.
    prompt_context = {
        "dataset": experiment_plan["dataset"],
        "training_parameters": experiment_plan["training_parameters"],
        "evaluation_metrics": experiment_plan["evaluation_metrics"],
        "model_to_implement": model_to_implement
    }
    
    user_prompt = f"""
    Generate the complete PyTorch script for the following experiment plan.
    The script should only implement the model specified in `model_to_implement`.

    Experiment Plan Details:
    {json.dumps(prompt_context, indent=2)}
    """

    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        code = message
        if "```python" in message:
            # Extract code from markdown block
            code = message.split("```python\n")[1].split("```")[0]
        
        return {"script": code}
    except Exception as e:
        print(f"ERROR: [Implementation Agent] Failed to generate code. {e}")
        return {"script": None}
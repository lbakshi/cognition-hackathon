# agents/strategy_agent.py
import json
from typing import Dict, Any, Optional

def extract_json_from_response(response_text: str) -> str:
    # This helper function is correct and stays the same
    try:
        json_start_index = response_text.find('{')
        json_end_index = response_text.rfind('}')
        if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
            return response_text[json_start_index:json_end_index+1]
    except Exception:
        pass
    return response_text

def run(client, conceptual_plan: Dict, target_framework: str, model_name: str, previous_error: Optional[Dict] = None) -> Dict[str, Any]:
    if previous_error:
        print(f"INFO: [Strategy Agent] Revising plan based on error: {previous_error.get('error_type')}")
    else:
        print("INFO: [Strategy Agent] Creating new experiment plan...")

    # --- THIS IS THE UPDATED PROMPT ---
    system_prompt = f"""
    You are an expert AI research strategist specializing in {target_framework}.
    Your task is to create a detailed, structured experiment plan in JSON format.

    **IT IS MANDATORY THAT THE OUTPUT JSON OBJECT FOLLOWS THIS EXACT SCHEMA. DO NOT DEVIATE. EVERY KEY SHOWN IN THE EXAMPLE IS REQUIRED. DO NOT ADD EXTRA KEYS.**

    ```json
    {{
      "experiment_id": "a_unique_string_id",
      "experiment_name": "A descriptive name for the experiment",
      "hypothesis": "A clear statement about what is being tested",
      "dataset": {{ "name": "e.g., CIFAR-10 or IMDB", "source": "e.g., torchvision.datasets.CIFAR10", "validation_split": 0.2 }},
      "training_parameters": {{ "optimizer": "e.g., Adam", "learning_rate": 0.001, "loss_function": "e.g., CrossEntropyLoss", "batch_size": 64, "epochs": 10 }},
      "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
      "models": {{
        "candidate": {{ "model_id": "candidate_model_name", "description": "...", "architecture": [{{ "type": "NameOfLayer", "params": {{...}} }}] }},
        "baselines": [ {{ "model_id": "baseline_model_name", "description": "...", "architecture": [{{ "type": "NameOfLayer", "params": {{...}} }}] }} ]
      }}
    }}
    ```

    The output must be ONLY the JSON object.

    If you receive a `previous_error` object, your main goal is to modify the plan to fix it.
    - If error_type is 'SCHEMA_VALIDATION_ERROR', the 'error_log' contains a list of missing or malformed keys. You MUST fix the JSON structure to match the mandatory schema exactly. This is your highest priority.
    - If error_type is 'CUDA_OUT_OF_MEMORY', significantly reduce the 'batch_size'.
    - If error_type is 'DIMENSION_MISMATCH', carefully correct the layer features.
    """

    error_context = ""
    if previous_error:
        error_context = f"""
        ---
        REVISION CONTEXT: The last experiment failed. You MUST modify the plan to fix this error.
        Error Type: {previous_error['error_type']}
        Error Log: {previous_error['error_log']}
        ---
        """

    user_prompt = f"""
    Conceptual Plan:
    {json.dumps(conceptual_plan, indent=2)}
    {error_context}

    Generate the complete, runnable experiment plan as a single JSON object adhering to the mandatory schema.
    """
    
    try:
        message_text = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        json_string = extract_json_from_response(message_text)
        return json.loads(json_string)

    except Exception as e:
        raw_response = locals().get('message_text', 'No response from API.')
        print(f"ERROR: [Strategy Agent] Failed to create a plan. {e}\n--- RAW RESPONSE ---\n{raw_response}")
        return {}
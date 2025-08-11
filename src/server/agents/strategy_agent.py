# agents/strategy_agent.py
import json
from typing import Dict, Any, Optional

def extract_json_from_response(response_text: str) -> str:
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

    # --- NEW, FLEXIBLE PROMPT FOR MULTIPLE TASK TYPES ---
    system_prompt = f"""
    You are an expert AI research strategist. Your task is to create a detailed, structured experiment plan in JSON format based on a conceptual plan.
    You must identify the type of experiment and use the correct JSON schema.

    **CRITICAL:** For any experiment involving iterative training or optimization, you MUST include a `time_series_metrics` key. This key should list the names of important metrics to track at each epoch or iteration (e.g., "training_loss", "validation_accuracy").

    **TYPE 1: Standard Classification/Regression Experiment**
    If the goal is to train and evaluate a model, use this schema. Propose a candidate and at least one baseline.
    ```json
    {{
      "experiment_type": "classification",
      "experiment_id": "...",
      "experiment_name": "...",
      "hypothesis": "...",
      "dataset": {{ "name": "...", "source": "..." }},
      "training_parameters": {{ "optimizer": "...", "learning_rate": 0.001, ... }},
      "evaluation_metrics": ["accuracy", "f1_score", ...],
      "time_series_metrics": ["training_loss", "validation_accuracy"],
      "models": {{
        "candidate": {{ "model_id": "...", "architecture": [...] }},
        "baselines": [ {{ "model_id": "...", "architecture": [...] }} ]
      }}
    }}
    ```

    **TYPE 2: Model Inversion Attack Experiment**
    If the goal is to reconstruct a class prototype from a trained model, use this specialized schema.
    ```json
    {{
      "experiment_type": "model_inversion_attack",
      "experiment_id": "...",
      "experiment_name": "...",
      "hypothesis": "It is possible to reconstruct a class prototype using gradient-based optimization on a trained model.",
      "dataset": {{ "name": "...", "source": "..." }},
      "victim_model_architecture": [
          {{ "type": "Linear", "params": {{...}} }},
          {{ "type": "ReLU" }},
          {{ "type": "Linear", "params": {{...}} }}
      ],
      "attack_parameters": {{
        "target_class_name": "The name of the class to reconstruct",
        "target_class_index": "The integer index of the target class",
        "optimizer": "Adam",
        "learning_rate": 0.1,
        "iterations": 500
      }},
      "time_series_metrics": ["confidence_score"],
      "evaluation_metrics": ["final_confidence_score", "generated_feature_vector"]
    }}
    ```

    The output must be ONLY the JSON object.
    If you receive a `previous_error`, your main goal is to modify the plan to fix it.
    """

    error_context = ""
    if previous_error:
        error_context = f"REVISION CONTEXT: The last experiment failed. Fix this error: {previous_error}"

    user_prompt = f"""
    Conceptual Plan:
    {json.dumps(conceptual_plan, indent=2)}
    {error_context}

    Generate the complete, runnable experiment plan as a single JSON object, choosing the correct schema based on the conceptual plan's task.
    """
    
    try:
        message_text = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        json_string = extract_json_from_response(message_text)
        result = json.loads(json_string)
        return result

    except Exception as e:
        raw_response = locals().get('message_text', 'No response from API.')
        print(f"ERROR: [Strategy Agent] Failed to create a plan. {e}\n--- RAW RESPONSE ---\n{raw_response}")
        return {}
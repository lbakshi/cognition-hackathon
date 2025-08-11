# agents/implementation_agent.py
import json
from typing import Dict, Any

def run(client, experiment_plan: Dict, model_spec: Dict, model_name: str) -> Dict[str, Any]:
    """
    Generates the full, executable Python script.
    It now intelligently handles different experiment types based on the plan.
    """
    experiment_type = experiment_plan.get("experiment_type", "classification")
    print(f"INFO: [Implementation Agent] Generating code for task type: '{experiment_type}'")
    print(f"INFO: [Implementation Agent] Target model/procedure: '{model_spec.get('model_id', 'attack_procedure')}'")

    # --- NEW, FLEXIBLE PROMPT ---
    system_prompt = """
    You are an expert AI engineer who writes clean, simple, and runnable Python scripts for advanced machine learning experiments.
    Your task is to write a complete, standalone Python script based on the provided plan. You must handle both standard training and complex, non-standard experiments like optimization attacks.

    **CRITICAL INSTRUCTIONS:**
    1.  **Interpret the Goal:** Carefully analyze the `experiment_type`.
        -   If `experiment_type` is **"classification"**, generate a script with a `nn.Module` class for the `model_to_implement`, a standard training loop, an optimizer, and a final evaluation.
        -   If `experiment_type` is **"model_inversion_attack"**, the generated script MUST have two parts:
            1.  **Part 1: Train Victim:** Define and quickly train a "victim" classifier using the `victim_model_architecture`.
            2.  **Part 2: Attack Loop:** Freeze the victim model. Write an optimization loop (using e.g., `torch.optim.Adam`) that iteratively modifies an *input tensor* to maximize the output probability for the `target_class_index`. The loss should be the *negative* of the target class probability.
    2.  **Data Loading:** Use datasets from `sklearn.datasets`. The script should load data and perform a standard train-test split.
    3.  **Verbose Logging:** The script MUST include print statements to track progress. For an attack, this means printing the optimization progress (e.g., "Iteration 100/500 | Target Confidence: 0.98").
    4.  **Time-Series Logging:** During any iterative process (like training epochs or optimization steps), you MUST collect the values for each metric listed in the `time_series_metrics` plan key. Store them in lists.
    5.  **Final JSON Output:** AT THE VERY END of the script, it MUST print a single JSON object. This object MUST contain two top-level keys:
        -   `"metrics"`: An object containing the final, single-value scalar metrics (e.g., final test accuracy).
        -   `"time_series_data"`: An object where each key is a metric name from the plan. The value for each key MUST be another object with two keys: `"X"` (a list of the steps, e.g., epochs `[1, 2, 3...]`) and `"Y"` (a list of the corresponding metric values `[0.85, 0.91, 0.93...]`).

    **Example Final JSON structure:**
    ```json
    {
        "status": "completed",
        "metrics": { "final_accuracy": 0.98 },
        "time_series_data": {
            "training_loss": { "X":, "Y": [0.5, 0.3, 0.1] },
            "validation_accuracy": { "X":, "Y": [0.92, 0.95, 0.98] }
        }
    }
    ```
    **Do not print absolutely anything after this final JSON object.**
    """

    # We now pass the entire plan, as the agent needs full context for complex tasks.
    user_prompt = f"""
    Generate the complete, runnable Python script for the following experiment plan.
    For a 'classification' task, implement the model specified in the `model_to_implement` variable.
    For a 'model_inversion_attack' task, implement the full two-part procedure described in the plan.

    Full Experiment Plan:
    {json.dumps(experiment_plan, indent=2)}

    Model to Implement (for classification tasks only):
    {json.dumps(model_spec, indent=2)}
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
            code = message.split("```python\n")[1].split("```")[0]
        
        return {"script": code}
    except Exception as e:
        print(f"ERROR: [Implementation Agent] Failed to generate code. {e}")
        return {"script": None}
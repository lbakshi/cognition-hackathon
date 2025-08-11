# agents/analysis_agent.py
import json
from typing import Dict, Any, List

def run(client, experiment_results: List[Dict], experiment_plan: Dict, model_name: str) -> str:
    """
    Analyzes all experimental results and generates a final markdown report.
    It now handles different types of experiments.
    """
    print("INFO: [Analysis Agent] Synthesizing final report...")

    # --- NEW, FLEXIBLE PROMPT ---
    system_prompt = """
    You are a principal research scientist. Your task is to write a concise and clear summary report in Markdown format based on a set of experiment results and the original plan.

    **CRITICAL INSTRUCTIONS:**
    1.  Start with the original `hypothesis` from the plan.
    2.  Check the `experiment_type` from the plan.
    3.  If the type is **"classification"** and there are multiple results, create a markdown table to compare the key metrics of all tested models.
    4.  If the type is **"model_inversion_attack"** or there is only one result, clearly present the key findings and metrics from the result. Do not use a table if it's not a comparison.
    5.  Write a "Conclusion" section where you state whether the results support the hypothesis and provide a brief justification based on the data.
    """

    user_prompt = f"""
    Original Experiment Plan:
    {json.dumps(experiment_plan, indent=2)}

    ---
    Experiment Results:
    {json.dumps(experiment_results, indent=2)}
    ---

    Generate the final markdown report based on the instructions.
    """

    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        return message
    except Exception as e:
        print(f"ERROR: [Analysis Agent] Failed to generate report. {e}")
        return "Report generation failed."
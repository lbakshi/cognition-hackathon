# agents/analysis_agent.py
from typing import Dict, Any, List

def run(client, experiment_results: List[Dict], original_hypothesis: str, model_name: str) -> str:
    """
    Analyzes all experimental results and generates a final markdown report.
    """
    print("INFO: [Analysis Agent] Synthesizing final report...")

    system_prompt = """
    You are a principal research scientist. Your task is to write a concise and clear summary report based on a set of experiment results.
    - Start with the original hypothesis.
    - Create a markdown table to compare the key metrics of all tested models.
    - Write a "Conclusion" section where you state whether the results support or refute the hypothesis and provide a brief justification.
    """

    user_prompt = f"""
    Original Hypothesis:
    {original_hypothesis}

    ---
    Experiment Results:
    {json.dumps(experiment_results, indent=2)}
    ---

    Generate the final markdown report.
    """

    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content.text
        
        return message
    except Exception as e:
        print(f"ERROR: [Analysis Agent] Failed to generate report. {e}")
        return "Report generation failed."
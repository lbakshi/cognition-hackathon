# agents/evaluator_agent.py
import json
from typing import Dict, Any

def extract_json_from_response(response_text: str) -> str:
    try:
        json_start_index = response_text.find('{')
        json_end_index = response_text.rfind('}')
        if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
            return response_text[json_start_index:json_end_index+1]
    except Exception:
        pass
    return response_text

def run(client, hypothesis: str, execution_results: Dict, model_name: str) -> Dict[str, Any]:
    """
    Analyzes the results of a completed run to check for scientific validity and logical errors.
    """
    print("INFO: [Evaluator Agent] Peer reviewing experiment results...")

    system_prompt = """
    You are an expert AI research scientist acting as a peer reviewer.
    Your task is to analyze the results of a completed experiment to determine if they are scientifically valid or if they indicate a hidden logical flaw, even if the code ran without crashing.

    Follow this structured thought process:
    1.  **Understand the Goal:** Read the `hypothesis`. What was the experiment trying to achieve?
    2.  **Examine the Results:** Look at the `metrics` from the experiment. Are they reasonable?
    3.  **Critically Evaluate:** Compare the goal to the results. Ask critical questions.
        -   If the goal was to maximize something, did it actually increase? A final confidence of 0.0 or a training accuracy of 10% is a failure.
        -   Are the output values within a realistic range? An unscaled feature vector of `[50, -50, ...]` for the Iris dataset is a clear sign of unconstrained optimization, a logical failure.
        -   Does the result align with basic principles, or does it seem nonsensical?
    4.  **Formulate a Verdict:**
        -   If the results are reasonable and align with the hypothesis (even if the hypothesis is ultimately proven false by the numbers), set `verdict` to "VALIDATED".
        -   If the results are nonsensical and indicate a flaw in the experimental code, set `verdict` to "LOGICAL_FAILURE".
    5.  **Provide Rationale:** In the `rationale` field, explain your decision clearly. If it's a failure, this rationale will be passed to the debugger.

    You MUST output a single JSON object with the following structure:
    {
        "verdict": "VALIDATED" or "LOGICAL_FAILURE",
        "rationale": "Your detailed analysis and reason for the verdict."
    }
    """

    user_prompt = f"""
    **Hypothesis (The Goal):**
    {hypothesis}

    **Execution Results (The Output):**
    ```json
    {json.dumps(execution_results, indent=2)}
    ```

    Please perform your peer review and provide your verdict as a single JSON object.
    """

    try:
        message_text = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        json_string = extract_json_from_response(message_text)
        return json.loads(json_string)
        
    except Exception as e:
        raw_response = locals().get('message_text', 'No response from API.')
        print(f"ERROR: [Evaluator Agent] Failed to process. {e}\n--- RAW RESPONSE ---\n{raw_response}")
        return {}
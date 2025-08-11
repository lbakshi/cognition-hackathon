# agents/debugger_agent.py
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

def run(client, hypothesis: str, original_code: str, execution_log: str, model_name: str) -> Dict[str, Any]:
    """
    Analyzes failed experiment results, diagnoses the code, and suggests a fix.
    """
    print("INFO: [Debugger Agent] Analyzing failed experiment...")

    system_prompt = """
    You are an expert AI researcher and Python programmer, acting as a code debugger.
    Your task is to analyze a failed experiment, diagnose the root cause, and propose a fix.
    The failure might be a code crash, or a "logical failure" where the code ran but the results are nonsensical (e.g., a model not learning, confidence being zero).

    Follow this structured thought process:
    1.  **Analyze the Goal:** Read the `hypothesis`. What was the code *supposed* to accomplish?
    2.  **Analyze the Failure:** Read the `execution_log`. What actually happened? Did it crash? Were the metrics nonsensical?
    3.  **Diagnose the Root Cause:** Connect the goal to the failure. State your diagnosis clearly. For example, "The goal was to maximize confidence, but it remained at 0.0. The log shows the generated features exploded to unrealistic values. The root cause is unconstrained gradient ascent."
    4.  **Propose a Fix:** Describe the specific code change you will make. For example, "I will add an L2 regularization term to the loss function to penalize large feature values and keep the optimization within a realistic range."
    5.  **Decide the Next Step:**
        -   If you believe the code can be fixed, set `decision` to "MODIFY_CODE".
        -   If you believe the code is correct but the underlying scientific plan is flawed, set `decision` to "ESCALATE_TO_STRATEGIST".
    6.  **Provide the Code:** If your decision is "MODIFY_CODE", provide the complete, corrected code in the `modified_code` field.

    You MUST output a single JSON object with the following structure:
    {
        "analysis": "Your detailed analysis of the problem.",
        "decision": "MODIFY_CODE" or "ESCALATE_TO_STRATEGIST",
        "modified_code": "The full, corrected Python script, or an empty string."
    }
    """

    user_prompt = f"""
    **Hypothesis (The Goal):**
    {hypothesis}

    **Execution Log (The Failure):**
    ```
    {execution_log}
    ```

    **Original Code that Produced this Failure:**
    ```python
    {original_code}
    ```

    Please perform the debugging analysis and provide your response as a single JSON object.
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
        print(f"ERROR: [Debugger Agent] Failed to process. {e}\n--- RAW RESPONSE ---\n{raw_response}")
        return {}
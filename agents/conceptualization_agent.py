# agents/conceptualization_agent.py
import json
from typing import Dict, Any

def extract_json_from_response(response_text: str) -> str:
    """Finds and extracts a JSON object from a string."""
    try:
        # Find the first opening curly brace
        json_start_index = response_text.find('{')
        # Find the last closing curly brace
        json_end_index = response_text.rfind('}')
        
        if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
            # Slice the string to get the JSON block
            json_string = response_text[json_start_index:json_end_index+1]
            return json_string
    except Exception:
        # Fallback or in case of errors
        pass
    
    # Return the original text if we can't find a JSON object
    return response_text

def run(client, research_idea: str, model_name: str) -> Dict[str, Any]:
    """
    Deconstructs the user's idea, identifies the domain, task, and novelty.
    """
    print("INFO: [Conceptualization Agent] Deconstructing research idea...")
    
    system_prompt = """
    You are a research scientist. Your task is to analyze a user's research idea and deconstruct it into its fundamental components.
    Identify the core research domain (e.g., Computer Vision, NLP), the specific task (e.g., Image Classification, Sentiment Analysis), and the core novelty or technique being proposed.
    Provide a brief summary of the state-of-the-art context based on your internal knowledge.
    
    Output a JSON object with the following structure:
    {
      "domain": "string",
      "task": "string",
      "core_novelty": ["list", "of", "strings"],
      "sota_context": "string"
    }
    """
    
    user_prompt = f"Analyze the following research idea:\n\n---\n{research_idea}\n---"

    try:
        message_text = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ).content[0].text
        
        # Use the robust extraction function
        json_string = extract_json_from_response(message_text)
        return json.loads(json_string)
        
    except Exception as e:
        # The error message now includes the raw response for easier debugging
        raw_response = locals().get('message_text', 'No response from API.')
        print(f"ERROR: [Conceptualization Agent] Failed to process idea. {e}\n--- RAW RESPONSE ---\n{raw_response}")
        return {}
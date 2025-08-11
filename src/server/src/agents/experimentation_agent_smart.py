# Smart experimentation agent with dynamic Modal configuration and self-healing
import json
import os
import time
import traceback
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class SmartExperimentationAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.modal_credentials_set = False
        self._ensure_modal_credentials()
    
    def _ensure_modal_credentials(self):
        """Ensure Modal credentials are properly set"""
        if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
            os.environ["MODAL_TOKEN_ID"] = os.environ.get("MODAL_TOKEN_ID")
            os.environ["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN_SECRET")
            self.modal_credentials_set = True
        else:
            print("WARNING: Modal credentials not found in environment")
    
    def _get_modal_config_suggestions(self, script_content: str, error_context: str = None) -> Dict[str, Any]:
        """Use GPT-4 to analyze script and suggest optimal Modal configuration"""
        
        system_prompt = """You are a Modal cloud infrastructure expert. Analyze the provided Python script and suggest optimal Modal configuration.

Consider:
1. GPU requirements (based on model complexity, training data size)
2. Memory requirements (based on model size, batch size, data loading)
3. Timeout requirements (based on training complexity)
4. Required Python packages (infer from imports and code)

Return a JSON object with this structure:
{
    "gpu_config": "any" | "T4" | "A10G" | "A100" | "H100" | null,
    "memory_gb": number (4-64),
    "timeout_seconds": number (300-7200),
    "required_packages": ["package1", "package2", ...],
    "reasoning": "explanation of choices"
}"""

        user_prompt = f"""Analyze this Python script for optimal Modal configuration:

```python
{script_content}
```

{f"Previous error context: {error_context}" if error_context else ""}

Provide configuration recommendations."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "modal_config",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "gpu_config": {
                                    "type": "string",
                                    "enum": ["any", "T4", "A10G", "A100", "H100", "null"],
                                    "description": "GPU type needed"
                                },
                                "memory_gb": {
                                    "type": "integer",
                                    "minimum": 4,
                                    "maximum": 64,
                                    "description": "Memory in GB"
                                },
                                "timeout_seconds": {
                                    "type": "integer",
                                    "minimum": 300,
                                    "maximum": 7200,
                                    "description": "Timeout in seconds"
                                },
                                "required_packages": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Python packages needed"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Explanation of configuration choices"
                                }
                            },
                            "required": ["gpu_config", "memory_gb", "timeout_seconds", "required_packages", "reasoning"],
                            "additionalProperties": false
                        }
                    }
                }
            )
            
            # Parse the JSON response
            config = json.loads(response.choices[0].message.content)
            print(f"GPT-5 Modal Config: {config.get('reasoning', 'No reasoning provided')}")
            return config
            
        except Exception as e:
            print(f"Error getting GPT-4 config suggestions: {e}")
        
        # Fallback default config
        return {
            "gpu_config": "any",
            "memory_gb": 8,
            "timeout_seconds": 1800,
            "required_packages": ["torch", "torchvision", "scikit-learn", "numpy", "pandas"],
            "reasoning": "Default fallback configuration"
        }
    
    def _create_dynamic_modal_app(self, config: Dict[str, Any], script_content: str, experiment_config: Dict[str, Any]) -> Any:
        """Create a Modal app with dynamic configuration"""
        try:
            import modal
            from pathlib import Path
            
            app_name = f"experiment-{int(time.time())}"
            app = modal.App(app_name)
            
            # Build image with required packages
            image = modal.Image.debian_slim().pip_install(*config["required_packages"])
            
            # Create volume for artifacts
            volume = modal.Volume.from_name("experiment-artifacts", create_if_missing=True)
            
            # Configure GPU
            gpu_config = None if config["gpu_config"] == "null" else config["gpu_config"]
            
            @app.function(
                image=image,
                volumes={'/artifacts': volume},
                gpu=gpu_config,
                timeout=config["timeout_seconds"],
                memory=config["memory_gb"] * 1024,  # Convert GB to MB
            )
            def run_experiment_dynamic(script_content: str, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
                """Dynamically created Modal function"""
                import torch
                import numpy as np
                import json
                import time
                from pathlib import Path
                from datetime import datetime
                
                # Create artifacts directory
                artifacts_dir = Path(f"/artifacts/{job_id}")
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                
                # Save config
                with open(artifacts_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)
                
                try:
                    # Create execution namespace
                    namespace = {
                        'config': config,
                        'artifacts_dir': str(artifacts_dir),
                        'results': {},
                        'torch': torch,
                        'np': np,
                        'numpy': np,
                        'time': time,
                        'json': json,
                        'Path': Path,
                    }
                    
                    # Import additional modules based on requirements
                    try:
                        import sklearn
                        namespace['sklearn'] = sklearn
                    except ImportError:
                        pass
                    
                    try:
                        import pandas as pd
                        namespace['pd'] = pd
                        namespace['pandas'] = pd
                    except ImportError:
                        pass
                    
                    # Execute the script
                    exec(script_content, namespace)
                    
                    # Get results
                    results = namespace.get('results', {})
                    
                    # Save results
                    with open(artifacts_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2)
                    
                    # Commit volume changes
                    volume.commit()
                    
                    return {
                        'status': 'success',
                        'results': results,
                        'artifacts_path': str(artifacts_dir)
                    }
                    
                except Exception as e:
                    error_info = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now().isoformat(),
                        'traceback': traceback.format_exc()
                    }
                    
                    with open(artifacts_dir / "error.json", "w") as f:
                        json.dump(error_info, f, indent=2)
                    
                    return {
                        'status': 'error',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'artifacts_path': str(artifacts_dir),
                        'traceback': traceback.format_exc()
                    }
            
            return app, run_experiment_dynamic
            
        except Exception as e:
            print(f"Error creating Modal app: {e}")
            raise
    
    def _analyze_failure_and_retry(self, error_info: Dict[str, Any], script_content: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Use GPT-4 to analyze failure and suggest retry strategy"""
        
        system_prompt = """You are an expert at debugging Modal cloud execution failures. Analyze the error and suggest fixes.

Common issues and solutions:
1. Out of memory -> Increase memory_gb, optimize batch size
2. Timeout -> Increase timeout_seconds, optimize algorithm
3. Missing packages -> Add to required_packages
4. GPU issues -> Change gpu_config or use CPU
5. Code errors -> Suggest code fixes

Return a JSON object with:
{
    "should_retry": boolean,
    "config_changes": {
        "gpu_config": "...", 
        "memory_gb": number,
        "timeout_seconds": number,
        "required_packages": [...]
    },
    "script_changes": "suggested code modifications or null",
    "reasoning": "explanation"
}"""

        user_prompt = f"""Analysis failure from attempt {attempt}:

Error: {error_info.get('error', 'Unknown error')}
Error Type: {error_info.get('error_type', 'Unknown')}
Traceback: {error_info.get('traceback', 'None')}

Original script:
```python
{script_content}
```

Suggest retry strategy."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "failure_analysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_retry": {"type": "boolean"},
                                "config_changes": {
                                    "type": "object",
                                    "properties": {
                                        "gpu_config": {"type": "string"},
                                        "memory_gb": {"type": "integer"},
                                        "timeout_seconds": {"type": "integer"},
                                        "required_packages": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "additionalProperties": false
                                },
                                "script_changes": {"type": ["string", "null"]},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["should_retry", "reasoning"],
                            "additionalProperties": false
                        }
                    }
                }
            )
            
            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)
            print(f"GPT-5 Failure Analysis: {analysis.get('reasoning', 'No reasoning provided')}")
            return analysis
            
        except Exception as e:
            print(f"Error analyzing failure: {e}")
        
        return {"should_retry": False, "reasoning": "Could not analyze failure"}
    
    def run_experiment(self, script_content: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run experiment with smart Modal configuration and retry logic"""
        
        if not self.modal_credentials_set:
            return {
                "status": "failed",
                "error_type": "CONFIGURATION_ERROR",
                "error_log": "Modal credentials not configured"
            }
        
        max_attempts = 3
        job_id = f"job_{int(time.time())}"
        
        for attempt in range(max_attempts):
            try:
                print(f"INFO: [Smart Experimentation Agent] Attempt {attempt + 1}/{max_attempts}")
                
                # Get or update Modal configuration
                if attempt == 0:
                    modal_config = self._get_modal_config_suggestions(script_content)
                else:
                    # Use previous error to refine config
                    error_context = last_result.get('error', '') if 'last_result' in locals() else None
                    modal_config = self._get_modal_config_suggestions(script_content, error_context)
                
                print(f"INFO: [Smart Agent] Using config: GPU={modal_config['gpu_config']}, Memory={modal_config['memory_gb']}GB")
                
                # Create dynamic Modal app
                app, run_func = self._create_dynamic_modal_app(modal_config, script_content, config or {})
                
                # Execute on Modal
                with app.run():
                    result = run_func.remote(script_content, config or {}, f"{job_id}_attempt_{attempt}")
                
                if result['status'] == 'success':
                    print("INFO: [Smart Agent] Experiment completed successfully")
                    return {
                        "status": "completed",
                        "results": result['results'],
                        "job_id": job_id,
                        "attempts": attempt + 1,
                        "modal_config": modal_config
                    }
                else:
                    print(f"ERROR: [Smart Agent] Attempt {attempt + 1} failed: {result.get('error')}")
                    last_result = result
                    
                    if attempt < max_attempts - 1:
                        # Analyze failure and decide if we should retry
                        analysis = self._analyze_failure_and_retry(result, script_content, attempt + 1)
                        
                        if not analysis.get('should_retry', False):
                            break
                        
                        # Apply suggested changes for next attempt
                        if 'config_changes' in analysis:
                            modal_config.update(analysis['config_changes'])
                        
                        print(f"INFO: [Smart Agent] Retrying with modifications: {analysis['reasoning']}")
                        time.sleep(2)  # Brief pause before retry
                    
            except Exception as e:
                print(f"ERROR: [Smart Agent] Attempt {attempt + 1} exception: {str(e)}")
                last_result = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
                
                if attempt == max_attempts - 1:
                    break
                    
                time.sleep(2)
        
        # All attempts failed
        return {
            "status": "failed",
            "error_type": "MAX_RETRIES_EXCEEDED",
            "error_log": f"Failed after {max_attempts} attempts. Last error: {last_result.get('error', 'Unknown')}",
            "job_id": job_id,
            "attempts": max_attempts,
            "last_error": last_result
        }

# Global instance
smart_agent = SmartExperimentationAgent()

def run(executable_code: Dict) -> Dict[str, Any]:
    """Main entry point for smart experimentation agent"""
    print("INFO: [Smart Experimentation Agent] Starting intelligent Modal execution...")
    
    script_content = executable_code.get("script", "")
    if not script_content:
        return {"status": "failed", "error_type": "CODE_GENERATION_FAILED", "error_log": "No script provided."}
    
    config = executable_code.get("config", {})
    return smart_agent.run_experiment(script_content, config)
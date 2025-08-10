# agents/experimentation_agent.py
import subprocess
import json
from typing import Dict, Any

def run(executable_code: Dict) -> Dict[str, Any]:
    """
    Executes a script in a subprocess, captures output, and reports results or structured errors.
    """
    print("INFO: [Experimentation Agent] Running experiment script...")
    
    script_content = executable_code.get("script", "")
    if not script_content:
        return {"status": "failed", "error_type": "CODE_GENERATION_FAILED", "error_log": "No script provided."}

    try:
        # Write the script to a temporary file
        with open("temp_experiment.py", "w") as f:
            f.write(script_content)

        # Execute the script
        result = subprocess.run(
            ["python", "temp_experiment.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30-minute timeout
        )

        # Check for successful execution
        if result.returncode == 0:
            print("INFO: [Experimentation Agent] Experiment completed successfully.")
            # The last line of stdout should be our results JSON
            last_line = result.stdout.strip().split('\n')[-1]
            return json.loads(last_line)
        else:
            # Execution failed, analyze the error
            print("ERROR: [Experimentation Agent] Experiment failed.")
            stderr = result.stderr
            error_type = "UNKNOWN_RUNTIME_ERROR"
            if "CUDA out of memory" in stderr:
                error_type = "CUDA_OUT_OF_MEMORY"
            elif "dimension mismatch" in stderr or "shape" in stderr and "mismatch" in stderr:
                error_type = "DIMENSION_MISMATCH"
            elif "ModuleNotFoundError" in stderr:
                error_type = "MODULE_NOT_FOUND"

            return {
                "status": "failed",
                "error_type": error_type,
                "error_log": stderr
            }
            
    except json.JSONDecodeError:
        return {"status": "failed", "error_type": "JSON_DECODE_ERROR", "error_log": "Script did not output a valid JSON object.", "stdout": result.stdout}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error_type": "TIMEOUT_EXPIRED", "error_log": "Experiment took too long to run."}
    except Exception as e:
        return {"status": "failed", "error_type": "EXECUTION_ENVIRONMENT_ERROR", "error_log": str(e)}
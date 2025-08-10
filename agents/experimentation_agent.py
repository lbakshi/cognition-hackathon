# agents/experimentation_agent.py
import subprocess
import json
import os
from typing import Dict, Any

def run(executable_code: Dict) -> Dict[str, Any]:
    """
    Executes a script in a subprocess, streams its output to the console in real-time,
    and reports results or structured errors.
    """
    print("INFO: [Experimentation Agent] Starting experiment execution...")
    
    script_content = executable_code.get("script", "")
    if not script_content:
        return {"status": "failed", "error_type": "CODE_GENERATION_FAILED", "error_log": "No script provided by Implementation Agent."}

    temp_script_path = "temp_experiment.py"
    try:
        # Write the script to a temporary file
        with open(temp_script_path, "w") as f:
            f.write(script_content)

        # Use Popen for real-time output streaming
        # - stdout=subprocess.PIPE lets us read the output.
        # - stderr=subprocess.STDOUT merges error messages into the standard output stream.
        # - text=True decodes the output as text.
        # - bufsize=1 enables line-buffering.
        process = subprocess.Popen(
            ["python", "-u", temp_script_path], # The "-u" flag forces unbuffered output from Python
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Store all output lines while also printing them
        output_lines = []
        print("--- Script Output (Live) ---")
        for line in process.stdout:
            print(line, end='')  # Print line to console in real-time
            output_lines.append(line)
        print("--- End of Script Output ---")

        # Wait for the process to complete and get the return code
        process.wait()
        returncode = process.returncode
        full_output = "".join(output_lines)

        # Check for successful execution
        if returncode == 0:
            print("INFO: [Experimentation Agent] Experiment script completed successfully.")
            # The last line of stdout should be our results JSON
            try:
                last_line = full_output.strip().split('\n')[-1]
                return json.loads(last_line)
            except (json.JSONDecodeError, IndexError):
                 return {"status": "failed", "error_type": "JSON_DECODE_ERROR", "error_log": "Script completed but did not output a valid JSON object as its last line.", "stdout": full_output}
        
        else: # Execution failed, analyze the error
            print(f"ERROR: [Experimentation Agent] Experiment script failed with return code {returncode}.")
            error_type = "UNKNOWN_RUNTIME_ERROR"
            if "CUDA out of memory" in full_output:
                error_type = "CUDA_OUT_OF_MEMORY"
            elif "dimension mismatch" in full_output or "shape" in full_output and "mismatch" in full_output:
                error_type = "DIMENSION_MISMATCH"
            elif "ModuleNotFoundError" in full_output:
                error_type = "MODULE_NOT_FOUND"

            return {
                "status": "failed",
                "error_type": error_type,
                "error_log": full_output
            }
            
    except Exception as e:
        return {"status": "failed", "error_type": "EXECUTION_ENVIRONMENT_ERROR", "error_log": str(e)}
    finally:
        # Clean up the temporary script file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
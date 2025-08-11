# orchestrator.py
import os
import json
import anthropic
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from agents import (
    conceptualization_agent, 
    strategy_agent, 
    implementation_agent, 
    experimentation_agent, 
    analysis_agent, 
    debugger_agent, 
    evaluator_agent
)

load_dotenv()
MODEL_NAME = "claude-opus-4-1-20250805"
TARGET_FRAMEWORK = "pytorch"
MAX_DEBUG_RETRIES = 4

# This is now the main entry point, a callable function
def main(research_idea: str, job_id: str) -> Optional[Dict[str, str]]:
    """
    The main orchestration function, now callable and returning result paths.
    """
    print(f"--- [Job ID: {job_id}] AI Research Lab Task Started ---")
    
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        print(f"FATAL [Job ID: {job_id}]: ANTHROPIC_API_KEY not found.")
        return None

    # Step 1: Conceptualization
    conceptual_plan = conceptualization_agent.run(client, research_idea, MODEL_NAME)
    if not conceptual_plan:
        print(f"FATAL [Job ID: {job_id}]: Conceptualization failed. Aborting.")
        return None
    print(f"INFO [Job ID: {job_id}]: [Conceptualization] Domain='{conceptual_plan.get('domain', 'N/A')}'")

    # Step 2: Strategy
    experiment_plan = strategy_agent.run(client, conceptual_plan, TARGET_FRAMEWORK, MODEL_NAME)
    if not experiment_plan:
        print(f"FATAL [Job ID: {job_id}]: Strategy failed. Aborting.")
        return None

    # Step 3: Setup Directory using job_id for uniqueness
    exp_dir = f"experiments/{job_id}_{experiment_plan.get('experiment_name', 'unnamed').replace(' ', '_')}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/initial_plan.json", 'w') as f:
        json.dump(experiment_plan, f, indent=2)
    print(f"INFO [Job ID: {job_id}]: Plan saved to '{exp_dir}/initial_plan.json'")

    # --- EXECUTION -> EVALUATION -> DEBUGGING LOOP ---
    final_result = None
    model_spec = experiment_plan
    current_code_str = None

    for attempt in range(MAX_DEBUG_RETRIES + 1):
        # ... (The entire loop from your existing orchestrator goes here, unchanged)
        # Just add the job_id to the print statements for better logging, e.g.,
        print(f"\n--- [Job ID: {job_id}] Execution Cycle: Attempt {attempt + 1}/{MAX_DEBUG_RETRIES + 1} ---")

        if current_code_str is None:
            print("INFO: [Orchestrator] Calling Implementation Agent to generate initial code...")
            code_obj = implementation_agent.run(client, experiment_plan, model_spec, MODEL_NAME)
            current_code_str = code_obj.get("script")

        if not current_code_str:
            print("FATAL: Code generation failed. Aborting.")
            break
        
        # Save the current version of the code for this attempt
        script_path = os.path.join(exp_dir, f"attempt_{attempt+1}_script.py")
        with open(script_path, 'w') as f: f.write(current_code_str)
        print(f"INFO: [Orchestrator] Code for attempt {attempt+1} saved to '{script_path}'")
        
        # Step 6b: Execute the Code
        exec_result = experimentation_agent.run({"script": current_code_str})
        # ... (The rest of the loop logic is identical) ...
        # ... (if success, break, if failure, call debugger, etc.)
        if exec_result.get("status") == "COMPLETED_SUCCESSFULLY":
            print("INFO: [Orchestrator] Execution completed. Calling Evaluator Agent for peer review...")
            evaluation = evaluator_agent.run(
                client=client,
                hypothesis=experiment_plan.get('hypothesis'),
                execution_results=exec_result.get('results'),
                model_name=MODEL_NAME
            )
            print(f"INFO: [Evaluator Agent] Verdict: {evaluation.get('verdict')}")
            print(f"INFO: [Evaluator Agent] Rationale: {evaluation.get('rationale')}")
            if evaluation.get("verdict") == "VALIDATED":
                print("SUCCESS: [Orchestrator] Experiment validated by Evaluator.")
                final_result = exec_result.get('results')
                final_result['procedure_id'] = experiment_plan.get('experiment_id')
                break
            else:  # Logical Failure found by Evaluator
                print("ERROR: [Orchestrator] Evaluator found a logical failure in the results.")
                # We will now fall through to the debugger, passing the evaluator's rationale as the error log
                exec_result['status'] = 'LOGICAL_FAILURE'
                exec_result['error_log'] = evaluation.get('rationale')
            # If we've reached max retries, the loop will terminate after this block
        if attempt >= MAX_DEBUG_RETRIES:
            print("FATAL: Maximum debugging retries reached. Aborting.")
            break

        print("INFO: [Orchestrator] Calling Debugger Agent to fix the code...")
        debug_result = debugger_agent.run(
            client=client,
            hypothesis=experiment_plan.get('hypothesis'),
            original_code=current_code_str,
            execution_log=exec_result.get('error_log', 'No log available.'),
            model_name=MODEL_NAME
        )

        print(f"INFO: [Debugger Agent] Analysis: {debug_result.get('analysis')}")
        
        if debug_result.get("decision") == "MODIFY_CODE" and debug_result.get("modified_code"):
            print("INFO: [Orchestrator] Debugger provided a code modification. Retrying with new code...")
            current_code_str = debug_result.get("modified_code") # The loop will now use this new code
        else:
            print("FATAL: [Orchestrator] Debugger could not fix the code or decided to escalate. Aborting experiment.")
            break # Exit loop if debugger gives up

    # --- 7. FINAL ANALYSIS ---
    if not final_result:
        print(f"--- [Job ID: {job_id}] Research Incomplete ---")
        return None
    else:
        print(f"--- [Job ID: {job_id}] Research Complete: Finalizing ---")
        
        # We need to add the procedure_id to the final_result here
        final_result['procedure_id'] = experiment_plan.get('experiment_id')
        
        analysis_report_payload = [{"metrics": final_result.get('metrics'), "time_series_data": final_result.get('time_series_data')}]
        final_report = analysis_agent.run(client, analysis_report_payload, experiment_plan, MODEL_NAME)
        
        report_path = os.path.join(exp_dir, "final_report.md")
        with open(report_path, "w") as f: f.write(final_report)
        print(f"INFO [Job ID: {job_id}]: Report saved to {report_path}")

        frontend_data = {}
        experiment_id = experiment_plan.get('experiment_id')
        if experiment_id:
            time_series_data = final_result.get('time_series_data')
            if time_series_data:
                frontend_data[experiment_id] = time_series_data
                progress_path = os.path.join(exp_dir, "progress_metrics.json")
                with open(progress_path, 'w') as f: json.dump(frontend_data, f, indent=2)
                print(f"SUCCESS [Job ID: {job_id}]: Time-series data saved to '{progress_path}'")
                
                # Return the paths to the generated artifacts
                return {
                    "report_path": report_path,
                    "progress_metrics_path": progress_path,
                    "final_plan_path": os.path.join(exp_dir, "initial_plan.json")
                }
    
    return None

# This block is no longer needed here, as the entry point is main.py
# if __name__ == "__main__":
#     main()
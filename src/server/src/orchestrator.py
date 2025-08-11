# orchestrator.py
import os
import json
import anthropic
from dotenv import load_dotenv
from typing import Dict, Any, List

# --- IMPORTANT: Import all agents, including the new Evaluator ---
from agents import (
    conceptualization_agent, 
    strategy_agent, 
    implementation_agent, 
    experimentation_agent, 
    analysis_agent, 
    debugger_agent, 
    evaluator_agent
)

# --- CONFIGURATION ---
load_dotenv()
MODEL_NAME = "claude-opus-4-1-20250805"  # Use the latest available model from Anthropic
TARGET_FRAMEWORK = "pytorch"
MAX_DEBUG_RETRIES = 2 # Max times the debugger can try to fix the code

def main():
    """
    The main orchestration loop with the new Evaluator-Debugger workflow.
    This system mimics the scientific process: Plan -> Execute -> Peer Review -> Debug.
    """
    print("--- AI Research Lab Initialized ---")
    
    # 1. Initialize API Client
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        print("FATAL ERROR: ANTHROPIC_API_KEY not found in .env file. Please create one.")
        return

    # 2. Get Research Idea from User
    research_idea = input("Please enter your research idea:\n> ")

    # 3. Conceptualization Step
    conceptual_plan = conceptualization_agent.run(client, research_idea, MODEL_NAME)
    if not conceptual_plan:
        print("Could not start research: Conceptualization failed. Aborting.")
        return
    print(f"INFO: [Conceptualization] Domain='{conceptual_plan.get('domain', 'N/A')}', Task='{conceptual_plan.get('task', 'N/A')}'")

    # 4. Strategy Step
    experiment_plan = strategy_agent.run(client, conceptual_plan, TARGET_FRAMEWORK, MODEL_NAME)
    if not experiment_plan:
        print("Could not create an experiment plan. Aborting.")
        return

    # 5. Setup Experiment Artifacts Directory
    exp_dir = f"experiments/{experiment_plan.get('experiment_name', 'unnamed_experiment').replace(' ', '_')}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/initial_plan.json", 'w') as f:
        json.dump(experiment_plan, f, indent=2)
    print(f"INFO: [Orchestrator] Experiment plan saved to '{exp_dir}/initial_plan.json'")

    # --- 6. EXECUTION -> EVALUATION -> DEBUGGING LOOP ---
    final_result = None
    model_spec = experiment_plan  # For single-procedure tasks like model inversion
    current_code_str = None

    for attempt in range(MAX_DEBUG_RETRIES + 1):
        print(f"\n--- [Orchestrator] Execution Cycle: Attempt {attempt + 1}/{MAX_DEBUG_RETRIES + 1} ---")

        # Step 6a: Generate Code (only on the first attempt or if missing)
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

        # Step 6c: Evaluate the Outcome
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
                break  # Exit the loop on a validated success!
            
            else:  # Logical Failure found by Evaluator
                print("ERROR: [Orchestrator] Evaluator found a logical failure in the results.")
                # We will now fall through to the debugger, passing the evaluator's rationale as the error log
                exec_result['status'] = 'LOGICAL_FAILURE'
                exec_result['error_log'] = evaluation.get('rationale')
        
        # --- Step 6d: Debugging Intervention (if not successful) ---
        # This block is reached if exec_result.status is RUNTIME_ERROR or LOGICAL_FAILURE
        
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
        print("\n--- Research Incomplete: No validated result was produced. ---")
    else:
        print("\n--- Research Complete: Generating Final Report ---")
        # Analysis agent expects a list of results, even if there's only one.
        final_report = analysis_agent.run(client, [final_result], experiment_plan, MODEL_NAME)
        print("\n" + "="*20 + " FINAL REPORT " + "="*20)
        print(final_report)
        print("="*54)
        
        report_path = os.path.join(exp_dir, "final_report.md")
        with open(report_path, "w") as f:
            f.write(final_report)
        print(f"Report saved to {report_path}")

        print("\n--- Aggregating Time-Series Data for Frontend ---")
        frontend_data = {}
        experiment_id = experiment_plan.get('experiment_id')
        
        if experiment_id:
            time_series_data = final_result.get('time_series_data')
            # time_series_data = final_result.get('results', {}).get('time_series_data')
            if time_series_data:
                frontend_data[experiment_id] = time_series_data
                
                # Save the aggregated data to a file
                progress_path = os.path.join(exp_dir, "progress_metrics.json")
                try:
                    with open(progress_path, 'w') as f:
                        json.dump(frontend_data, f, indent=2)
                    print(f"SUCCESS: Time-series data successfully saved to '{progress_path}'")
                except IOError as e:
                    print(f"ERROR: Could not save progress data file. {e}")
            else:
                print("INFO: No time-series data was found in the final validated result.")
        else:
            print("WARNING: Could not determine experiment_id to save progress data.")

if __name__ == "__main__":
    main()
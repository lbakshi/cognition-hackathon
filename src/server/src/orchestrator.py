# orchestrator.py
import os
import json
import anthropic
from dotenv import load_dotenv
from typing import Dict, Any, List

# Import agent functions
from agents import conceptualization_agent, strategy_agent, implementation_agent, experimentation_agent, analysis_agent

# --- CONFIGURATION ---
load_dotenv()
MODEL_NAME = "claude-opus-4-1-20250805"
TARGET_FRAMEWORK = "pytorch"
MAX_PLAN_RETRIES = 2 
MAX_EXEC_RETRIES = 1

def validate_plan(plan: Dict) -> List[str]:
    """
    Validates the structure of the experiment plan.
    Returns a list of error messages. An empty list means the plan is valid.
    """
    errors = []
    required_top_keys = ["experiment_id", "experiment_name", "hypothesis", "dataset", 
                           "training_parameters", "evaluation_metrics", "models"]
    for key in required_top_keys:
        if key not in plan:
            errors.append(f"Missing required top-level key: '{key}'")
    
    if "models" in plan:
        if not isinstance(plan["models"], dict):
            errors.append("'models' must be a dictionary.")
        else:
            if "candidate" not in plan["models"]:
                errors.append("Missing required key 'candidate' inside 'models'.")
            if "baselines" not in plan["models"]:
                errors.append("Missing required key 'baselines' inside 'models'.")
            elif not isinstance(plan["models"]["baselines"], list):
                 errors.append("'baselines' must be a list of model objects.")

    return errors


def main():
    """The main orchestration loop."""
    print("--- AI Research Lab Initialized ---")
    
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        print("FATAL ERROR: ANTHROPIC_API_KEY not found in .env file.")
        return

    research_idea = input("Please enter your research idea:\n> ")

    # 1. Conceptualization
    conceptual_plan = conceptualization_agent.run(client, research_idea, MODEL_NAME)
    if not conceptual_plan:
        print("Could not start research. Aborting.")
        return
    print(f"CONCEPT: Domain='{conceptual_plan.get('domain', 'N/A')}', Task='{conceptual_plan.get('task', 'N/A')}'")

    # 2. Strategy & Validation Loop
    experiment_plan = None
    last_error_log = None
    for attempt in range(MAX_PLAN_RETRIES + 1):
        print(f"\n--- [Orchestrator] Calling Strategy Agent (Attempt {attempt + 1}/{MAX_PLAN_RETRIES + 1}) ---")
        
        error_context = {"error_type": "SCHEMA_VALIDATION_ERROR", "error_log": last_error_log} if last_error_log else None
        
        plan_candidate = strategy_agent.run(client, conceptual_plan, TARGET_FRAMEWORK, MODEL_NAME, previous_error=error_context)
        
        if not plan_candidate:
            last_error_log = "Agent returned an empty plan."
            continue 

        validation_errors = validate_plan(plan_candidate)
        if not validation_errors:
            print("INFO: [Orchestrator] Plan validation successful!")
            experiment_plan = plan_candidate
            break 
        
        print("ERROR: [Orchestrator] Plan validation failed.")
        last_error_log = "\n".join(validation_errors)
        print(f"Validation Issues:\n{last_error_log}")

    if not experiment_plan:
        print("\nFATAL ERROR: Strategy Agent failed to produce a valid experiment plan. Aborting.")
        return

    exp_dir = f"experiments/{experiment_plan.get('experiment_name', 'unnamed_experiment').replace(' ', '_')}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/final_validated_plan.json", 'w') as f:
        json.dump(experiment_plan, f, indent=2)

    # 3. Execution Loop
    models_to_run = [experiment_plan["models"]["candidate"]] + experiment_plan["models"]["baselines"]
    all_results = []
    
    for model_spec in models_to_run:
        print(f"\n--- Starting Experiment for: {model_spec['model_id']} ---")
        current_plan = experiment_plan
        
        for exec_attempt in range(MAX_EXEC_RETRIES + 1):
            executable_code = implementation_agent.run(client, current_plan, model_spec, MODEL_NAME)
            
            # --- NEW CODE SAVING LOGIC ---
            script_content = executable_code.get("script")
            if not script_content:
                print(f"FATAL: Code generation failed for {model_spec['model_id']}. Skipping model.")
                break

            # Construct a descriptive filename
            model_id = model_spec['model_id']
            attempt_suffix = f"_attempt_{exec_attempt + 1}" if exec_attempt > 0 else ""
            script_filename = f"{model_id}{attempt_suffix}_script.py"
            script_path = os.path.join(exp_dir, script_filename)

            try:
                with open(script_path, 'w') as f:
                    f.write(script_content)
                print(f"INFO: [Orchestrator] Generated code saved to '{script_path}'")
            except IOError as e:
                print(f"WARNING: [Orchestrator] Could not save script to file. Error: {e}")
            # --- END OF NEW CODE SAVING LOGIC ---

            result = experimentation_agent.run(executable_code)

            if result["status"] == "completed":
                print(f"SUCCESS: Experiment for {model_spec['model_id']} completed.")
                result['model_id'] = model_spec['model_id']
                all_results.append(result)
                break
            
            else:
                print(f"ERROR: Execution Attempt {exec_attempt + 1}/{MAX_EXEC_RETRIES + 1} for {model_spec['model_id']} failed.")
                print(f"Error Type: {result.get('error_type')}")
                if exec_attempt >= MAX_EXEC_RETRIES:
                    print(f"FATAL: Max execution retries reached for {model_spec['model_id']}. Aborting this model.")
                    break

                print("INFO: Attempting to self-correct by revising the plan for runtime error...")
                revised_plan = strategy_agent.run(client, conceptual_plan, TARGET_FRAMEWORK, MODEL_NAME, previous_error=result)
                
                if not revised_plan or validate_plan(revised_plan):
                    print("FATAL: Failed to revise the plan for runtime error, or revision was invalid. Aborting this model.")
                    break
                
                current_plan = revised_plan
                print("INFO: Plan revised for runtime error. Retrying experiment.")
    
    # 4. Analysis
    if not all_results or len(all_results) < len(models_to_run):
        print("\n--- Research Incomplete ---")
        print("Not all models completed successfully. Final analysis cannot be performed.")
    else:
        print("\n--- Research Complete: Generating Final Report ---")
        final_report = analysis_agent.run(client, all_results, experiment_plan['hypothesis'], MODEL_NAME)
        print("\n" + "="*20 + " FINAL REPORT " + "="*20)
        print(final_report)
        print("="*54)
        
        report_path = os.path.join(exp_dir, "final_report.md")
        with open(report_path, "w") as f:
            f.write(final_report)
        print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
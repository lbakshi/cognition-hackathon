"""
Orchestrator that runs the experiment pipeline and communicates via API
"""
import os
import json
import asyncio
import httpx
import anthropic
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import agent functions
from src.agents import conceptualization_agent, strategy_agent, implementation_agent, experimentation_agent_smart as experimentation_agent, analysis_agent

load_dotenv()

MODEL_NAME = "claude-opus-4-1-20250805"
TARGET_FRAMEWORK = "pytorch"
MAX_PLAN_RETRIES = 2
MAX_EXEC_RETRIES = 1
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

class OrchestratorAPI:
    def __init__(self, experiment_id: str, query: str):
        self.experiment_id = experiment_id
        self.query = query
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def update_stage(self, stage: str, data: Dict[str, Any] = None):
        """Update experiment stage via API"""
        try:
            await self.http_client.post(
                f"{SERVER_URL}/api/subagent/update_stage",
                json={
                    "experiment_id": self.experiment_id,
                    "agent_type": stage,
                    "data": data or {}
                }
            )
        except Exception as e:
            print(f"Failed to update stage {stage}: {e}")
    
    async def store_file(self, filename: str, content: str):
        """Store a file via API"""
        try:
            await self.http_client.post(
                f"{SERVER_URL}/api/subagent/store_file",
                json={
                    "experiment_id": self.experiment_id,
                    "filename": filename,
                    "content": content
                }
            )
        except Exception as e:
            print(f"Failed to store file {filename}: {e}")
    
    async def update_kv(self, key: str, value: Any):
        """Update key-value pair via API"""
        try:
            await self.http_client.post(
                f"{SERVER_URL}/api/subagent/update_kv",
                json={
                    "experiment_id": self.experiment_id,
                    "key": key,
                    "value": value
                }
            )
        except Exception as e:
            print(f"Failed to update kv {key}: {e}")
    
    async def check_cancelled(self) -> bool:
        """Check if experiment was cancelled"""
        try:
            response = await self.http_client.get(
                f"{SERVER_URL}/api/poll/{self.experiment_id}/status"
            )
            data = response.json()
            return data.get("status") == "cancelled"
        except:
            return False
    
    def validate_plan(self, plan: Dict) -> List[str]:
        """Validates the structure of the experiment plan"""
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
    
    async def run(self):
        """Main orchestration loop"""
        print(f"[{self.experiment_id}] Starting orchestration for: {self.query}")
        
        try:
            # 1. Conceptualization
            if await self.check_cancelled():
                return
            
            await self.update_stage("conceptualization", {"status": "in_progress"})
            print(f"[{self.experiment_id}] Running conceptualization...")
            
            conceptual_plan = conceptualization_agent.run(self.client, self.query, MODEL_NAME)
            if not conceptual_plan:
                raise Exception("Conceptualization failed")
            
            await self.update_kv("conceptual_plan", conceptual_plan)
            await self.update_stage("conceptualization", {
                "status": "completed",
                "domain": conceptual_plan.get('domain', 'N/A'),
                "task": conceptual_plan.get('task', 'N/A')
            })
            
            # 2. Strategy & Validation Loop
            if await self.check_cancelled():
                return
            
            await self.update_stage("strategy", {"status": "in_progress"})
            print(f"[{self.experiment_id}] Planning strategy...")
            
            experiment_plan = None
            last_error_log = None
            
            for attempt in range(MAX_PLAN_RETRIES + 1):
                error_context = {"error_type": "SCHEMA_VALIDATION_ERROR", "error_log": last_error_log} if last_error_log else None
                
                plan_candidate = strategy_agent.run(
                    self.client, conceptual_plan, TARGET_FRAMEWORK, MODEL_NAME, 
                    previous_error=error_context
                )
                
                if not plan_candidate:
                    last_error_log = "Agent returned an empty plan."
                    continue
                
                validation_errors = self.validate_plan(plan_candidate)
                if not validation_errors:
                    experiment_plan = plan_candidate
                    break
                
                last_error_log = "\n".join(validation_errors)
                print(f"[{self.experiment_id}] Plan validation failed: {last_error_log}")
            
            if not experiment_plan:
                raise Exception("Strategy agent failed to produce valid plan")
            
            await self.update_kv("experiment_plan", experiment_plan)
            await self.update_stage("strategy", {"status": "completed"})
            
            # Save plan to file
            await self.store_file("experiment_plan.json", json.dumps(experiment_plan, indent=2))
            
            # 3. Implementation & Execution Loop
            if await self.check_cancelled():
                return
            
            await self.update_stage("implementation", {"status": "in_progress"})
            print(f"[{self.experiment_id}] Generating implementation...")
            
            models_to_run = [experiment_plan["models"]["candidate"]] + experiment_plan["models"]["baselines"]
            all_results = []
            
            for model_spec in models_to_run:
                if await self.check_cancelled():
                    return
                
                print(f"[{self.experiment_id}] Implementing {model_spec['model_id']}...")
                
                # Generate code
                executable_code = implementation_agent.run(
                    self.client, experiment_plan, model_spec, MODEL_NAME
                )
                
                script_content = executable_code.get("script")
                if script_content:
                    filename = f"{model_spec['model_id']}_script.py"
                    await self.store_file(filename, script_content)
                    print(f"[{self.experiment_id}] Saved {filename}")
                
                # Run experiment (simulated for now)
                result = experimentation_agent.run(executable_code)
                
                if result["status"] == "completed":
                    result['model_id'] = model_spec['model_id']
                    all_results.append(result)
                    await self.update_kv(f"result_{model_spec['model_id']}", result)
            
            await self.update_stage("implementation", {"status": "completed"})
            
            # 4. Analysis
            if await self.check_cancelled():
                return
            
            if all_results:
                await self.update_stage("analysis", {"status": "in_progress"})
                print(f"[{self.experiment_id}] Generating analysis...")
                
                final_report = analysis_agent.run(
                    self.client, all_results, experiment_plan['hypothesis'], MODEL_NAME
                )
                
                await self.store_file("final_report.md", final_report)
                await self.update_kv("final_report", final_report)
                await self.update_stage("analysis", {"status": "completed"})
                
                print(f"[{self.experiment_id}] Experiment completed successfully!")
            
        except Exception as e:
            print(f"[{self.experiment_id}] Error: {str(e)}")
            await self.update_stage("error", {"error": str(e)})
            raise
        finally:
            await self.http_client.aclose()

async def run_orchestrator_with_api(experiment_id: str, query: str):
    """Entry point for running orchestrator with API communication"""
    orchestrator = OrchestratorAPI(experiment_id, query)
    await orchestrator.run()
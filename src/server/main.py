# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os

# app = FastAPI()

# # Allow cross-origin requests from any domain so the frontend can
# # be hosted separately (e.g. on a different Railway service).
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class PlanRequest(BaseModel):
#     prompt: str

#     class Config:
#         # Pydantic v1 configuration
#         schema_extra = {
#             "example": {
#                 "prompt": "Create a machine learning experiment for image classification"
#             }
#         }

# @app.get("/api/hello")
# async def read_hello():
#     return {"message": "Hello from server!"}

# @app.post("/api/plan")
# async def create_plan(request: PlanRequest):
#     """Create an experiment plan based on the user's prompt"""
#     return {
#         "experimentSpec": f"Experiment Plan for: {request.prompt}\n\n1. Data Collection\n2. Preprocessing\n3. Model Training\n4. Evaluation\n5. Analysis"
#     }

# @app.post("/api/codegen")
# async def generate_code():
#     """Generate code files for the experiment"""
#     return {
#         "files": {
#             "main.py": "import pandas as pd\nimport numpy as np\n\n# Your experiment code here\nprint('Hello from experiment!')",
#             "requirements.txt": "pandas\nnumpy\nscikit-learn",
#             "README.md": "# Experiment Setup\n\nRun with: python main.py"
#         }
#     }

# @app.post("/api/execute")
# async def execute_experiment():
#     """Execute the experiment"""
#     return {
#         "status": "success",
#         "results": {
#             "accuracy": 0.85,
#             "training_time": "2.3s",
#             "model_size": "15MB"
#         }
#     }

# @app.post("/api/report")
# async def generate_report():
#     """Generate a summary report"""
#     return {
#         "summary": "Experiment completed successfully!\n\nKey Results:\n- Accuracy: 85%\n- Training time: 2.3 seconds\n- Model performance: Good\n\nRecommendations:\n- Consider hyperparameter tuning\n- Collect more training data"
#     }

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=port)


# main.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import orchestrator_app

# Pydantic model for the incoming request body
class ResearchRequest(BaseModel):
    research_idea: str

app = FastAPI(
    title="AI Research Lab API",
    description="An API to autonomously design, execute, and analyze scientific experiments from a high-level idea.",
    version="1.0.0"
)

# In-memory "database" to track job status (for a real app, use a proper DB like Redis or Postgres)
job_status = {}

def run_research_task(job_id: str, research_idea: str):
    """
    Wrapper function to run the orchestrator and update job status.
    This is what the background task will execute.
    """
    try:
        # The orchestrator's main function now returns the final results path
        result_paths = orchestrator_app.main(research_idea=research_idea, job_id=job_id)
        if result_paths:
            job_status[job_id] = {"status": "COMPLETED", **result_paths}
        else:
            job_status[job_id] = {"status": "FAILED", "detail": "Orchestrator did not complete successfully."}
    except Exception as e:
        job_status[job_id] = {"status": "ERROR", "detail": str(e)}

@app.post("/start-experiment/", status_code=202)
def start_experiment(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Accepts a research idea and starts the full research pipeline as a background task.
    """
    # Create a unique ID for this research job
    import uuid
    job_id = str(uuid.uuid4())
    
    # Immediately add the job to our status tracker
    job_status[job_id] = {"status": "PENDING", "detail": "Research task has been queued."}
    
    # Add the long-running orchestrator function to the background tasks
    background_tasks.add_task(run_research_task, job_id, request.research_idea)
    
    # Return immediately to the user with the job ID
    return {"message": "Research task accepted.", "job_id": job_id}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Returns the current status and results of a research job.
    """
    status = job_status.get(job_id)
    if not status:
        return {"status": "NOT_FOUND", "detail": "No job found with this ID."}
    return status

@app.get("/results/{job_id}")
def get_results(job_id: str):
    res= job_status.get(job_id)
    if "progress_metrics_path" in res:
        return res["progress_metrics_path"]
    else:
        return {}

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
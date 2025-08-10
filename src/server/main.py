from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

# Allow cross-origin requests from any domain so the frontend can
# be hosted separately (e.g. on a different Railway service).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlanRequest(BaseModel):
    prompt: str

    class Config:
        # Pydantic v1 configuration
        schema_extra = {
            "example": {
                "prompt": "Create a machine learning experiment for image classification"
            }
        }

@app.get("/api/hello")
async def read_hello():
    return {"message": "Hello from server!"}

@app.post("/api/plan")
async def create_plan(request: PlanRequest):
    """Create an experiment plan based on the user's prompt"""
    return {
        "experimentSpec": f"Experiment Plan for: {request.prompt}\n\n1. Data Collection\n2. Preprocessing\n3. Model Training\n4. Evaluation\n5. Analysis"
    }

@app.post("/api/codegen")
async def generate_code():
    """Generate code files for the experiment"""
    return {
        "files": {
            "main.py": "import pandas as pd\nimport numpy as np\n\n# Your experiment code here\nprint('Hello from experiment!')",
            "requirements.txt": "pandas\nnumpy\nscikit-learn",
            "README.md": "# Experiment Setup\n\nRun with: python main.py"
        }
    }

@app.post("/api/execute")
async def execute_experiment():
    """Execute the experiment"""
    return {
        "status": "success",
        "results": {
            "accuracy": 0.85,
            "training_time": "2.3s",
            "model_size": "15MB"
        }
    }

@app.post("/api/report")
async def generate_report():
    """Generate a summary report"""
    return {
        "summary": "Experiment completed successfully!\n\nKey Results:\n- Accuracy: 85%\n- Training time: 2.3 seconds\n- Model performance: Good\n\nRecommendations:\n- Consider hyperparameter tuning\n- Collect more training data"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)

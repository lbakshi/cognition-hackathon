"""
Modal execution service for ML experiments
Perfect integration with your /codegen system
"""
import modal
import json
import os
from typing import Dict, Any

# Create Modal app
app = modal.App("agentic-research-mvp")

# Define the image with ML dependencies
ml_image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "tqdm>=4.66.0",
])

# Create persistent volume for results
volume = modal.Volume.from_name("ml-experiment-results", create_if_missing=True)

@app.function(
    image=ml_image,
    volumes={"/results": volume},
    gpu="T4",  # Optional GPU - can use CPU only
    timeout=3600,  # 1 hour max
    memory=8192,  # 8GB RAM
)
def execute_ml_experiment(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute generated PyTorch experiment code
    
    Args:
        experiment_data: {
            "experiment_id": str,
            "code": str,  # Generated PyTorch code
            "dependencies": List[str]
        }
    
    Returns:
        {
            "experiment_id": str,
            "status": "completed" | "failed",
            "results": Dict[str, Any],
            "error": str | None,
            "artifacts": List[str]
        }
    """
    experiment_id = experiment_data["experiment_id"]
    code = experiment_data["code"]
    
    try:
        print(f"🚀 Starting experiment: {experiment_id}")
        
        # Create experiment directory
        exp_dir = f"/results/{experiment_id}"
        os.makedirs(exp_dir, exist_ok=True)
        os.chdir(exp_dir)
        
        # Write the code to a file for debugging
        with open("experiment_code.py", "w") as f:
            f.write(code)
        
        # Execute the generated code
        print("⚡ Executing generated PyTorch code...")
        exec_globals = {
            "__name__": "__main__",
            "__file__": "experiment_code.py"
        }
        exec(code, exec_globals)
        
        # Collect results and artifacts
        artifacts = []
        results = {}
        
        # Look for common result files
        if os.path.exists("experiment_results.json"):
            with open("experiment_results.json") as f:
                results = json.load(f)
            artifacts.append("experiment_results.json")
        
        # Look for model checkpoints
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                artifacts.append(f"checkpoints/{f}")
        
        # Look for plots/images
        for ext in [".png", ".jpg", ".pdf"]:
            for f in os.listdir("."):
                if f.endswith(ext):
                    artifacts.append(f)
        
        print(f"✅ Experiment completed successfully!")
        print(f"📊 Results: {results}")
        print(f"📁 Artifacts: {artifacts}")
        
        # Commit volume changes
        volume.commit()
        
        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "results": results,
            "error": None,
            "artifacts": artifacts,
            "execution_path": exp_dir
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Experiment failed: {error_msg}")
        
        return {
            "experiment_id": experiment_id,
            "status": "failed", 
            "results": {},
            "error": error_msg,
            "artifacts": [],
            "execution_path": exp_dir
        }

@app.function(
    image=ml_image,
    volumes={"/results": volume}
)
def get_experiment_artifacts(experiment_id: str, artifact_name: str) -> bytes:
    """Get specific artifact from experiment"""
    artifact_path = f"/results/{experiment_id}/{artifact_name}"
    
    if os.path.exists(artifact_path):
        with open(artifact_path, "rb") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Artifact {artifact_name} not found")

# FastAPI integration endpoint
@app.function()
@modal.web_endpoint(method="POST")
def execute_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for executing experiments
    Compatible with your /execute API design
    """
    return execute_ml_experiment.remote(request_data)

# Local development/testing
@app.local_entrypoint()
def test_execution():
    """Test the execution with sample data"""
    
    sample_code = '''
import torch
import torch.nn as nn

print("🧪 Testing PyTorch execution...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Simple test model
model = nn.Linear(10, 1)
x = torch.randn(5, 10)
y = model(x)
print(f"Model output shape: {y.shape}")

# Save results
import json
results = {
    "test_status": "success",
    "pytorch_version": torch.__version__,
    "output_shape": list(y.shape)
}

with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("✅ Test completed!")
'''
    
    test_data = {
        "experiment_id": "test_experiment",
        "code": sample_code,
        "dependencies": ["torch"]
    }
    
    result = execute_ml_experiment.remote(test_data)
    print("🎉 Execution result:", result)


if __name__ == "__main__":
    print("🚀 Modal ML Experiment Executor")
    print("Run with: modal run modal_execution_example.py")

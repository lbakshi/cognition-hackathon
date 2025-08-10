"""
Modal Execution Service for ML Experiments
This service handles submitting experiments to Modal and retrieving results.
"""

import modal
import json
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("ml-experiment-runner")

# Define the image with ML dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "scikit-learn", 
    "matplotlib",
    "numpy",
    "pandas",
    "tqdm",
    "Pillow"
)

# Create volume for experiment artifacts
volume = modal.Volume.from_name("experiment-artifacts", create_if_missing=True)

@app.function(
    image=image,
    volumes={'/artifacts': volume},
    gpu="any",
    timeout=1800,
    memory=8192,
)
def execute_experiment(
    experiment_files: Dict[str, str],  # filename -> content
    experiment_spec: Dict[str, Any],
    job_id: str
) -> Dict[str, Any]:
    """Execute ML experiment on Modal with given files and spec"""
    
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn, optim
    from sklearn.metrics import f1_score, accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import json
    import sys
    from pathlib import Path
    
    # Setup artifacts directory
    artifacts_dir = Path(f"/artifacts/{job_id}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment spec
    with open(artifacts_dir / "experiment_spec.json", "w") as f:
        json.dump(experiment_spec, f, indent=2)
    
    # Write experiment files to disk
    for filename, content in experiment_files.items():
        file_path = artifacts_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
    
    results = {}
    
    try:
        # Find and execute the main training script
        main_script = None
        for filename, content in experiment_files.items():
            if filename == "train.py" or "train" in filename:
                main_script = content
                break
        
        if not main_script:
            # Use the first .py file
            main_script = next((content for filename, content in experiment_files.items() 
                              if filename.endswith('.py')), None)
        
        if not main_script:
            raise ValueError("No Python training script found")
        
        # Create execution namespace
        namespace = {
            'config': experiment_spec,
            'job_id': job_id,
            'artifacts_dir': str(artifacts_dir),
            'results': {},
            'torch': torch,
            'torchvision': torchvision,
            'transforms': transforms,
            'nn': nn,
            'optim': optim,
            'f1_score': f1_score,
            'accuracy_score': accuracy_score,
            'plt': plt,
            'np': np,
            'Path': Path,
            'json': json,
        }
        
        # Execute the main script
        logger.info(f"Executing experiment {job_id}")
        exec(main_script, namespace)
        
        # Extract results
        results = namespace.get('results', {})
        
        # Save results
        with open(artifacts_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary plots if metrics exist
        _generate_summary_plots(results, artifacts_dir)
        
        # Commit volume changes
        volume.commit()
        
        logger.info(f"Experiment {job_id} completed successfully")
        return {
            'status': 'success',
            'job_id': job_id,
            'results': results,
            'artifacts_path': str(artifacts_dir)
        }
        
    except Exception as e:
        logger.error(f"Experiment {job_id} failed: {str(e)}")
        
        # Save error info
        error_info = {
            'error': str(e),
            'job_id': job_id,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(artifacts_dir / "error.json", "w") as f:
            json.dump(error_info, f, indent=2)
        
        volume.commit()
        
        return {
            'status': 'error',
            'job_id': job_id,
            'error': str(e),
            'artifacts_path': str(artifacts_dir)
        }

def _generate_summary_plots(results: Dict[str, Any], artifacts_dir: Path):
    """Generate summary plots from results"""
    import matplotlib.pyplot as plt
    
    # Extract metrics across models
    models = []
    accuracies = []
    losses = []
    
    for model_name, model_results in results.items():
        if isinstance(model_results, dict):
            models.append(model_name)
            accuracies.append(model_results.get('accuracy', 0))
            losses.append(model_results.get('loss', 0))
    
    if models:
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        ax1.bar(models, accuracies)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Loss comparison
        ax2.bar(models, losses)
        ax2.set_title('Model Loss Comparison')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(artifacts_dir / "model_comparison.png")
        plt.close()
    
    # Generate training curves if available
    for model_name, model_results in results.items():
        if isinstance(model_results, dict):
            loss_history = model_results.get('loss_history', [])
            acc_history = model_results.get('accuracy_history', [])
            
            if loss_history or acc_history:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                if loss_history:
                    axes[0].plot(loss_history)
                    axes[0].set_title(f'{model_name} - Training Loss')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                
                if acc_history:
                    axes[1].plot(acc_history)
                    axes[1].set_title(f'{model_name} - Training Accuracy')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Accuracy')
                
                plt.tight_layout()
                plt.savefig(artifacts_dir / f"{model_name}_training_curves.png")
                plt.close()

@app.function(
    image=image,
    volumes={'/artifacts': volume},
)
def fetch_job_artifacts(job_id: str) -> Dict[str, Any]:
    """Fetch all artifacts for a completed job"""
    artifacts_dir = Path(f"/artifacts/{job_id}")
    
    if not artifacts_dir.exists():
        return {'error': f'No artifacts found for job {job_id}'}
    
    artifacts = {'job_id': job_id}
    
    # Load results
    results_file = artifacts_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            artifacts['results'] = json.load(f)
    
    # Load experiment spec
    spec_file = artifacts_dir / "experiment_spec.json"
    if spec_file.exists():
        with open(spec_file, "r") as f:
            artifacts['experiment_spec'] = json.load(f)
    
    # Load error info if exists
    error_file = artifacts_dir / "error.json"
    if error_file.exists():
        with open(error_file, "r") as f:
            artifacts['error_info'] = json.load(f)
    
    # Encode plots as base64
    import base64
    for plot_file in artifacts_dir.glob("*.png"):
        with open(plot_file, "rb") as f:
            artifacts[f"plot_{plot_file.stem}"] = base64.b64encode(f.read()).decode('utf-8')
    
    return artifacts

class ModalExecutionService:
    """Service for managing Modal experiment execution"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    async def submit_experiment(
        self, 
        experiment_files: Dict[str, str],
        experiment_spec: Dict[str, Any],
        experiment_id: Optional[str] = None
    ) -> str:
        """Submit an experiment to Modal and return job ID"""
        
        job_id = experiment_id or f"job_{uuid.uuid4().hex[:8]}"
        
        # Store job metadata
        self.jobs[job_id] = {
            'job_id': job_id,
            'status': 'submitted',
            'submitted_at': datetime.now().isoformat(),
            'experiment_spec': experiment_spec
        }
        
        logger.info(f"Submitting experiment {job_id} to Modal")
        
        try:
            # Execute on Modal
            with app.run():
                result = await execute_experiment.remote.aio(
                    experiment_files, 
                    experiment_spec, 
                    job_id
                )
            
            # Update job status
            self.jobs[job_id].update({
                'status': result['status'],
                'completed_at': datetime.now().isoformat(),
                'results': result.get('results'),
                'error': result.get('error'),
                'artifacts_path': result.get('artifacts_path')
            })
            
            logger.info(f"Experiment {job_id} completed with status: {result['status']}")
            
        except Exception as e:
            logger.error(f"Failed to execute experiment {job_id}: {str(e)}")
            self.jobs[job_id].update({
                'status': 'error',
                'completed_at': datetime.now().isoformat(),
                'error': str(e)
            })
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and metadata"""
        return self.jobs.get(job_id)
    
    async def get_job_artifacts(self, job_id: str) -> Dict[str, Any]:
        """Fetch artifacts for a job"""
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        try:
            with app.run():
                artifacts = await fetch_job_artifacts.remote.aio(job_id)
            return artifacts
        except Exception as e:
            logger.error(f"Failed to fetch artifacts for {job_id}: {str(e)}")
            return {'error': str(e)}
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        return list(self.jobs.values())

# Global service instance
modal_service = ModalExecutionService()

async def main():
    """Test the Modal execution service"""
    # Sample experiment files
    experiment_files = {
        "train.py": '''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# Simple CNN for testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print(f"Training experiment {job_id}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mock training for demo
model = SimpleCNN().to(device)
start_time = time.time()

# Simulate training
epochs = config.get('training_params', {}).get('epochs', 2)
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    time.sleep(1)  # Simulate training time

training_time = time.time() - start_time

# Generate mock results
results['custom_model'] = {
    'accuracy': 0.75 + np.random.random() * 0.15,
    'f1_score': 0.72 + np.random.random() * 0.15,
    'loss': 0.4 + np.random.random() * 0.3,
    'training_time': training_time,
    'inference_time': 0.002,
    'loss_history': [0.8, 0.6, 0.45, 0.4],
    'accuracy_history': [0.5, 0.65, 0.72, 0.75]
}

results['baseline_cnn'] = {
    'accuracy': 0.70,
    'f1_score': 0.68,
    'loss': 0.5,
    'training_time': training_time * 0.8,
    'inference_time': 0.001,
    'loss_history': [0.9, 0.7, 0.55, 0.5],
    'accuracy_history': [0.45, 0.6, 0.68, 0.70]
}

print(f"Training completed! Results: {results}")
'''
    }
    
    experiment_spec = {
        'experiment_id': 'test_exp_001',
        'model_config': {'type': 'simple_cnn'},
        'dataset': 'cifar10',
        'training_params': {'epochs': 2, 'batch_size': 128}
    }
    
    # Submit experiment
    job_id = await modal_service.submit_experiment(experiment_files, experiment_spec)
    print(f"Experiment submitted: {job_id}")
    
    # Check status
    status = modal_service.get_job_status(job_id)
    print(f"Job status: {status}")
    
    # Fetch artifacts
    if status and status.get('status') == 'success':
        artifacts = await modal_service.get_job_artifacts(job_id)
        print(f"Artifacts: {list(artifacts.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
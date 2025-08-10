#!/usr/bin/env python3
"""
Modal Online Execution Runtime for ML Experiments
This script provides a Modal serverless execution runtime similar to the notebook implementation.
"""

import modal
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Create Modal app
app = modal.App("ml-experiment-runner")

# Define the image with all required dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision", 
    "scikit-learn",
    "matplotlib",
    "numpy",
    "pandas",
    "tqdm"
)

# Create a volume for storing experiment artifacts
volume = modal.Volume.from_name("experiment-artifacts", create_if_missing=True)

@app.function(
    image=image,
    volumes={'/artifacts': volume},
    gpu="any",  # Use GPU if available, otherwise CPU
    timeout=1800,  # 30 minutes timeout
    memory=8192,  # 8GB RAM
)
def run_experiment(
    experiment_script: str,
    config: Dict[str, Any],
    job_id: str
) -> Dict[str, Any]:
    """Run an ML experiment on Modal"""
    
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn, optim
    from sklearn.metrics import f1_score, accuracy_score
    import time
    import json
    
    # Create artifacts directory for this job
    artifacts_dir = Path(f"/artifacts/{job_id}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Execute the experiment script
    results = {}
    
    try:
        # Create a namespace for script execution
        namespace = {
            'config': config,
            'artifacts_dir': str(artifacts_dir),
            'results': {},
            'torch': torch,
            'torchvision': torchvision,
            'transforms': transforms,
            'nn': nn,
            'optim': optim,
            'f1_score': f1_score,
            'accuracy_score': accuracy_score,
        }
        
        # Execute the script
        exec(experiment_script, namespace)
        
        # Get results from namespace
        results = namespace.get('results', {})
        
        # Save results
        with open(artifacts_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Commit volume changes
        volume.commit()
        
        return {
            'status': 'success',
            'results': results,
            'artifacts_path': str(artifacts_dir)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'artifacts_path': str(artifacts_dir)
        }

@app.function(image=image)
def square(x):
    """Simple test function to verify Modal is working"""
    print("This code is running on a remote worker!")
    result = x ** 2
    print(f"the square is {result}")
    return result

@app.local_entrypoint()
def main():
    """Main entry point for the Modal app"""
    print("Starting Modal ML Experiment Runner...")
    
    # Test the connection with a simple function
    result = square.remote(42)
    print(f"Square of 42 is: {result}")
    
    # Example of running an experiment
    sample_config = {
        "experiment_id": "test_exp_001",
        "model_config": {
            "type": "simple_test",
            "activation": "relu"
        },
        "dataset": "mock",
        "training_params": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    # Simple test experiment script
    test_script = '''
import numpy as np
import time

print("Running test experiment on Modal...")
time.sleep(2)

# Generate mock results
results['test_model'] = {
    'accuracy': 0.85 + np.random.random() * 0.1,
    'f1_score': 0.83 + np.random.random() * 0.1,
    'loss': 0.3 + np.random.random() * 0.2,
    'training_time': 5.2,
    'inference_time': 0.001,
    'loss_history': [0.8, 0.6, 0.4, 0.3],
    'accuracy_history': [0.6, 0.7, 0.8, 0.85]
}

print("Test experiment completed!")
print(f"Results: {results}")
'''
    
    # Run the experiment
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    print(f"Running experiment with job ID: {job_id}")
    
    experiment_result = run_experiment.remote(test_script, sample_config, job_id)
    
    if experiment_result['status'] == 'success':
        print("✓ Experiment completed successfully!")
        print(f"Results: {experiment_result['results']}")
        print(f"Artifacts saved to: {experiment_result['artifacts_path']}")
    else:
        print(f"✗ Experiment failed: {experiment_result['error']}")
    
    print("Modal execution completed.")

if __name__ == "__main__":
    main()
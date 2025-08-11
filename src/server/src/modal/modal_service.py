# agents/modal_service.py
"""
Modal service for running ML experiments in the cloud.
This module handles all Modal-specific logic and provides a clean interface for the experimentation agent.
"""

import modal
import json
import uuid
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
# Modal credentials should already be set in the environment by the calling code

# Add the agents directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.modal.modal_config import (
    MODAL_APP_NAME, GPU_CONFIG, MEMORY_GB, TIMEOUT_SECONDS, 
    VOLUME_NAME, REQUIRED_PACKAGES, DEFAULT_EXPERIMENT_CONFIG
)

# Create Modal app
app = modal.App(MODAL_APP_NAME)

# Define the image with all required dependencies
image = modal.Image.debian_slim().pip_install(*REQUIRED_PACKAGES)

# Create a volume for storing experiment artifacts
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={'/artifacts': volume},
    gpu=GPU_CONFIG,  # Use GPU if available, otherwise CPU
    timeout=TIMEOUT_SECONDS,
    memory=MEMORY_GB * 1024,  # Convert GB to MB
)
def run_experiment_remote(
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
    import numpy as np
    
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
            'results': results,
            'torch': torch,
            'torchvision': torchvision,
            'transforms': transforms,
            'nn': nn,
            'optim': optim,
            'f1_score': f1_score,
            'accuracy_score': accuracy_score,
            'np': np,
            'time': time,
            'json': json,
            'Path': Path,
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
        # Save error details
        error_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(artifacts_dir / "error.json", "w") as f:
            json.dump(error_info, f, indent=2)
        
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'artifacts_path': str(artifacts_dir)
        }

class ModalExperimentRunner:
    """Wrapper class for running experiments on Modal"""
    
    def __init__(self):
        self.app = app
        self.job_id = None
    
    def run_experiment(self, script_content: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run an experiment on Modal
        
        Args:
            script_content: The Python script to execute
            config: Optional configuration dictionary
            
        Returns:
            Dictionary containing execution results
        """
        if not script_content:
            return {
                "status": "failed", 
                "error_type": "CODE_GENERATION_FAILED", 
                "error_log": "No script provided."
            }
        
        # Generate unique job ID
        self.job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Merge with default config
        if config is None:
            config = {}
        
        final_config = {**DEFAULT_EXPERIMENT_CONFIG, **config}
        final_config.update({
            "experiment_id": self.job_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            print(f"INFO: [Modal Service] Starting experiment with job ID: {self.job_id}")
            print(f"INFO: [Modal Service] Using GPU: {GPU_CONFIG}, Memory: {MEMORY_GB}GB, Timeout: {TIMEOUT_SECONDS}s")
            
            # Run the experiment on Modal
            with app.run():
                result = run_experiment_remote.remote(script_content, final_config, self.job_id)
            
            if result['status'] == 'success':
                print("INFO: [Modal Service] Experiment completed successfully.")
                return {
                    "status": "success",
                    "results": result['results'],
                    "job_id": self.job_id,
                    "artifacts_path": result['artifacts_path']
                }
            else:
                print(f"ERROR: [Modal Service] Experiment failed: {result.get('error', 'Unknown error')}")
                return {
                    "status": "failed",
                    "error_type": result.get('error_type', 'MODAL_EXECUTION_ERROR'),
                    "error_log": result.get('error', 'Unknown error occurred on Modal'),
                    "job_id": self.job_id,
                    "artifacts_path": result.get('artifacts_path')
                }
                
        except Exception as e:
            print(f"ERROR: [Modal Service] Failed to execute experiment: {str(e)}")
            return {
                "status": "failed",
                "error_type": "MODAL_SERVICE_ERROR",
                "error_log": str(e),
                "job_id": self.job_id
            }
    
    def get_job_status(self) -> str:
        """Get the current job ID"""
        return self.job_id

# Global instance for easy access
modal_runner = ModalExperimentRunner() 
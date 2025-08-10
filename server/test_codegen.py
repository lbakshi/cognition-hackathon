"""
Test script for the /codegen endpoint
"""
import json
import requests
import asyncio
import sys
import os

# Add parent directory to path to import main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ExperimentPlan, CodegenRequest

# Sample ExperimentPlan data (as provided by the user)
sample_experiment_plan = {
    "experiment_id": "exp_20250810_1",
    "experiment_name": "GELU_vs_ReLU_with_DepthwiseConv",
    "hypothesis": "A CNN with GELU and depthwise separable convolutions (candidate) will outperform a standard CNN with ReLU (baseline) on CIFAR-10.",
    "dataset": {
        "name": "CIFAR-10",
        "source": "torchvision.datasets.CIFAR10",
        "validation_split": 0.2
    },
    "training_parameters": {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss_function": "CrossEntropyLoss",
        "batch_size": 64,
        "epochs": 10
    },
    "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
    "models": {
        "candidate": {
            "model_id": "candidate_model",
            "description": "CNN with GELU and Depthwise Separable Convolutions",
            "architecture": [
                {"type": "Conv2D", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3}},
                {"type": "Activation", "params": {"function": "GELU"}},
                {"type": "DepthwiseSeparableConv2D", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3}},
                {"type": "Activation", "params": {"function": "GELU"}},
                {"type": "MaxPooling2D", "params": {"kernel_size": 2}},
                {"type": "Flatten"},
                {"type": "Linear", "params": {"out_features": 10}}
            ]
        },
        "baselines": [
            {
                "model_id": "baseline_model_relu",
                "description": "Standard CNN with ReLU and standard convolutions",
                "architecture": [
                    {"type": "Conv2D", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3}},
                    {"type": "Activation", "params": {"function": "ReLU"}},
                    {"type": "Conv2D", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3}},
                    {"type": "Activation", "params": {"function": "ReLU"}},
                    {"type": "MaxPooling2D", "params": {"kernel_size": 2}},
                    {"type": "Flatten"},
                    {"type": "Linear", "params": {"out_features": 10}}
                ]
            }
        ]
    }
}

def test_codegen_locally():
    """
    Test the code generation logic locally (without HTTP)
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import generate_pytorch_code, create_code_generation_prompt, ExperimentPlan
    
    # Create ExperimentPlan object
    plan = ExperimentPlan(**sample_experiment_plan)
    
    # Test prompt generation
    prompt = create_code_generation_prompt(plan)
    print("=== GENERATED PROMPT ===")
    print(prompt)
    print("=" * 50)
    
    # Note: To test actual Claude API, you need to set ANTHROPIC_API_KEY
    print("\nTo test actual code generation, set ANTHROPIC_API_KEY in .env file")
    print("Then run: python test_codegen.py --api")

def test_codegen_api():
    """
    Test the /codegen endpoint via HTTP
    """
    url = "http://localhost:8000/codegen"
    
    request_data = {
        "plan_id": "test_plan_1",
        "experiment_plan": sample_experiment_plan
    }
    
    try:
        response = requests.post(url, json=request_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Code generation successful!")
            print(f"Code ID: {result['code_id']}")
            print(f"Dependencies: {result['executable_code']['dependencies']}")
            print("\n=== GENERATED CODE ===")
            print(result['executable_code']['code'])
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("Make sure the server is running: uvicorn main:app --reload")

if __name__ == "__main__":
    import sys
    
    if "--api" in sys.argv:
        test_codegen_api()
    else:
        test_codegen_locally()

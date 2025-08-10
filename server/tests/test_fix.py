"""
Test script for the /codegen/fix endpoint
"""
import requests
import sys

SERVER_URL = "http://localhost:8002"

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
        "epochs": 1
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

def run_test_fix():
    # 1) Generate code to get a code_id
    gen_req = {"plan_id": "fix_plan_1", "experiment_plan": sample_experiment_plan}
    r = requests.post(f"{SERVER_URL}/codegen", json=gen_req, timeout=300)
    if r.status_code != 200:
        print("❌ /codegen failed:", r.status_code, r.text)
        sys.exit(1)
    gen = r.json()
    code_id = gen["code_id"]
    print("✅ /codegen ok, code_id:", code_id)

    # 2) Send a synthetic error to /codegen/fix
    # Use a plausible runtime error so Claude can adjust code if needed
    fix_req = {
        "code_id": code_id,
        "error_message": "RuntimeError: CUDA out of memory on device 0 while allocating tensor.",
        "error_type": "RuntimeError"
    }
    r2 = requests.post(f"{SERVER_URL}/codegen/fix", json=fix_req, timeout=300)
    if r2.status_code != 200:
        print("❌ /codegen/fix failed:", r2.status_code, r2.text)
        sys.exit(1)
    print("✅ /codegen/fix ok:", r2.json())

if __name__ == "__main__":
    run_test_fix()

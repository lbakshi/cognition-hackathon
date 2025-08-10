"""
Test script for the /codegen/fix endpoint
Tests various error scenarios and code fixing capabilities
"""
import json
import requests
import time
import sys
import os

# Add parent directory to path to import main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fix_syntax_error():
    """Test fixing a syntax error"""
    
    url = "http://localhost:8002/codegen/fix"
    
    # Simulate a code with syntax error
    original_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self)
        super(Model, self).__init__()  # Missing colon
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.layer(x)

model = Model()
print("Model created successfully")
'''
    
    test_data = {
        "code_id": "test_syntax_error",
        "error_message": "SyntaxError: invalid syntax at line 6",
        "error_type": "SyntaxError"
    }
    
    print("🧪 Testing Syntax Error Fix...")
    print(f"Original error: {test_data['error_message']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Syntax error fix successful!")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_fix_import_error():
    """Test fixing an import error"""
    
    url = "http://localhost:8002/codegen/fix"
    
    test_data = {
        "code_id": "test_import_error", 
        "error_message": "ModuleNotFoundError: No module named 'torchvision'",
        "error_type": "ImportError"
    }
    
    print("\n🧪 Testing Import Error Fix...")
    print(f"Original error: {test_data['error_message']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Import error fix successful!")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_fix_tensor_shape_error():
    """Test fixing a tensor shape mismatch error"""
    
    url = "http://localhost:8002/codegen/fix"
    
    test_data = {
        "code_id": "test_tensor_shape_error",
        "error_message": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 1024x10)",
        "error_type": "RuntimeError"
    }
    
    print("\n🧪 Testing Tensor Shape Error Fix...")
    print(f"Original error: {test_data['error_message']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Tensor shape error fix successful!")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_fix_cuda_error():
    """Test fixing a CUDA device error"""
    
    url = "http://localhost:8002/codegen/fix"
    
    test_data = {
        "code_id": "test_cuda_error",
        "error_message": "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!",
        "error_type": "RuntimeError"
    }
    
    print("\n🧪 Testing CUDA Device Error Fix...")
    print(f"Original error: {test_data['error_message']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CUDA device error fix successful!")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_fix_missing_code_id():
    """Test handling of missing code_id"""
    
    url = "http://localhost:8002/codegen/fix"
    
    test_data = {
        "code_id": "nonexistent_code_id",
        "error_message": "Some error occurred",
        "error_type": "RuntimeError"
    }
    
    print("\n🧪 Testing Missing Code ID Handling...")
    print(f"Testing with non-existent code_id: {test_data['code_id']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=60)
        
        if response.status_code == 404:
            print("✅ Missing code ID handled correctly!")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_full_codegen_fix_cycle():
    """Test complete cycle: generate code -> simulate error -> fix code"""
    
    print("\n🚀 Testing Full CodeGen -> Fix Cycle...")
    
    # Step 1: Generate code first
    codegen_url = "http://localhost:8002/codegen"
    
    sample_experiment_plan = {
        "experiment_id": "test_fix_cycle",
        "experiment_name": "Test_Fix_Cycle",
        "hypothesis": "Test the fix functionality",
        "dataset": {
            "name": "CIFAR-10",
            "source": "torchvision.datasets.CIFAR10",
            "validation_split": 0.2
        },
        "training_parameters": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss_function": "CrossEntropyLoss",
            "batch_size": 32,
            "epochs": 1  # Quick test
        },
        "evaluation_metrics": ["accuracy"],
        "models": {
            "candidate": {
                "model_id": "test_model",
                "description": "Simple test model",
                "architecture": [
                    {"type": "Conv2D", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}},
                    {"type": "Activation", "params": {"function": "ReLU"}},
                    {"type": "Flatten"},
                    {"type": "Linear", "params": {"out_features": 10}}
                ]
            },
            "baselines": [
                {
                    "model_id": "baseline_simple",
                    "description": "Simple baseline",
                    "architecture": [
                        {"type": "Linear", "params": {"in_features": 3072, "out_features": 10}}
                    ]
                }
            ]
        }
    }
    
    request_data = {
        "plan_id": "test_fix_cycle",
        "experiment_plan": sample_experiment_plan
    }
    
    print("📝 Step 1: Generating code...")
    
    try:
        # Generate code
        codegen_response = requests.post(codegen_url, json=request_data, timeout=300)
        
        if codegen_response.status_code == 200:
            codegen_result = codegen_response.json()
            code_id = codegen_result['code_id']
            print(f"✅ Code generated with ID: {code_id}")
            
            # Step 2: Simulate a fix
            print("🔧 Step 2: Testing fix functionality...")
            
            fix_url = "http://localhost:8002/codegen/fix"
            fix_data = {
                "code_id": code_id,
                "error_message": "RuntimeError: CIFAR-10 dataset dimension mismatch in Linear layer",
                "error_type": "RuntimeError"
            }
            
            fix_response = requests.post(fix_url, json=fix_data, timeout=120)
            
            if fix_response.status_code == 200:
                fix_result = fix_response.json()
                print("✅ Fix applied successfully!")
                print(f"Message: {fix_result['message']}")
            else:
                print(f"❌ Fix failed: {fix_response.status_code}")
                print(fix_response.text)
                
        else:
            print(f"❌ Code generation failed: {codegen_response.status_code}")
            print(codegen_response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def run_all_fix_tests():
    """Run all fix endpoint tests"""
    
    print("=" * 80)
    print("🧪 RUNNING /codegen/fix ENDPOINT TESTS")
    print("=" * 80)
    
    # Test individual error types
    test_fix_syntax_error()
    time.sleep(2)
    
    test_fix_import_error()
    time.sleep(2)
    
    test_fix_tensor_shape_error()
    time.sleep(2)
    
    test_fix_cuda_error()
    time.sleep(2)
    
    test_fix_missing_code_id()
    time.sleep(2)
    
    # Test full cycle
    test_full_codegen_fix_cycle()
    
    print("\n" + "=" * 80)
    print("🎉 ALL FIX TESTS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    if "--full-cycle" in sys.argv:
        test_full_codegen_fix_cycle()
    else:
        run_all_fix_tests()

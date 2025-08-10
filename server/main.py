from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import asyncio
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables - try multiple paths
load_dotenv()  # Current directory
load_dotenv(dotenv_path="../.env")  # Parent directory
load_dotenv(dotenv_path=".env")  # Root directory

app = FastAPI(title="Agentic Mini-Research MVP - CodeGen Service")

# Pydantic models for request/response
class ExperimentPlan(BaseModel):
    experiment_id: str
    experiment_name: str
    hypothesis: str
    dataset: Dict[str, Any]
    training_parameters: Dict[str, Any]
    evaluation_metrics: List[str]
    models: Dict[str, Any]

class ExecutableCode(BaseModel):
    experiment_id: str
    language: str = "python"
    framework: str = "pytorch"
    code: str
    dependencies: List[str]

class CodegenRequest(BaseModel):
    plan_id: str
    experiment_plan: ExperimentPlan

class CodegenResponse(BaseModel):
    code_id: str
    executable_code: ExecutableCode
    code_file_path: str

class ErrorFeedback(BaseModel):
    code_id: str
    error_message: str
    error_type: str

# In-memory storage (for hackathon speed)
generated_codes = {}
experiment_plans = {}

# Initialize Claude client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

@app.get("/")
async def root():
    return {"message": "Agentic Mini-Research MVP - CodeGen Service"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "codegen"}

@app.post("/codegen", response_model=CodegenResponse)
async def generate_code(request: CodegenRequest):
    """
    Generate executable PyTorch code from experiment plan
    """
    try:
        # Store the experiment plan
        experiment_plans[request.plan_id] = request.experiment_plan
        
        # Generate code using Claude
        generated_code = await generate_pytorch_code(request.experiment_plan)
        
        # Create unique code ID
        code_id = f"code_{uuid.uuid4().hex[:8]}"
        
        # Save generated code to file
        os.makedirs("results", exist_ok=True)
        code_filename = f"results/{request.experiment_plan.experiment_name}_{code_id}.py"
        
        with open(code_filename, 'w') as f:
            f.write(generated_code)
        
        print(f"✅ Code saved to: {code_filename}")
        
        # Create executable code response
        executable_code = ExecutableCode(
            experiment_id=request.experiment_plan.experiment_id,
            code=generated_code,
            dependencies=get_dependencies(request.experiment_plan)
        )
        
        # Store generated code
        generated_codes[code_id] = executable_code
        
        return CodegenResponse(
            code_id=code_id,
            executable_code=executable_code,
            code_file_path=code_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post("/codegen/fix")
async def fix_code(feedback: ErrorFeedback):
    """
    Fix code based on execution feedback
    """
    try:
        if feedback.code_id not in generated_codes:
            raise HTTPException(status_code=404, detail="Code ID not found")
        
        original_code = generated_codes[feedback.code_id]
        
        # Generate fixed code using Claude
        fixed_code = await fix_pytorch_code(
            original_code.code, 
            feedback.error_message,
            feedback.error_type
        )
        
        # Update the stored code
        generated_codes[feedback.code_id].code = fixed_code
        
        return {"message": "Code fixed successfully", "code_id": feedback.code_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code fixing failed: {str(e)}")

async def generate_pytorch_code(experiment_plan: ExperimentPlan) -> str:
    """
    Generate complete PyTorch training code using Claude 4.1
    """
    
    prompt = create_code_generation_prompt(experiment_plan)
    
    try:
        # Using the modern Messages API
        # Using streaming for long operations as required by Anthropic API
        stream = anthropic_client.messages.create(
            model="claude-opus-4-1-20250805",  # Latest Claude model
            max_tokens=32000,
            temperature=0.1,  # Low temperature for more consistent code generation
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        # Collect all chunks from the stream
        response_text = ""
        for chunk in stream:
            if chunk.type == "content_block_delta":
                response_text += chunk.delta.text
        
        return response_text.strip()
        
    except Exception as e:
        raise Exception(f"Claude API error: {str(e)}")

async def fix_pytorch_code(original_code: str, error_message: str, error_type: str) -> str:
    """
    Fix PyTorch code based on execution error feedback
    """
    
    fix_prompt = f"""
You are an expert PyTorch developer. The following code failed during execution:

ORIGINAL CODE:
```python
{original_code}
```

ERROR MESSAGE:
{error_message}

ERROR TYPE: {error_type}

Please provide the COMPLETE FIXED CODE that addresses this error. Make sure:
1. The code is syntactically correct
2. All imports are included
3. The error is properly fixed
4. The code structure remains the same

Return ONLY the complete Python code, no explanations:
"""
    
    try:
        # Using the modern Messages API
        stream = anthropic_client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=32000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": fix_prompt}
            ],
            stream=True
        )
        
        # Collect all chunks from the stream
        response_text = ""
        for chunk in stream:
            if chunk.type == "content_block_delta":
                response_text += chunk.delta.text
        
        return response_text.strip()
        
    except Exception as e:
        raise Exception(f"Claude API error during fix: {str(e)}")

def create_code_generation_prompt(experiment_plan: ExperimentPlan) -> str:
    """
    Create a detailed prompt for Claude to generate PyTorch code
    """
    
    models_description = ""
    
    # Process candidate model
    candidate = experiment_plan.models.get("candidate", {})
    if candidate:
        models_description += f"\nCANDIDATE MODEL ({candidate.get('model_id', 'candidate')}):\n"
        models_description += f"Description: {candidate.get('description', '')}\n"
        models_description += "Architecture:\n"
        for i, layer in enumerate(candidate.get('architecture', [])):
            models_description += f"  {i+1}. {layer['type']}: {layer.get('params', {})}\n"
    
    # Process baseline models
    baselines = experiment_plan.models.get("baselines", [])
    for baseline in baselines:
        models_description += f"\nBASELINE MODEL ({baseline.get('model_id', 'baseline')}):\n"
        models_description += f"Description: {baseline.get('description', '')}\n"
        models_description += "Architecture:\n"
        for i, layer in enumerate(baseline.get('architecture', [])):
            models_description += f"  {i+1}. {layer['type']}: {layer.get('params', {})}\n"
    
    prompt = f"""
You are an expert PyTorch developer. Generate COMPLETE, EXECUTABLE PyTorch code for the following machine learning experiment:

EXPERIMENT: {experiment_plan.experiment_name}
HYPOTHESIS: {experiment_plan.hypothesis}

DATASET:
- Name: {experiment_plan.dataset['name']}
- Source: {experiment_plan.dataset['source']}
- Validation Split: {experiment_plan.dataset.get('validation_split', 0.2)}

TRAINING PARAMETERS:
- Optimizer: {experiment_plan.training_parameters['optimizer']}
- Learning Rate: {experiment_plan.training_parameters['learning_rate']}
- Loss Function: {experiment_plan.training_parameters['loss_function']}
- Batch Size: {experiment_plan.training_parameters['batch_size']}
- Epochs: {experiment_plan.training_parameters['epochs']}

EVALUATION METRICS: {', '.join(experiment_plan.evaluation_metrics)}

MODELS TO IMPLEMENT:
{models_description}

REQUIREMENTS:
1. Generate a COMPLETE Python script that can be executed directly
2. Include ALL necessary imports
3. Implement ALL models as PyTorch nn.Module classes
4. Include data loading and preprocessing for CIFAR-10
5. Implement training loop with proper loss calculation
6. Implement evaluation with all specified metrics (accuracy, precision, recall, f1_score)
7. Train and evaluate ALL models (candidate + baselines)
8. Print results in a clear format
9. Handle DepthwiseSeparableConv2D by implementing it as a combination of depthwise and pointwise convolutions
10. Use proper activation functions (GELU, ReLU) as specified
11. Save model checkpoints and results

SPECIAL INSTRUCTIONS:
- For DepthwiseSeparableConv2D: Implement as nn.Sequential with depthwise conv + pointwise conv
- Use proper padding and stride calculations
- Include proper error handling
- Make sure all tensor dimensions are compatible
- Use device-agnostic code (CPU/GPU compatible)

Return ONLY the complete Python code, no explanations or markdown formatting:
"""
    
    return prompt

def get_dependencies(experiment_plan: ExperimentPlan) -> List[str]:
    """
    Determine required dependencies based on the experiment plan
    """
    base_deps = [
        "torch", 
        "torchvision", 
        "scikit-learn",
        "numpy",
        "matplotlib"
    ]
    
    # Add additional dependencies based on requirements
    if any("f1_score" in metric for metric in experiment_plan.evaluation_metrics):
        if "scikit-learn" not in base_deps:
            base_deps.append("scikit-learn")
    
    return base_deps

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

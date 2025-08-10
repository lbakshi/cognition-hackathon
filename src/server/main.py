from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import asyncio
import redis.asyncio as redis
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from src.endpoints import add_polling_and_subagent_endpoints

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Redis connection
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at: {redis_url}")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        # Test the connection
        await redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        raise
    # Add polling and subagent endpoints
    add_polling_and_subagent_endpoints(app, redis_client)
    
@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

# Allow cross-origin requests from any domain so the frontend can
# be hosted separately (e.g. on a different Railway service).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartExperimentRequest(BaseModel):
    query: str

    class Config:
        schema_extra = {
            "example": {
                "query": "Create a machine learning experiment for image classification"
            }
        }

class SubagentRequest(BaseModel):
    experiment_id: str
    agent_type: str
    data: Dict[str, Any]
    
class FileStoreRequest(BaseModel):
    experiment_id: str
    filename: str
    content: str
    
class KVUpdateRequest(BaseModel):
    experiment_id: str
    key: str
    value: Any

async def run_orchestrator_background(experiment_id: str, query: str):
    """Run orchestrator_prod in background and update Redis with progress"""
    try:
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "running",
            "stage": "conceptualization",
            "started_at": datetime.now().isoformat(),
            "query": query
        })
        
        # Import and run orchestrator logic
        # For now, simulate the process - you'll need to adapt orchestrator_prod.py
        import time
        
        # Simulate conceptualization
        await redis_client.hset(f"experiment:{experiment_id}", "stage", "conceptualization")
        await redis_client.hset(f"experiment:{experiment_id}:plan", "status", "in_progress")
        await asyncio.sleep(2)  # Simulate work
        
        # Simulate strategy
        await redis_client.hset(f"experiment:{experiment_id}", "stage", "strategy")
        await redis_client.hset(f"experiment:{experiment_id}:plan", "status", "completed")
        await redis_client.hset(f"experiment:{experiment_id}:plan", "result", json.dumps({"plan": "ML experiment plan"}))
        await asyncio.sleep(2)
        
        # Simulate code generation
        await redis_client.hset(f"experiment:{experiment_id}", "stage", "codegen")
        await redis_client.hset(f"experiment:{experiment_id}:codegen", "status", "in_progress")
        await asyncio.sleep(3)
        
        await redis_client.hset(f"experiment:{experiment_id}:codegen", "status", "completed")
        await redis_client.hset(f"experiment:{experiment_id}:codegen", "files", json.dumps({
            "main.py": "# Generated experiment code\nprint('Experiment running')",
            "requirements.txt": "numpy\npandas\nscikit-learn"
        }))
        
        # Complete
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "completed",
            "stage": "finished",
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "error",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

@app.get("/api/hello")
async def read_hello():
    return {"message": "Hello from server!"}

@app.post("/api/start")
async def start_experiment(request: StartExperimentRequest, background_tasks: BackgroundTasks):
    """Start a new experiment and return experiment_id"""
    experiment_id = str(uuid.uuid4())
    
    # Initialize experiment in Redis
    await redis_client.hset(f"experiment:{experiment_id}", mapping={
        "status": "initializing",
        "query": request.query,
        "created_at": datetime.now().isoformat()
    })
    
    # Start orchestrator in background
    background_tasks.add_task(run_orchestrator_background, experiment_id, request.query)
    
    return {"experiment_id": experiment_id}





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)

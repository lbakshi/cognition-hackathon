"""
Background orchestrator runner that uses the real orchestrator
"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

async def run_orchestrator_background(experiment_id: str, query: str, redis_client):
    """Run orchestrator in background and update Redis with progress"""
    try:
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "running",
            "stage": "initializing",
            "started_at": datetime.now().isoformat(),
            "query": query
        })
        
        # Check for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise Exception("ANTHROPIC_API_KEY not found in environment. Cannot run orchestrator.")
        
        # Use the real orchestrator with API communication
        print(f"[{experiment_id}] Starting orchestrator with API communication")
        from src.orchestrator_api import run_orchestrator_with_api
        await run_orchestrator_with_api(experiment_id, query)
        
        # Mark as completed (orchestrator will update stages via API)
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "completed",
            "stage": "finished",
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[{experiment_id}] Error: {str(e)}")
        await redis_client.hset(f"experiment:{experiment_id}", mapping={
            "status": "error",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
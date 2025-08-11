# Additional endpoints for polling and subagent communication
from fastapi import HTTPException, Body
import json
import redis.asyncio as redis
from typing import Dict, Any

def add_polling_and_subagent_endpoints(app, redis_client):
    # Polling endpoints for experiment status
    @app.get("/api/poll/{experiment_id}/status")
    async def poll_experiment_status(experiment_id: str):
        """Get overall experiment status"""
        experiment_data = await redis_client.hgetall(f"experiment:{experiment_id}")
        if not experiment_data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment_data

    @app.get("/api/poll/{experiment_id}/plan")
    async def poll_plan_status(experiment_id: str):
        """Get planning stage status and results"""
        plan_data = await redis_client.hgetall(f"experiment:{experiment_id}:plan")
        if not plan_data:
            return {"status": "not_started"}
        
        if "result" in plan_data:
            plan_data["result"] = json.loads(plan_data["result"])
        return plan_data

    @app.get("/api/poll/{experiment_id}/codegen")
    async def poll_codegen_status(experiment_id: str):
        """Get code generation status and results"""
        codegen_data = await redis_client.hgetall(f"experiment:{experiment_id}:codegen")
        if not codegen_data:
            return {"status": "not_started"}
            
        if "files" in codegen_data:
            codegen_data["files"] = json.loads(codegen_data["files"])
        return codegen_data

    @app.get("/api/poll/{experiment_id}/execution")
    async def poll_execution_status(experiment_id: str):
        """Get execution status and results"""
        execution_data = await redis_client.hgetall(f"experiment:{experiment_id}:execution")
        if not execution_data:
            return {"status": "not_started"}
            
        if "results" in execution_data:
            execution_data["results"] = json.loads(execution_data["results"])
        return execution_data

    # Subagent endpoints that orchestrator can call
    @app.post("/api/subagent/store_file")
    async def store_file(request: Dict[str, Any] = Body(...)):
        """Store a file for an experiment"""
        key = f"experiment:{request['experiment_id']}:files:{request['filename']}"
        await redis_client.set(key, request['content'])
        
        # Update file list
        files_list_key = f"experiment:{request['experiment_id']}:file_list"
        await redis_client.sadd(files_list_key, request['filename'])
        
        return {"status": "stored", "filename": request['filename']}

    @app.post("/api/subagent/update_kv")
    async def update_key_value(request: Dict[str, Any] = Body(...)):
        """Update a key-value pair for an experiment"""
        key = f"experiment:{request['experiment_id']}:kv:{request['key']}"
        value = json.dumps(request['value']) if not isinstance(request['value'], str) else request['value']
        await redis_client.set(key, value)
        return {"status": "updated", "key": request['key']}

    @app.get("/api/subagent/get_kv/{experiment_id}/{key}")
    async def get_key_value(experiment_id: str, key: str):
        """Get a key-value pair for an experiment"""
        redis_key = f"experiment:{experiment_id}:kv:{key}"
        value = await redis_client.get(redis_key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string if not JSON
            
        return {"key": key, "value": value}

    @app.post("/api/subagent/update_stage")
    async def update_experiment_stage(request: Dict[str, Any] = Body(...)):
        """Allow orchestrator to update experiment stage"""
        await redis_client.hset(f"experiment:{request['experiment_id']}", 
                               f"{request['agent_type']}_status", json.dumps(request['data']))
        return {"status": "updated"}

    @app.get("/api/experiments/{experiment_id}/files")
    async def get_experiment_files(experiment_id: str):
        """Get list of files for an experiment"""
        files_list_key = f"experiment:{experiment_id}:file_list"
        files = await redis_client.smembers(files_list_key)
        return {"files": list(files)}

    @app.get("/api/experiments/{experiment_id}/files/{filename}")
    async def get_experiment_file(experiment_id: str, filename: str):
        """Get specific file content for an experiment"""
        key = f"experiment:{experiment_id}:files:{filename}"
        content = await redis_client.get(key)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        return {"filename": filename, "content": content}
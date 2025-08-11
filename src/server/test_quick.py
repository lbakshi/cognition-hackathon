#!/usr/bin/env python
"""
Quick test to verify the API is working without waiting for full completion
"""
import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8000"

async def quick_test():
    """Start an experiment and check initial status"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check server
        try:
            response = await client.get(f"{BASE_URL}/api/hello")
            print(f"✅ Server is running: {response.json()}")
        except:
            print("❌ Cannot connect to server")
            sys.exit(1)
        
        # Start experiment
        response = await client.post(
            f"{BASE_URL}/api/start",
            json={"query": "Test experiment"}
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start: {response.status_code}")
            print(response.text)
            sys.exit(1)
            
        data = response.json()
        experiment_id = data["experiment_id"]
        print(f"✅ Started experiment: {experiment_id}")
        
        # Check status a few times
        for i in range(5):
            await asyncio.sleep(2)
            response = await client.get(f"{BASE_URL}/api/poll/{experiment_id}/status")
            status_data = response.json()
            print(f"   Stage: {status_data.get('stage', 'unknown')} | Status: {status_data.get('status', 'unknown')}")
        
        print("\n✅ API is working! Experiment continues in background.")
        print(f"   Check progress: curl http://localhost:8000/api/poll/{experiment_id}/status")
        print(f"   Cancel if needed: curl -X POST http://localhost:8000/api/cancel/{experiment_id}")

if __name__ == "__main__":
    asyncio.run(quick_test())
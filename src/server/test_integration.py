#!/usr/bin/env python
"""
Integration test script that cycles through the experiment API
"""
import asyncio
import httpx
import json
import sys
from datetime import datetime
import argparse

BASE_URL = "http://localhost:8000"

async def start_experiment(query: str):
    """Start a new experiment"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/start", json={"query": query})
        if response.status_code != 200:
            print(f"❌ Failed to start experiment: {response.status_code}")
            print(f"   Response: {response.text}")
            raise Exception(f"Server error: {response.status_code}")
        data = response.json()
        experiment_id = data["experiment_id"]
        print(f"✅ Started experiment: {experiment_id}")
        print(f"   Query: {query}")
        return experiment_id

async def poll_status(experiment_id: str, interval: float = 2.0, max_polls: int = 600):
    """Poll experiment status and display updates"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        last_stage = None
        polls = 0
        
        while polls < max_polls:
            # Poll overall status
            response = await client.get(f"{BASE_URL}/api/poll/{experiment_id}/status")
            status_data = response.json()
            
            current_stage = status_data.get("stage", "unknown")
            status = status_data.get("status", "unknown")
            
            # Only print if stage changed
            if current_stage != last_stage:
                print(f"\n📊 Stage: {current_stage} | Status: {status}")
                last_stage = current_stage
            
            # Poll specific endpoints based on stage
            if current_stage in ["conceptualization", "strategy"]:
                plan_response = await client.get(f"{BASE_URL}/api/poll/{experiment_id}/plan")
                plan_data = plan_response.json()
                if plan_data.get("status") == "completed":
                    print(f"   ✓ Plan ready: {json.dumps(plan_data.get('result', {}), indent=2)[:200]}...")
            
            elif current_stage == "codegen":
                codegen_response = await client.get(f"{BASE_URL}/api/poll/{experiment_id}/codegen")
                codegen_data = codegen_response.json()
                if codegen_data.get("status") == "completed":
                    files = codegen_data.get("files", {})
                    print(f"   ✓ Generated {len(files)} files: {list(files.keys())}")
            
            # Check if completed or errored
            if status in ["completed", "error", "cancelled"]:
                if status == "completed":
                    print(f"\n✅ Experiment completed successfully!")
                elif status == "cancelled":
                    print(f"\n⚠️  Experiment was cancelled")
                else:
                    error = status_data.get("error", "Unknown error")
                    print(f"\n❌ Experiment failed: {error}")
                return status
            
            await asyncio.sleep(interval)
            polls += 1
            
            # Print progress indicator
            if polls % 10 == 0 and polls > 0:
                elapsed = polls * interval
                print(f"\n⏱️  Elapsed: {elapsed:.0f}s - Still running...", end="", flush=True)
        
        print(f"\n⏱️  Timeout: Experiment still running after {max_polls} polls")
        return "timeout"

async def cancel_experiment(experiment_id: str):
    """Cancel a running experiment"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/cancel/{experiment_id}")
        data = response.json()
        print(f"🛑 Cancelled experiment: {experiment_id}")
        return data

async def run_full_cycle(query: str, cancel_after: int = None):
    """Run a full experiment cycle"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Integration Test")
    print(f"   Time: {datetime.now().isoformat()}")
    print(f"{'='*60}")
    
    # Start experiment
    experiment_id = await start_experiment(query)
    
    # If cancel_after is set, schedule cancellation
    if cancel_after:
        print(f"⏰ Will cancel after {cancel_after} seconds...")
        asyncio.create_task(cancel_after_delay(experiment_id, cancel_after))
    
    # Poll for status
    final_status = await poll_status(experiment_id)
    
    print(f"\n{'='*60}")
    print(f"📋 Test Complete")
    print(f"   Final Status: {final_status}")
    print(f"   Experiment ID: {experiment_id}")
    print(f"{'='*60}\n")
    
    return experiment_id, final_status

async def cancel_after_delay(experiment_id: str, delay: int):
    """Cancel an experiment after a delay"""
    await asyncio.sleep(delay)
    await cancel_experiment(experiment_id)

async def run_multiple_experiments():
    """Run multiple experiments concurrently"""
    queries = [
        "Create a CNN for image classification",
        "Build an LSTM for text generation",
        "Implement a GAN for image synthesis"
    ]
    
    print(f"\n🔬 Running {len(queries)} experiments concurrently...")
    
    tasks = [run_full_cycle(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    print("\n📊 Summary of all experiments:")
    for (exp_id, status), query in zip(results, queries):
        print(f"   • {exp_id[:8]}... | {status:10} | {query}")

async def main():
    parser = argparse.ArgumentParser(description="Integration test for experiment API")
    parser.add_argument("--query", default="Create a neural network for MNIST", help="Experiment query")
    parser.add_argument("--cancel-after", type=int, help="Cancel experiment after N seconds")
    parser.add_argument("--multiple", action="store_true", help="Run multiple experiments")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    try:
        # Check if server is running
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/hello")
            if response.status_code != 200:
                print("❌ Server is not responding. Please start the server first.")
                sys.exit(1)
    except httpx.ConnectError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Please run: make dev-server")
        sys.exit(1)
    
    if args.multiple:
        await run_multiple_experiments()
    else:
        await run_full_cycle(args.query, args.cancel_after)

if __name__ == "__main__":
    asyncio.run(main())
6-Hour Hackathon PRD — Agentic Mini-Research MVP
Goal
Given a short natural-language research intent, the system:

Plans the experiment (metrics, baselines, dataset, training params).

Generates runnable PyTorch code.

Runs the experiment on Modal.

Returns metrics + plots + short research summary.

One local FastAPI server orchestrates everything; the science runs on Modal.

Scope (MVP)
In scope:

One dataset (CIFAR-10).

Fixed 3 metrics: accuracy, F1-score, loss curve.

2 baselines: basic CNN, ResNet18.

Training params fixed to 1–3 epochs (demo speed).

LLMs: GPT-5, Claude 4.1 (pick one per call).

Modal for execution (CPU; GPU optional if time allows).

Out of scope:

Dataset catalog, UI, auth, experiment registry.

Full error handling beyond basic logging.

Multi-job concurrency.

Architecture Overview
pgsql
Copy
Edit
[Researcher Input] 
     ↓
 /plan   → Planning Agent (LLM) → ExperimentSpec JSON
     ↓
 /codegen → Codegen Agent (LLM) → train.py, eval.py, model files
     ↓
 /execute → Modal run → metrics.json, plots
     ↓
 /report  → Reporting Agent (LLM) → 3–5 sentence summary
Local FastAPI Server — Orchestrator, LLM calls, job state.

Modal Runtime — Runs generated training/eval code with CIFAR-10.

Storage — Modal Volume (mounted by server) or local disk for artifacts.

Endpoints
1. POST /plan
Req:

json
Copy
Edit
{ "intent": "test CNN with GELU + depthwise separable convs" }
Resp:

json
Copy
Edit
{
  "plan_id": "pln1",
  "spec": {
    "model": {"type": "cnn_custom", "activations": "gelu", "layers": ["dw_sep_conv"]},
    "metrics": ["accuracy","f1"],
    "baselines": ["cnn_basic","resnet18"],
    "dataset": "cifar10",
    "train": {"epochs": 3, "batch_size": 128, "lr": 0.001}
  }
}
2. POST /codegen
Req:

json
Copy
Edit
{ "plan_id": "pln1" }
Resp:

json
Copy
Edit
{
  "code_id": "c1",
  "files": [
    {"path": "models/custom_cnn.py", "content": "<...>"},
    {"path": "train.py", "content": "<...>"},
    {"path": "eval.py", "content": "<...>"}
  ]
}
3. POST /execute
Req:

json
Copy
Edit
{ "plan_id": "pln1", "code_id": "c1" }
Resp:

json
Copy
Edit
{ "job_id": "j1", "status": "running" }
Execution steps:

Write spec.json + code files to Modal Volume.

Trigger Modal function:

Loads CIFAR-10 (download if needed).

Trains custom model + baselines for N epochs.

Saves metrics.json, loss.png, acc.png.

4. POST /report
Req:

json
Copy
Edit
{ "job_id": "j1" }
Resp:

json
Copy
Edit
{
  "summary": "The GELU + DW sep CNN achieved 78% accuracy...",
  "bullets": [
    "Custom model outperformed basic CNN by 4%.",
    "ResNet18 still led overall."
  ]
}
Data Model (in-memory for speed)
python
Copy
Edit
plans = { plan_id: spec_dict }
codes  = { code_id: [ {path, content}, ... ] }
jobs   = { job_id: {status, metrics, plots} }
6-Hour Execution Plan
Hour 0–1 — Infra Engineer
Scaffold FastAPI app with /plan, /codegen, /execute, /report.

Set up Modal project + minimal PyTorch runtime (torch, torchvision, matplotlib, scikit-learn).

Hour 1–2 — CS Research Scientist
Provide tiny PyTorch scripts for:

CIFAR-10 data loader.

cnn_basic.py, resnet18.py.

Metrics calculation (accuracy, F1).

Plotting loss/accuracy curves.

Hour 2–3 — AI Integrations / Fullstack
Build LLM prompts for /plan → ExperimentSpec.

Build /codegen prompt that inserts GELU + DW sep conv in template code.

Hour 3–4 — Infra Engineer
Implement /execute to:

Write code/spec to Modal Volume.

Call Modal function to run training + save outputs.

Ensure results saved as metrics.json + .png plots in Volume.

Hour 4–5 — AI Integrations
Implement /report:

Read metrics.json.

Call LLM to summarize findings in 3–5 sentences.

Hour 5–6 — All
End-to-end run from /plan → /report.

Shorten epochs to 1–2 if needed for speed.

Demo: show generated plots + summary in terminal or Postman.

Role Breakdown
CS Research Scientist

Write baseline models + metrics functions + plotting code.

Validate generated code correctness.

Infra Engineer

FastAPI scaffold + endpoint wiring.

Modal runtime build + execute orchestration.

AI Integrations / Fullstack

LLM prompt engineering for planning/codegen/report.

JSON parsing + structuring responses for the server.

Acceptance Criteria
/plan → returns a structured spec.

/codegen → returns runnable PyTorch scripts.

/execute → runs on Modal, returns metrics + plots in < 10 min.

/report → returns coherent research summary.
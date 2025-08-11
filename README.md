# Cognition Research - AI-Powered ML Experiment Platform

An autonomous research platform that designs, implements, executes, and analyzes machine learning experiments from natural language descriptions. Built for the 6-hour Cognition hackathon challenge.

## 🎯 Purpose

Cognition Research automates the entire machine learning research pipeline:

1. **Input**: Researcher describes an experiment in natural language
2. **Planning**: AI agents design the experiment architecture, select metrics, and choose baselines
3. **Implementation**: AI generates runnable PyTorch code tailored to the experiment
4. **Execution**: Code runs on Modal's serverless infrastructure with GPU support
5. **Analysis**: AI analyzes results and generates research summaries with visualizations

## 🏗️ Architecture

### Multi-Agent System
The platform uses specialized AI agents orchestrated through a central controller:

- **Conceptualization Agent** (`src/server/src/agents/conceptualization_agent.py`) - Understands research intent and domain
- **Strategy Agent** (`src/server/src/agents/strategy_agent.py`) - Plans experiment architecture and validation
- **Implementation Agent** (`src/server/src/agents/implementation_agent.py`) - Generates PyTorch code
- **Experimentation Agent** (`src/server/src/agents/experimentation_agent.py`) - Executes experiments on Modal
- **Analysis Agent** (`src/server/src/agents/analysis_agent.py`) - Analyzes results and generates reports

### Infrastructure Components

**Backend Server** (`src/server/`)
- FastAPI server with Redis for state management
- Orchestrator for coordinating multi-agent workflows
- RESTful API endpoints for experiment lifecycle management
- Background task processing for long-running experiments

**Frontend Interface** (`src/frontend/`)
- Next.js 15 web application with TypeScript
- Real-time experiment progress tracking
- Interactive results dashboard with metrics visualization
- Modern UI with Tailwind CSS and glass morphism design

**Cloud Execution** (`src/server/src/modal/`)
- Modal integration for serverless ML experiment execution
- GPU/CPU resource management
- Persistent volume storage for artifacts and datasets
- Timeout and memory configuration for different experiment types

## 📊 Features

### Experiment Design
- Natural language to structured experiment specification
- Automatic baseline model selection (CNN, ResNet18)
- CIFAR-10 dataset with configurable training parameters
- Fixed metrics: accuracy, F1-score, loss curves

### Code Generation
- Template-based PyTorch code generation
- Support for custom architectures (GELU activations, depthwise separable convolutions)
- Automatic training and evaluation script creation
- Error handling and retry mechanisms

### Execution Pipeline
- Serverless execution on Modal infrastructure
- Real-time progress tracking and status updates
- Artifact storage and retrieval
- Comprehensive error logging and debugging

### Results Analysis
- Automated metrics comparison between models
- Visualization generation (loss curves, accuracy plots)
- AI-generated research summaries
- Exportable results and code artifacts

## 🚀 API Endpoints

### Core Experiment Flow
- `POST /api/start` - Initialize new experiment with natural language query
- `GET /api/poll/{experiment_id}/status` - Get overall experiment status
- `GET /api/poll/{experiment_id}/plan` - Get planning stage results
- `GET /api/poll/{experiment_id}/codegen` - Get code generation results
- `GET /api/poll/{experiment_id}/execution` - Get execution results

### File Management
- `GET /api/experiments/{experiment_id}/files` - List experiment artifacts
- `GET /api/experiments/{experiment_id}/files/{filename}` - Download specific files
- `POST /api/subagent/store_file` - Store experiment artifacts
- `POST /api/subagent/update_kv` - Update experiment metadata

## 🛠️ Technology Stack

### AI & ML
- **LLMs**: Claude 4.1 for all agent reasoning and code generation
- **ML Framework**: PyTorch with torchvision for model implementations
- **Metrics**: scikit-learn for evaluation metrics

### Backend
- **Framework**: FastAPI with async support
- **Database**: Redis for experiment state and caching
- **Cloud**: Modal for serverless ML execution
- **Task Queue**: Background task processing

### Frontend
- **Framework**: Next.js 15 with React
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS with custom glass morphism components
- **Charts**: Recharts for metrics visualization

### Infrastructure
- **Deployment**: Railway for web services
- **Container**: Docker with multi-stage builds
- **Environment**: Python 3.11+ with Poetry dependency management

## 📁 Project Structure

```
/
├── src/
│   ├── frontend/                 # Next.js web application
│   │   ├── src/
│   │   │   ├── app/             # App router pages
│   │   │   ├── components/      # React components
│   │   │   └── types/           # TypeScript definitions
│   │   └── package.json
│   │
│   ├── server/                   # FastAPI backend
│   │   ├── src/
│   │   │   ├── agents/          # AI agent implementations
│   │   │   ├── modal/           # Modal cloud integration
│   │   │   ├── endpoints.py     # API endpoint definitions
│   │   │   └── orchestrator.py  # Multi-agent coordination
│   │   ├── main.py              # FastAPI app entry point
│   │   └── pyproject.toml       # Poetry dependencies
│   │
│   └── modal_execution.py        # Modal runner script
│
├── results/                      # Generated experiment results
├── scripts/                      # Utility scripts and examples
├── infra/                        # Infrastructure configuration
├── PRD.md                       # Product Requirements Document
└── README.md                    # This file
```

## 🎯 Scope & Limitations (MVP)

**In Scope:**
- Single dataset (CIFAR-10)
- Fixed metrics (accuracy, F1-score, loss curves)
- Two baseline models (basic CNN, ResNet18)
- Fast training (1-3 epochs for demo speed)
- CPU/GPU execution on Modal

**Out of Scope:**
- Multi-dataset catalog
- User authentication
- Experiment registry/history
- Multi-job concurrency
- Production error handling

## 📝 Example Usage

```bash
# Natural language input
"Test a CNN with GELU activations and depthwise separable convolutions"

# System automatically:
# 1. Plans experiment with CIFAR-10, accuracy/F1 metrics, CNN/ResNet18 baselines
# 2. Generates custom PyTorch model code with GELU + depthwise separable layers
# 3. Executes training on Modal infrastructure
# 4. Compares results: "Custom model achieved 78% accuracy, 4% better than basic CNN"
```

## 🚧 Development Status

This is a hackathon MVP demonstrating the core concept of autonomous ML research. The platform successfully orchestrates multi-agent workflows to transform natural language research ideas into executable experiments with automated analysis and reporting.

---

**Built with ❤️ during the Cognition hackathon**  
*Powered by Next.js 15, Claude 4.1, and Modal serverless infrastructure*
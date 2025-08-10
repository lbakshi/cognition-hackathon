.PHONY: help install notebook run-notebook execute test clean modal-auth jupyter-kernel modal-run modal-test server frontend dev-local stop-local

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies with Poetry"
	@echo "  make notebook       - Start Jupyter notebook server"
	@echo "  make run-notebook   - Execute the Modal execution notebook"
	@echo "  make execute        - Run a sample experiment"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make modal-auth    - Authenticate with Modal"
	@echo "  make modal-run     - Run Modal execution service"
	@echo "  make modal-test    - Test Modal execution with sample experiment"
	@echo "  make jupyter-kernel - Install Jupyter kernel for Poetry environment"
	@echo ""
	@echo "Local Development:"
	@echo "  make server        - Start the FastAPI backend server (port 8000)"
	@echo "  make frontend      - Start the Express.js frontend server (port 3000)"
	@echo "  make dev-local     - Start both server and frontend in background"

# Install dependencies
install:
	@echo "Installing dependencies with Poetry..."
	poetry install --no-root
	@echo "Installing Jupyter kernel..."
	poetry run python -m ipykernel install --user --name cognition-hackathon --display-name "Python (cognition-hackathon)"
	@echo "✓ Installation complete!"

# Start Jupyter notebook server
notebook:
	@echo "Starting Jupyter Notebook..."
	@echo "Remember to select kernel: Python (cognition-hackathon)"
	poetry run jupyter notebook scripts/modal_execution_runtime.ipynb

# Execute the notebook programmatically
run-notebook:
	@echo "Executing Modal execution notebook..."
	@poetry run jupyter nbconvert --to notebook --execute scripts/modal_execution_runtime.ipynb --output modal_execution_runtime_executed.ipynb --ExecutePreprocessor.kernel_name=cognition-hackathon
	@echo "✓ Notebook execution complete! Results saved to modal_execution_runtime_executed.ipynb"

# Run a sample experiment
execute:
	@echo "Running sample experiment..."
	poetry run python -c "import asyncio; import sys; sys.path.append('.'); \
		from scripts.sample.execution import sample_experiment; \
		print('Starting experiment execution...'); \
		# Add your experiment execution code here"

# Run tests - execute the Modal execution notebook
test:
	@echo "Executing Modal execution runtime notebook..."
	@poetry run jupyter nbconvert --to notebook --execute scripts/modal_execution_runtime.ipynb --output modal_execution_runtime_executed.ipynb --ExecutePreprocessor.kernel_name=cognition-hackathon
	@echo "✓ Notebook execution complete!"

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	@echo "✓ Cleanup complete!"

# Authenticate with Modal
modal-auth:
	@echo "Authenticating with Modal..."
	@echo "This will open a browser window for authentication"
	poetry run modal token new
	@echo "✓ Modal authentication complete!"

# Install Jupyter kernel
jupyter-kernel:
	@echo "Installing Jupyter kernel for Poetry environment..."
	poetry run python -m ipykernel install --user --name cognition-hackathon --display-name "Python (cognition-hackathon)"
	@echo "✓ Kernel installed!"

# Development shortcuts
dev: install notebook

# Quick execution with Modal
quick-run:
	@echo "Quick execution of Modal notebook cells..."
	poetry run python -c "\
		import nbformat; \
		from nbconvert.preprocessors import ExecutePreprocessor; \
		import os; \
		os.environ['MODAL_RUN_MODE'] = 'test'; \
		nb = nbformat.read('modal_execution_runtime.ipynb', as_version=4); \
		ep = ExecutePreprocessor(timeout=600, kernel_name='cognition-hackathon'); \
		ep.preprocess(nb, {'metadata': {'path': '.'}}); \
		print('✓ Execution complete!')"

# Run notebook in background
notebook-bg:
	@echo "Starting Jupyter notebook in background..."
	nohup poetry run jupyter notebook modal_execution_runtime.ipynb > jupyter.log 2>&1 &
	@echo "✓ Jupyter started in background. Check jupyter.log for details"
	@echo "To stop: pkill -f jupyter"

# Convert notebook to Python script
notebook-to-py:
	@echo "Converting notebook to Python script..."
	poetry run jupyter nbconvert --to python scripts/modal_execution_runtime.ipynb
	@echo "✓ Converted to scripts/modal_execution_runtime.py"

# Run specific test experiment
test-experiment:
	@echo "Running test experiment validation..."
	@poetry run python test_experiment.py

# Check Modal status
modal-status:
	@echo "Checking Modal connection status..."
	poetry run modal --version
	poetry run python -c "import modal; print('✓ Modal imported successfully')"
	@echo "Checking for Modal token..."
	@if [ -n "$$MODAL_TOKEN_ID" ]; then \
		echo "✓ Modal token found in environment"; \
	else \
		echo "⚠ No Modal token in environment. Run 'make modal-auth' to authenticate"; \
	fi

# Run Modal execution service
modal-run:
	@echo "Starting Modal execution service..."
	@echo "This will spin up the Modal runtime for ML experiments"
	poetry run python src/modal_execution.py

# Test Modal execution with sample experiment
modal-test:
	@echo "Testing Modal execution with sample experiment..."
	@echo "Submitting test experiment to Modal runtime..."
	poetry run modal run modal_online.py

# Local Development Commands
server:
	@echo "Starting FastAPI backend server on port 8000..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	cd src/server && poetry run python main.py

frontend:
	@echo "Starting Express.js frontend server on port 3000..."
	@echo "Frontend will be available at: http://localhost:3000"
	@echo "Press Ctrl+C to stop"
	cd src/frontend && npm start

dev-local:
	@echo "Starting both server and frontend in background..."
	@echo "Backend API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo ""
	@echo "Starting backend server..."
	@cd src/server && nohup poetry run python main.py > ../../server.log 2>&1 &
	@echo "Backend started with PID: $$(pgrep -f 'python main.py')"
	@echo ""
	@echo "Starting frontend server..."
	@cd src/frontend && nohup npm start > ../../frontend.log 2>&1 &
	@echo "Frontend started with PID: $$(pgrep -f 'node server.js')"
	@echo ""
	@echo "✓ Both services are running in background!"
	@echo "Check server.log and frontend.log for output"
	@echo "To stop both: make stop-local"

stop-local:
	@echo "Stopping local development services..."
	@pkill -f "python main.py" || true
	@pkill -f "node server.js" || true
	@echo "✓ Services stopped"
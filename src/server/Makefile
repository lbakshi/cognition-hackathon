.PHONY: dev-server test test-cancel test-multiple redis-start redis-stop clean help

# Server commands
dev-server:
	poetry run python main.py

dev-server-verbose:
	PYTHONUNBUFFERED=1 poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# Integration tests
test: ## Run a single integration test (waits for completion)
	poetry run python test_integration.py --query "Create a machine learning model for sentiment analysis"

test-quick: ## Quick API test without waiting for completion
	poetry run python test_quick.py

test-cancel: ## Test cancellation functionality
	poetry run python test_integration.py --query "Build a complex deep learning pipeline" --cancel-after 3

test-multiple: ## Run multiple experiments concurrently
	poetry run python test_integration.py --multiple

test-watch: ## Run continuous test with live output
	@echo "Starting integration test with live monitoring..."
	@echo "Server logs will appear here. Press Ctrl+C to stop."
	@poetry run python test_integration.py --query "Implement a transformer model" --interval 0.5

# Redis commands
redis-start: ## Start Redis in Docker
	docker run -d --name redis-hackathon -p 6379:6379 redis:alpine

redis-stop: ## Stop Redis container
	docker stop redis-hackathon && docker rm redis-hackathon

redis-cli: ## Connect to Redis CLI
	docker exec -it redis-hackathon redis-cli

# Utility commands
logs: ## Tail server logs
	tail -f *.log 2>/dev/null || echo "No log files found"

clean: ## Clean up cache and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f *.log

install: ## Install dependencies
	poetry install

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
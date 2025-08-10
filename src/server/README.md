# Server

Simple FastAPI server for Railway or local deployments.

## Development

```bash
poetry install
poetry run uvicorn server.main:app --reload
```

The server listens on the `PORT` environment variable (defaults to `3000`).

## Deployment

Use a standard Python environment with the start command:

```
poetry run uvicorn server.main:app
```

# Server

Simple FastAPI server for Railway or local deployments.

## Development

```bash
poetry install
poetry run uvicorn server.main:app --reload
```

The server listens on the `PORT` environment variable (defaults to `3000`).
It has permissive CORS settings so that a frontend deployed on another
service (or domain) can communicate with it.

## Deployment

Use a standard Python environment with the start command:

```
poetry run uvicorn server.main:app
```
